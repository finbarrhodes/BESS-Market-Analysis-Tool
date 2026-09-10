"""
National Energy System Operator (NESO) Data Collection Module

Collects data from the NESO Data Portal, which exposes a CKAN-based API.

CKAN (Comprehensive Knowledge Archive Network) is an open-source data
catalogue platform.  NESO uses it to publish energy datasets.  The two
main query mechanisms are:

  1. ``datastore_search`` — filter/sort/paginate over a single resource.
  2. ``datastore_search_sql`` — arbitrary SELECT queries (PostgreSQL
     dialect) against one or more resources.

We use SQL queries so that we can apply date-range filters server-side
and only pull the rows we need.

API base: https://api.neso.energy/api/3/action/

Rate limits (from NESO guidance):
  - CKAN metadata endpoints: max 1 req/s
  - Datastore endpoints:     max 2 req/min

Datasets used:
  - DC/DR/DM Results Summary (resource 888e5029-...)
  - DC Masterdata — per-unit bid detail (resource 0b8dbc3c-...)
  - DR Requirements (resource d6c576b9-...)
  - DM Requirements (resource 2aae8747-...)
"""

import re
import requests
import pandas as pd
import time
from datetime import date, timedelta
from typing import Optional, Dict, List
from loguru import logger

from ..utils import (
    load_config,
    setup_logging,
    save_dataframe,
)

BASE_URL = "https://api.neso.energy/api/3/action"

# Resource IDs — these are the UUIDs of the CSV resources on the portal.
# You can discover them yourself via:
#   GET {BASE_URL}/package_show?id=dynamic-containment-data
RESOURCE_IDS = {
    # Auction clearing prices & volumes per service per EFA block (Sep 2021 – Nov 2023)
    "results_summary": "888e5029-f786-41d2-bc15-cbfd1d285e96",
    # Per-unit bid/offer detail (DC only, 2020–2021)
    "dc_masterdata": "0b8dbc3c-e05e-44a4-b855-7dd1aa079c68",
    # Indicative volume requirements published ahead of auctions
    "dr_requirements": "d6c576b9-91d5-4c48-bf6d-300c7d7aa6ad",
    "dm_requirements": "2aae8747-776d-4fe5-af9c-adcf38f1af8a",
}

# EAC (Enduring Auction Capability) — successor to DC/DR/DM auctions.
# NESO publishes one resource per fiscal year (Apr–Mar) plus a live "Results
# Summary" resource, which is rotated into a new FY archive each April. Routing
# is therefore by segment; a single archive/current boundary goes stale on the
# next rotation and silently returns short.
#
# The rotation works by *addition*, not mutation: each April a brand-new
# "... FY<year> (Archive)" resource appears holding the year that just ended, and
# the live resource keeps its id while being truncated to the new year. So the
# annual edit is two lines — repoint the outgoing segment at the new archive id
# and close its end date, then open a new live segment on _EAC_LIVE_RESOURCE_ID.
#
# The three constants below are the naming premise the whole scheme rests on, and
# test_eac_naming_convention_is_unchanged asserts them against the live portal —
# so a rename by NESO fails with the reason named rather than silently returning
# short. Keep them in sync with any future dynamic discovery.
_EAC_PACKAGE_ID = "eac-auction-results"
# Exact name of the live resource. Must be an equality test, not a substring one:
# "NESO Response-Reserve Daily Results Summary" is a different, daily-granularity
# resource that a loose match would happily pick up.
_EAC_LIVE_RESOURCE_NAME = "NESO Response-Reserve Results Summary"
_EAC_ARCHIVE_NAME_RE = re.compile(
    r"^NESO Response-Reserve Results Summary FY(\d{4}) \(Archive\)$"
)

# (resource_id, first deliveryStart, last deliveryStart); None = to present.
# Adjacent segments overlap by a day — duplicates are dropped on load.
_EAC_SEGMENTS = [
    ("be5c6b0d-a335-4859-93f2-389585b4e9a1", "2023-11-02", "2024-03-31"),  # FY2023
    ("ab130833-3ce4-4361-90fb-69fa3cf30f15", "2024-03-31", "2025-03-31"),  # FY2024
    ("be55ee51-b79e-47da-b71e-a0f8865d9d66", "2025-03-31", "2026-03-31"),  # FY2025
    ("596f29ac-0387-4ba4-a6d3-95c243140707", "2026-03-31", None),          # FY2026 (live)
]

# Earliest date for which any EAC data exists
_EAC_ARCHIVE_START = "2023-11-02"

# Response products shared across both EAC resources (H/L split DC, DR, DM)
_EAC_RESPONSE_PRODUCTS = ("DCH", "DCL", "DRH", "DRL", "DMH", "DML")


class NESOCollector:
    """Collector for NESO Data Portal via the CKAN datastore API."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.api_config = config["apis"]["national_grid_eso"]
        self.rate_limit = self.api_config.get("rate_limit", 2)  # datastore default
        self.last_request_time = 0
        self.session = requests.Session()

        setup_logging(config)
        logger.info("NESO Collector initialized (CKAN datastore API)")

    # HTTP / rate-limit helpers

    def _rate_limit_wait(self):
        """Respect the 2 req/min datastore rate limit."""
        min_interval = 60 / self.rate_limit
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _datastore_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute a read-only SQL query against the NESO datastore.

        The CKAN ``datastore_search_sql`` action accepts a GET request
        with the query in the ``sql`` query-parameter.  The response
        JSON contains ``result.records`` (list of dicts) and
        ``result.fields`` (column metadata).
        """
        self._rate_limit_wait()
        resp = self.session.get(
            f"{BASE_URL}/datastore_search_sql",
            params={"sql": sql},
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()

        if not body.get("success"):
            error = body.get("error", {})
            raise RuntimeError(f"CKAN SQL query failed: {error}")

        records = body["result"]["records"]
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Drop the CKAN full-text-search column — it's internal noise
        df = df.drop(columns=["_full_text"], errors="ignore")
        return df

    def _datastore_search(
        self,
        resource_id: str,
        limit: int = 32000,
        offset: int = 0,
        sort: str = "_id asc",
    ) -> pd.DataFrame:
        """
        Simple paginated fetch (no date filter).  Useful for small
        reference tables like DR/DM requirements.
        """
        self._rate_limit_wait()
        resp = self.session.get(
            f"{BASE_URL}/datastore_search",
            params={
                "resource_id": resource_id,
                "limit": limit,
                "offset": offset,
                "sort": sort,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()

        if not body.get("success"):
            raise RuntimeError(f"CKAN search failed: {body.get('error')}")

        records = body["result"]["records"]
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.drop(columns=["_full_text"], errors="ignore")
        return df

    # DC / DR / DM auction results

    def collect_auction_results(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect DC, DR & DM auction clearing prices and volumes.

        Each row represents one service / EFA-block combination with
        columns: Service, EFA Date, Delivery Start, Delivery End, EFA,
        Cleared Volume, Clearing Price.

        The dataset covers 2021-09 to 2023-11.
        """
        logger.info(
            f"Collecting DC/DR/DM auction results from {start_date} to {end_date}"
        )
        resource = RESOURCE_IDS["results_summary"]
        sql = (
            f'SELECT * FROM "{resource}" '
            f"WHERE \"EFA Date\" >= '{start_date}' "
            f"AND \"EFA Date\" <= '{end_date}' "
            f'ORDER BY "EFA Date" ASC, "EFA" ASC'
        )

        try:
            df = self._datastore_sql(sql)

            if df.empty:
                logger.warning("No auction result records returned")
                return df

            df["EFA Date"] = pd.to_datetime(df["EFA Date"])
            df["Delivery Start"] = pd.to_datetime(df["Delivery Start"])
            df["Delivery End"] = pd.to_datetime(df["Delivery End"])
            df["Clearing Price"] = pd.to_numeric(df["Clearing Price"], errors="coerce")
            df["Cleared Volume"] = pd.to_numeric(
                df["Cleared Volume"], errors="coerce"
            )
            logger.info(f"Collected {len(df)} auction result records")

            if save:
                filename = f"auction_results_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type="raw", format="csv")

            return df

        except Exception as e:
            logger.error(f"Failed to collect auction results: {e}")
            return pd.DataFrame()

    # DR / DM requirements (small reference tables)

    def collect_dr_requirements(self, save: bool = True) -> pd.DataFrame:
        """Collect indicative Dynamic Regulation volume requirements."""
        logger.info("Collecting DR requirements")
        try:
            df = self._datastore_search(RESOURCE_IDS["dr_requirements"])
            if df.empty:
                logger.warning("No DR requirement records returned")
                return df
            df["EFA_DATE"] = pd.to_datetime(df["EFA_DATE"])
            logger.info(f"Collected {len(df)} DR requirement records")
            if save:
                save_dataframe(df, "dr_requirements", data_type="raw", format="csv")
            return df
        except Exception as e:
            logger.error(f"Failed to collect DR requirements: {e}")
            return pd.DataFrame()

    def collect_dm_requirements(self, save: bool = True) -> pd.DataFrame:
        """Collect indicative Dynamic Moderation volume requirements."""
        logger.info("Collecting DM requirements")
        try:
            df = self._datastore_search(RESOURCE_IDS["dm_requirements"])
            if df.empty:
                logger.warning("No DM requirement records returned")
                return df
            df["EFA_DATE"] = pd.to_datetime(df["EFA_DATE"])
            logger.info(f"Collected {len(df)} DM requirement records")
            if save:
                save_dataframe(df, "dm_requirements", data_type="raw", format="csv")
            return df
        except Exception as e:
            logger.error(f"Failed to collect DM requirements: {e}")
            return pd.DataFrame()

    # EAC (Enduring Auction Capability) — Nov 2023 onwards

    @staticmethod
    def _add_efa_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive ``efa`` (1–6) and ``efa_date`` from ``deliveryStart``.

        EFA blocks (local clock time — used consistently on NESO portal):
          EFA 1: 23:00 – 03:00  (starts previous calendar day)
          EFA 2: 03:00 – 07:00
          EFA 3: 07:00 – 11:00
          EFA 4: 11:00 – 15:00
          EFA 5: 15:00 – 19:00
          EFA 6: 19:00 – 23:00

        For EFA 1, the EFA date is the *end* date (i.e. deliveryStart.date + 1
        when hour == 23; deliveryStart.date when hour in 0–2).
        """
        ds = pd.to_datetime(df["deliveryStart"])
        hour = ds.dt.hour

        efa = pd.Series(0, index=df.index)
        efa_date = ds.dt.date

        efa[(hour >= 23) | (hour < 3)] = 1
        efa[(hour >= 3) & (hour < 7)]  = 2
        efa[(hour >= 7) & (hour < 11)] = 3
        efa[(hour >= 11) & (hour < 15)] = 4
        efa[(hour >= 15) & (hour < 19)] = 5
        efa[(hour >= 19) & (hour < 23)] = 6

        # For 23:00 slots the EFA date is the next calendar day
        next_day_mask = hour >= 23
        efa_date = pd.to_datetime(efa_date)
        efa_date[next_day_mask] = efa_date[next_day_mask] + pd.Timedelta(days=1)

        df = df.copy()
        df["efa"] = efa.astype(int)
        df["efa_date"] = efa_date.dt.date
        return df

    def _query_eac_resource(
        self, resource_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Pull Response products (DCH/DCL/DRH/DRL/DMH/DML) from one EAC resource
        for the given date range.  Returns raw 30-min rows.
        """
        products = ", ".join(f"'{p}'" for p in _EAC_RESPONSE_PRODUCTS)
        # end_date is inclusive: fetch up to midnight of end_date + 1
        end_dt = (
            pd.Timestamp(end_date) + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        sql = (
            f'SELECT "auctionProduct","deliveryStart","deliveryEnd",'
            f'"clearedVolume","clearingPrice" '
            f'FROM "{resource_id}" '
            f'WHERE "auctionProduct" IN ({products}) '
            f'AND "deliveryStart" >= \'{start_date}\' '
            f'AND "deliveryStart" < \'{end_dt}\' '
            f'ORDER BY "deliveryStart" ASC'
        )
        return self._datastore_sql(sql)

    def collect_eac_results(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect DC, DR & DM auction results from the EAC platform (Nov 2023 –
        present) and aggregate the 30-minute delivery windows to 4-hour EFA
        blocks so that the output schema matches the legacy ``auction_results``
        files exactly:

          Service, EFA Date, EFA, Delivery Start, Delivery End,
          Cleared Volume, Clearing Price

        The requested range is split across the per-fiscal-year EAC resources in
        ``_EAC_SEGMENTS`` and each overlapping segment is queried in turn, so a
        range spanning several fiscal years is stitched together transparently.
        """
        logger.info(f"Collecting EAC auction results from {start_date} to {end_date}")

        start = pd.Timestamp(start_date).date()
        end   = pd.Timestamp(end_date).date()

        frames: List[pd.DataFrame] = []

        for resource_id, seg_start, seg_end in _EAC_SEGMENTS:
            seg_lo = pd.Timestamp(seg_start).date()
            seg_hi = pd.Timestamp(seg_end).date() if seg_end else end

            # Skip segments that do not intersect the requested range
            if end < seg_lo or start > seg_hi:
                continue

            chunk_start = max(start, seg_lo).strftime("%Y-%m-%d")
            chunk_end   = min(end,   seg_hi).strftime("%Y-%m-%d")
            logger.info(f"  Querying EAC segment {resource_id[:8]} ({chunk_start} – {chunk_end})")

            df = self._query_eac_resource(resource_id, chunk_start, chunk_end)
            if df.empty:
                logger.warning(
                    f"  Segment {resource_id[:8]} returned no rows for "
                    f"{chunk_start} – {chunk_end} — the resource may have been "
                    "rotated; re-check _EAC_SEGMENTS against the NESO portal."
                )
            else:
                frames.append(df)

        if not frames:
            logger.warning(
                "No EAC records returned — date range may be outside Nov 2023 – present"
            )
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        # Adjacent FY segments overlap by a day — keep one row per slot
        raw = raw.drop_duplicates(subset=["auctionProduct", "deliveryStart"])
        raw["clearedVolume"] = pd.to_numeric(raw["clearedVolume"], errors="coerce")
        raw["clearingPrice"] = pd.to_numeric(raw["clearingPrice"], errors="coerce")
        raw = self._add_efa_columns(raw)

        # Aggregate 30-min slots → EFA blocks
        # Cleared volume: mean MW across the 8 half-hour windows
        # Clearing price: volume-weighted average price
        raw["price_x_vol"] = raw["clearingPrice"] * raw["clearedVolume"]

        agg = (
            raw.groupby(["auctionProduct", "efa_date", "efa"])
            .agg(
                delivery_start=("deliveryStart", "min"),
                delivery_end=("deliveryStart", "max"),  # last slot start ≈ block end
                cleared_volume=("clearedVolume", "mean"),
                total_pxv=("price_x_vol", "sum"),
                total_vol=("clearedVolume", "sum"),
            )
            .reset_index()
        )
        agg["clearing_price"] = agg["total_pxv"] / agg["total_vol"].replace(0, pd.NA)
        agg = agg.drop(columns=["total_pxv", "total_vol"])

        # Rename to match legacy auction_results schema
        df_out = agg.rename(
            columns={
                "auctionProduct": "Service",
                "efa_date":       "EFA Date",
                "efa":            "EFA",
                "delivery_start": "Delivery Start",
                "delivery_end":   "Delivery End",
                "cleared_volume": "Cleared Volume",
                "clearing_price": "Clearing Price",
            }
        )
        df_out["EFA Date"]       = pd.to_datetime(df_out["EFA Date"])
        df_out["Delivery Start"] = pd.to_datetime(df_out["Delivery Start"])
        df_out["Delivery End"]   = pd.to_datetime(df_out["Delivery End"])
        df_out = df_out.sort_values(["EFA Date", "EFA", "Service"]).reset_index(drop=True)

        logger.info(
            f"Collected {len(df_out)} EAC auction records "
            f"({len(raw)} raw 30-min slots aggregated)"
        )

        if save:
            filename = f"eac_results_{start_date}_{end_date}"
            save_dataframe(df_out, filename, data_type="raw", format="csv")

        return df_out

    # Embedded solar and wind (NESO Historic Demand Data)

    def collect_embedded_solar_wind(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect half-hourly embedded solar and embedded wind generation estimates
        from the NESO Historic Demand Data dataset.

        The dataset (id: ``demand_data_dataset_id`` in config) publishes one CSV
        per calendar year at:
          https://api.neso.energy/dataset/{dataset_id}/resource/{resource_id}/download/demanddata_{year}.csv

        Resource IDs are discovered dynamically via ``package_show`` so that the
        method remains future-proof when NESO adds new yearly files.

        Returns a long-format DataFrame with columns:
          settlementDate, settlementPeriod, fuelType, generation

        ``fuelType`` is either ``"SOLAR"`` or ``"EMBEDDED_WIND"``.
        """
        dataset_id = self.api_config.get("demand_data_dataset_id")
        if not dataset_id:
            raise ValueError("demand_data_dataset_id not set in config under apis.national_grid_eso")

        logger.info(
            f"Collecting embedded solar/wind from {start_date} to {end_date} "
            f"(dataset {dataset_id})"
        )

        # --- Discover year → resource_id mapping via package_show ---
        resp = self.session.get(
            f"{BASE_URL}/package_show",
            params={"id": dataset_id},
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        if not body.get("success"):
            raise RuntimeError(f"package_show failed: {body.get('error')}")

        resources = body["result"]["resources"]
        # Each resource name follows "Historic Demand Data {year}"
        year_to_rid: Dict[int, str] = {}
        for r in resources:
            name = r.get("name", "")
            if name.startswith("Historic Demand Data "):
                try:
                    year = int(name.replace("Historic Demand Data ", "").strip())
                    year_to_rid[year] = r["id"]
                except ValueError:
                    pass

        if not year_to_rid:
            logger.warning("No demanddata_*.csv resources found in package_show response")
            return pd.DataFrame()

        start_year = pd.Timestamp(start_date).year
        end_year   = pd.Timestamp(end_date).year
        years_needed = range(start_year, end_year + 1)

        frames: List[pd.DataFrame] = []
        for year in years_needed:
            rid = year_to_rid.get(year)
            if rid is None:
                logger.warning(f"No resource found for year {year} — skipping")
                continue

            url = (
                f"https://api.neso.energy/dataset/{dataset_id}"
                f"/resource/{rid}/download/demanddata_{year}.csv"
            )
            logger.info(f"  Downloading {url}")
            try:
                r = self.session.get(url, timeout=120)
                r.raise_for_status()
            except Exception as exc:
                logger.error(f"  Failed to download {year}: {exc}")
                continue

            from io import StringIO
            raw = pd.read_csv(
                StringIO(r.text),
                usecols=lambda c: c.strip().upper() in {
                    "SETTLEMENT_DATE", "SETTLEMENT_PERIOD",
                    "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_WIND_GENERATION",
                },
            )
            # Normalise column names (strip whitespace, uppercase)
            raw.columns = [c.strip().upper() for c in raw.columns]

            raw["SETTLEMENT_DATE"] = pd.to_datetime(raw["SETTLEMENT_DATE"], dayfirst=True, format="mixed")

            # Reshape to long format — two rows per settlement period
            solar = raw[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "EMBEDDED_SOLAR_GENERATION"]].copy()
            solar.columns = ["settlementDate", "settlementPeriod", "generation"]
            solar["fuelType"] = "SOLAR"

            wind = raw[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "EMBEDDED_WIND_GENERATION"]].copy()
            wind.columns = ["settlementDate", "settlementPeriod", "generation"]
            wind["fuelType"] = "EMBEDDED_WIND"

            frames.append(pd.concat([solar, wind], ignore_index=True))
            logger.info(f"  Year {year}: {len(raw)} settlement periods loaded")

        if not frames:
            logger.warning("No embedded solar/wind data collected")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

        # Filter to exact requested date range
        df = df[
            (df["settlementDate"] >= pd.Timestamp(start_date)) &
            (df["settlementDate"] <= pd.Timestamp(end_date))
        ].reset_index(drop=True)

        logger.info(
            f"Collected {len(df):,} embedded solar/wind rows "
            f"({df['settlementDate'].min().date()} – {df['settlementDate'].max().date()})"
        )

        if save:
            filename = f"embedded_solar_wind_{start_date}_{end_date}"
            save_dataframe(df, filename, data_type="raw", format="csv")

        return df

    # Convenience: collect everything

    def collect_all_markets(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Collect all available NESO datasets."""
        logger.info(f"Collecting all NESO data from {start_date} to {end_date}")

        eac_start = pd.Timestamp(_EAC_ARCHIVE_START).date()
        req_start = pd.Timestamp(start_date).date()
        req_end   = pd.Timestamp(end_date).date()

        # Legacy DC/DR/DM (up to Nov 2023)
        legacy_end = min(req_end, pd.Timestamp(_EAC_ARCHIVE_START).date() - timedelta(days=1))
        legacy_results = pd.DataFrame()
        if req_start <= legacy_end:
            legacy_results = self.collect_auction_results(
                start_date, legacy_end.strftime("%Y-%m-%d"), save
            )

        # EAC data (Nov 2023 onwards)
        eac_results = pd.DataFrame()
        if req_end >= eac_start:
            eac_query_start = max(req_start, eac_start).strftime("%Y-%m-%d")
            eac_results = self.collect_eac_results(eac_query_start, end_date, save)

        data = {
            "auction_results":      legacy_results,
            "eac_results":          eac_results,
            "dr_requirements":      self.collect_dr_requirements(save),
            "dm_requirements":      self.collect_dm_requirements(save),
            "embedded_solar_wind":  self.collect_embedded_solar_wind(start_date, end_date, save),
        }

        total = sum(len(df) for df in data.values())
        logger.info(f"Completed NESO collection — {total} total records")
        return data


# Quick smoke-test when run directly

if __name__ == "__main__":
    config = load_config()
    collector = NESOCollector(config)

    results = collector.collect_all_markets("2023-10-01", "2023-10-07", save=False)

    for name, df in results.items():
        print(f"\n{name.upper()}: {len(df)} records")
        if not df.empty:
            print(df.head(3).to_string())
