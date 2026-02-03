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

import requests
import pandas as pd
import time
from typing import Optional, Dict
from loguru import logger

from ..utils import (
    load_config,
    setup_logging,
    save_dataframe,
)

BASE_URL = "https://api.neso.energy/api/3/action"

# -----------------------------------------------------------------------
# Resource IDs — these are the UUIDs of the CSV resources on the portal.
# You can discover them yourself via:
#   GET {BASE_URL}/package_show?id=dynamic-containment-data
# -----------------------------------------------------------------------
RESOURCE_IDS = {
    # Auction clearing prices & volumes per service per EFA block
    "results_summary": "888e5029-f786-41d2-bc15-cbfd1d285e96",
    # Per-unit bid/offer detail (DC only, 2020–2021)
    "dc_masterdata": "0b8dbc3c-e05e-44a4-b855-7dd1aa079c68",
    # Indicative volume requirements published ahead of auctions
    "dr_requirements": "d6c576b9-91d5-4c48-bf6d-300c7d7aa6ad",
    "dm_requirements": "2aae8747-776d-4fe5-af9c-adcf38f1af8a",
}


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

    # ------------------------------------------------------------------
    # HTTP / rate-limit helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # DC / DR / DM auction results
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # DR / DM requirements (small reference tables)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Convenience: collect everything
    # ------------------------------------------------------------------

    def collect_all_markets(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Collect all available NESO datasets."""
        logger.info(f"Collecting all NESO data from {start_date} to {end_date}")

        data = {
            "auction_results": self.collect_auction_results(
                start_date, end_date, save
            ),
            "dr_requirements": self.collect_dr_requirements(save),
            "dm_requirements": self.collect_dm_requirements(save),
        }

        total = sum(len(df) for df in data.values())
        logger.info(f"Completed NESO collection — {total} total records")
        return data


# ----------------------------------------------------------------------
# Quick smoke-test when run directly
# ----------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()
    collector = NESOCollector(config)

    results = collector.collect_all_markets("2023-10-01", "2023-10-07", save=False)

    for name, df in results.items():
        print(f"\n{name.upper()}: {len(df)} records")
        if not df.empty:
            print(df.head(3).to_string())
