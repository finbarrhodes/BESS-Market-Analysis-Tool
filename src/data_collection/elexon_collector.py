"""
Elexon BMRS Data Collection Module

Collects data from the Elexon Insights Solution API (replacement for legacy BMRS).
All endpoints are public and require no API key.

API base: https://data.elexon.co.uk/bmrs/api/v1/

Endpoints used:
- /balancing/settlement/system-prices/{date}  (was B1770)
- /balancing/pricing/market-index              (was B1780)
- /datasets/FUELHH                             (was B1620)
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
    generate_date_range,
)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


class ElexonBMRSCollector:
    """Collector for Elexon BMRS data via the Insights Solution REST API."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.api_config = config["apis"]["elexon"]
        self.rate_limit = self.api_config.get("rate_limit", 60)
        self.last_request_time = 0
        self.session = requests.Session()

        setup_logging(config)
        logger.info("Elexon BMRS Collector initialized (Insights Solution API)")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        min_interval = 60 / self.rate_limit
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """
        GET request against the Insights Solution API.

        Returns the parsed JSON response dict, or raises on HTTP errors.
        """
        self._rate_limit_wait()
        url = f"{BASE_URL}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # System Sell / Buy Prices  (was B1770)
    # ------------------------------------------------------------------

    def collect_system_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect System Sell & Buy Prices for a date range.

        The endpoint serves one settlement date at a time, so we iterate
        over each day in the range.
        """
        logger.info(f"Collecting system prices from {start_date} to {end_date}")
        frames = []

        for day in generate_date_range(start_date, end_date, freq="D"):
            date_str = day.strftime("%Y-%m-%d")
            try:
                data = self._get(f"/balancing/settlement/system-prices/{date_str}")
                records = data.get("data", [])
                if records:
                    frames.append(pd.DataFrame(records))
            except requests.RequestException as e:
                logger.warning(f"Failed for {date_str}: {e}")

        # Filter out any empty frames before concat to avoid FutureWarning
        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning("No system price data retrieved")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        df["startTime"] = pd.to_datetime(df["startTime"])
        logger.info(f"Collected {len(df)} system price records")

        if save:
            filename = f"system_prices_{start_date}_{end_date}"
            save_dataframe(df, filename, data_type="raw", format="csv")

        return df

    # ------------------------------------------------------------------
    # Market Index Prices  (was B1780)
    # ------------------------------------------------------------------

    def collect_imbalance_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect Market Index Price & Volume (imbalance proxy).

        This endpoint accepts from/to query params and returns all records
        in a single response.
        """
        logger.info(f"Collecting market index prices from {start_date} to {end_date}")

        try:
            data = self._get(
                "/balancing/pricing/market-index",
                params={"from": start_date, "to": end_date},
            )
            records = data.get("data", [])
            if not records:
                logger.warning("No market index data returned")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["settlementDate"] = pd.to_datetime(df["settlementDate"])
            df["startTime"] = pd.to_datetime(df["startTime"])
            logger.info(f"Collected {len(df)} market index records")

            if save:
                filename = f"market_index_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type="raw", format="csv")

            return df

        except requests.RequestException as e:
            logger.error(f"Failed to collect market index prices: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Generation by Fuel Type  (was B1620)
    # ------------------------------------------------------------------

    def collect_generation_by_fuel(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect half-hourly generation output by fuel type (FUELHH dataset).
        """
        logger.info(f"Collecting generation by fuel from {start_date} to {end_date}")

        try:
            data = self._get(
                "/datasets/FUELHH",
                params={
                    "settlementDateFrom": start_date,
                    "settlementDateTo": end_date,
                },
            )
            records = data.get("data", [])
            if not records:
                logger.warning("No generation data returned")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["settlementDate"] = pd.to_datetime(df["settlementDate"])
            df["startTime"] = pd.to_datetime(df["startTime"])
            df["publishTime"] = pd.to_datetime(df["publishTime"])
            logger.info(f"Collected {len(df)} generation records")

            if save:
                filename = f"generation_by_fuel_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type="raw", format="csv")

            return df

        except requests.RequestException as e:
            logger.error(f"Failed to collect generation data: {e}")
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
        """Collect data from all available endpoints."""
        logger.info(f"Collecting all BMRS data from {start_date} to {end_date}")

        data = {
            "system_prices": self.collect_system_prices(start_date, end_date, save),
            "market_index": self.collect_imbalance_prices(start_date, end_date, save),
            "generation_by_fuel": self.collect_generation_by_fuel(
                start_date, end_date, save
            ),
        }

        total = sum(len(df) for df in data.values())
        logger.info(f"Completed BMRS collection — {total} total records")
        return data


# ----------------------------------------------------------------------
# Quick smoke-test when run directly
# ----------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()
    collector = ElexonBMRSCollector(config)

    start, end = "2024-01-01", "2024-01-02"
    results = collector.collect_all_markets(start, end, save=False)

    for name, df in results.items():
        print(f"\n{name.upper()}: {len(df)} records")
        if not df.empty:
            print(df.head(3).to_string())
