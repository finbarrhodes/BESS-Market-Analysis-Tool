#!/usr/bin/env python3
"""
Pre-compute Backtest Cache
===========================
Runs all three MPC strategy backtests with a fixed 50 MW / 2h battery
configuration and writes results to data/cache/ as Parquet files.

The Streamlit app is a pure viewer that reads from this cache — no live
computation at app startup.

Usage:
    python scripts/precompute_cache.py

Re-run after any data update or methodology change, then commit the updated
cache files.  Runtime: ~30 minutes total, roughly 10 per strategy (measured
2026-09-09 on a test window running to 2026-09-08).

That figure grows over time and is worth re-checking. DEFAULT_TEST_START is held
fixed across refreshes so new data accrues to the test period, which means each
refresh lengthens the backtest and the runtime with it, close to linearly. The
refresh workflow's timeout-minutes is set with headroom for that, but not
infinite headroom.

Strategies computed:
  1. Perfect Foresight + MPC  — revenue ceiling
  2. Naive (D-1 prices) + MPC — zero-skill floor
  3. ML (Random Forest) + MPC — realistic best case; main result
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the repo root or the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.price_forecast import (
    DEFAULT_TEST_START,
    build_feature_matrix,
    get_feature_importances,
    load_bess_capacity,
    run_forecast_backtest,
    train_forecast_model,
)
from src.analysis.revenue_stack import ALL_SERVICES, REFERENCE_BATTERY, run_backtest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
CACHE     = Path(__file__).parent.parent / "data" / "cache"

# ---------------------------------------------------------------------------
# Fixed battery configuration (documented in methodology expander in the app)
# ---------------------------------------------------------------------------

BATTERY = REFERENCE_BATTERY
INITIAL_SOC    = 0.5   # Neutral midpoint; SoC tracked continuously thereafter
DISPATCH_METHOD = "mpc"
HORIZON         = 96    # 48h rolling LP horizon
SERVICES        = ALL_SERVICES
ML_MODEL_TYPE   = "rf"  # Random Forest selected at precompute time (see methodology expander)
N_IMPORTANCES   = 20    # Top-N feature importances stored in the manifest for display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _data_mtimes() -> dict:
    return {
        f.name: f.stat().st_mtime
        for f in sorted(PROCESSED.glob("*.parquet"))
    }


def _print_section(n: int, total: int, label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  [{n}/{total}] {label}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load source data
    # ------------------------------------------------------------------
    print("Loading processed data…")
    auctions  = pd.read_parquet(PROCESSED / "auctions.parquet")
    mkt_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    gen_daily = pd.read_parquet(PROCESSED / "generation_daily.parquet")

    # Full overlapping date range
    start_date = max(
        auctions["EFA Date"].min(), mkt_index["settlementDate"].min()
    ).date()
    end_date = min(
        auctions["EFA Date"].max(), mkt_index["settlementDate"].max()
    ).date()
    print(f"Date range: {start_date} → {end_date}")

    git_sha     = _git_sha()
    data_mtimes = _data_mtimes()
    manifest    = {}

    base_params = dict(
        power_mw             = BATTERY.power_mw,
        duration_h           = BATTERY.duration_h,
        efficiency_rt        = BATTERY.efficiency_rt,
        cycling_cost_per_mwh = BATTERY.cycling_cost_per_mwh,
        availability_factor  = BATTERY.availability_factor,
        initial_soc          = INITIAL_SOC,
        dispatch_method      = DISPATCH_METHOD,
        horizon              = HORIZON,
        start_date           = str(start_date),
        end_date             = str(end_date),
    )

    # ------------------------------------------------------------------
    # 1. Perfect Foresight + MPC
    # ------------------------------------------------------------------
    _print_section(1, 3, "Perfect Foresight + MPC")

    pf = run_backtest(
        auctions, mkt_index, BATTERY, SERVICES, start_date, end_date,
        initial_soc_frac=INITIAL_SOC,
        dispatch_method=DISPATCH_METHOD,
        horizon=HORIZON,
    )
    pf["monthly"].to_parquet(CACHE / "pf_mpc.parquet", index=False)
    if pf.get("soc_trajectory") is not None:
        pf["soc_trajectory"].to_parquet(CACHE / "soc_pf_mpc.parquet", index=False)

    manifest["pf_mpc"] = dict(
        computed_at = datetime.now(timezone.utc).isoformat(),
        git_sha     = git_sha,
        data_mtimes = data_mtimes,
        params      = base_params,
        summary     = pf["summary"],
    )
    print(f"  Total net revenue : £{pf['summary']['total_net']:>12,.0f}")
    print(f"  Ann. per MW       : £{pf['summary']['annualised_per_mw']:>10,.0f} / MW / yr")

    # ------------------------------------------------------------------
    # 2. Naive (D-1 prices) + MPC
    # ------------------------------------------------------------------
    _print_section(2, 3, "Naive (D-1 prices) + MPC")

    naive = run_forecast_backtest(
        strategy        = "naive",
        market_index    = mkt_index,
        auctions        = auctions,
        battery         = BATTERY,
        services        = SERVICES,
        start_date      = start_date,
        end_date        = end_date,
        initial_soc_frac = INITIAL_SOC,
        dispatch_method = DISPATCH_METHOD,
        horizon         = HORIZON,
    )
    naive["monthly"].to_parquet(CACHE / "naive_mpc.parquet", index=False)
    if naive.get("soc_trajectory") is not None:
        naive["soc_trajectory"].to_parquet(CACHE / "soc_naive_mpc.parquet", index=False)

    manifest["naive_mpc"] = dict(
        computed_at = datetime.now(timezone.utc).isoformat(),
        git_sha     = git_sha,
        data_mtimes = data_mtimes,
        params      = base_params,
        summary     = naive["summary"],
    )
    print(f"  Total net revenue : £{naive['summary']['total_net']:>12,.0f}")
    print(f"  Ann. per MW       : £{naive['summary']['annualised_per_mw']:>10,.0f} / MW / yr")

    # ------------------------------------------------------------------
    # 3. ML (Random Forest) + MPC
    # ------------------------------------------------------------------
    _print_section(3, 3, "ML (Random Forest) + MPC")

    print("  Building feature matrix…")
    feature_df = build_feature_matrix(mkt_index, gen_daily, load_bess_capacity())

    print(f"  Training {ML_MODEL_TYPE.upper()} model (test split: {DEFAULT_TEST_START})…")
    model, feature_cols, train_metrics, test_metrics = train_forecast_model(
        feature_df, model_type=ML_MODEL_TYPE, test_start=DEFAULT_TEST_START
    )
    print(f"  Test RMSE: {test_metrics['rmse']:.2f} £/MWh  |  "
          f"Spearman ρ: {test_metrics['spearman']:.3f}")

    print("  Running ML + MPC backtest…")
    ml = run_forecast_backtest(
        strategy        = "ml",
        market_index    = mkt_index,
        auctions        = auctions,
        battery         = BATTERY,
        services        = SERVICES,
        start_date      = start_date,
        end_date        = end_date,
        model           = model,
        feature_df      = feature_df,
        feature_cols    = feature_cols,
        initial_soc_frac = INITIAL_SOC,
        dispatch_method = DISPATCH_METHOD,
        horizon         = HORIZON,
    )
    ml["monthly"].to_parquet(CACHE / "ml_mpc.parquet", index=False)
    if ml.get("soc_trajectory") is not None:
        ml["soc_trajectory"].to_parquet(CACHE / "soc_ml_mpc.parquet", index=False)

    # Feature importances are stored here so the app never has to train a model
    # at runtime — the chart needs ~20 numbers, not a fresh Random Forest fit.
    importances = get_feature_importances(model, feature_cols).head(N_IMPORTANCES)

    manifest["ml_mpc"] = dict(
        computed_at   = datetime.now(timezone.utc).isoformat(),
        git_sha       = git_sha,
        data_mtimes   = data_mtimes,
        params        = {**base_params, "ml_model_type": ML_MODEL_TYPE,
                         "test_start": str(DEFAULT_TEST_START)},
        summary       = ml["summary"],
        model_metrics = {"train": train_metrics, "test": test_metrics},
        feature_importances = [
            {"feature": str(k), "importance": float(v)}
            for k, v in importances.items()
        ],
    )
    print(f"  Total net revenue : £{ml['summary']['total_net']:>12,.0f}")
    print(f"  Ann. per MW       : £{ml['summary']['annualised_per_mw']:>10,.0f} / MW / yr")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    manifest_path = CACHE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n✓  Cache written to data/cache/")
    print("   pf_mpc.parquet      naive_mpc.parquet      ml_mpc.parquet")
    print("   soc_pf_mpc.parquet  soc_naive_mpc.parquet  soc_ml_mpc.parquet")
    print("   manifest.json")
    print("\nCommit data/cache/ to the repository to make results available on")
    print("Streamlit Cloud without any compute step at deploy time.")


if __name__ == "__main__":
    main()
