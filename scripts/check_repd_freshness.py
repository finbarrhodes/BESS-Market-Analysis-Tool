"""
scripts/check_repd_freshness.py
===============================
Warn when the REPD projected tail has outgrown a quarter.

REPD is a quarterly Excel drop with no API, so the monthly data refresh cannot
update it: `bess_fleet_capacity.parquet` keeps its measured months and its tail
stays projected, widening by one month with every refresh until a new extract is
downloaded into data/raw by hand.

Roughly a quarter of projection is the normal steady state, because REPD is
published a quarter behind. Materially more than that means a drop was missed and
the fleet series — which feeds the Market Impact page and the ML feature set — is
drifting further from measured reality each month.

Warns rather than fails by default: a stale planning database is not a reason to
block a market data refresh. Pass --strict to exit non-zero instead.

Run from anywhere:
    python scripts/check_repd_freshness.py
    python scripts/check_repd_freshness.py --strict --max-months 3
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DEFAULT_PARQUET = ROOT / "data" / "processed" / "bess_fleet_capacity.parquet"

# REPD trails by about a quarter, so three projected months is the expected
# steady state, not a problem.
DEFAULT_MAX_MONTHS = 3


def trailing_extrapolated_months(df: pd.DataFrame) -> int:
    """
    Length of the unbroken run of projected months at the end of the series.

    Counted from the tail rather than as a total, because only the trailing run
    reflects a missed REPD drop. A flagged month in the middle of the series would
    mean something else entirely and should not inflate this number.

    Returns 0 when the series has no is_extrapolated column, which is how
    prepare_data.py handles an older extract that predates the flag.
    """
    if "is_extrapolated" not in df.columns or df.empty:
        return 0
    flags = df.sort_values("month")["is_extrapolated"].astype(bool).tolist()
    count = 0
    for flag in reversed(flags):
        if not flag:
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warn when the REPD projected tail exceeds a quarter."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--max-months", type=int, default=DEFAULT_MAX_MONTHS,
        help=f"Projected months tolerated before warning (default: {DEFAULT_MAX_MONTHS})",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 instead of warning, for use as a hard gate.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"SKIP: {args.parquet} not found — nothing to check.")
        return

    df = pd.read_parquet(args.parquet)
    projected = trailing_extrapolated_months(df)
    measured_to = df.loc[~df["is_extrapolated"].astype(bool), "month"].max() \
        if "is_extrapolated" in df.columns else df["month"].max()

    print(
        f"REPD fleet series: measured through {pd.Timestamp(measured_to).date()}, "
        f"{projected} projected month(s) after it."
    )

    if projected <= args.max_months:
        print(f"OK: within the {args.max_months}-month tolerance.")
        return

    msg = (
        f"REPD projected tail is {projected} months, over the {args.max_months}-month "
        f"tolerance — a quarterly extract has probably been missed. Download the latest "
        f"from the DESNZ REPD page into data/raw/, re-run repd_collector.py, then "
        f"prepare_data.py."
    )
    # GitHub renders these in the job summary; harmless noise elsewhere.
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning title=REPD extract is stale::{msg}")
    print(f"WARNING: {msg}", file=sys.stderr)

    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
