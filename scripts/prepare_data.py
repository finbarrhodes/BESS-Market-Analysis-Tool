"""
scripts/prepare_data.py
=======================
Converts raw CSVs in data/raw/ into small pre-processed Parquet files in
data/processed/. The Parquets are committed to git; the raw CSVs are not
(.gitignore excludes them), so they exist only on the machine that collected
them.

Two modes:

  Full rebuild (default) — read every raw CSV and rebuild each Parquet from
  scratch. This is the local workflow, where the whole raw history is present.

      python scripts/prepare_data.py

  Append (--append) — treat the committed Parquet as the base and fold in
  whatever raw CSVs are present, writing the union back. This is what lets a
  CI runner extend the dataset without the ~450 MB of raw history: collect a
  few weeks into a scratch directory, merge, discard the scratch directory.

      python scripts/prepare_data.py --append --raw-dir /tmp/delta

Append is a merge, not a blind concat. Every dataset reuses the same
deduplication key as the full rebuild, and the newly collected rows win over
the committed ones, so overlapping pulls collapse, upstream revisions land,
and re-running the same slice twice is a no-op. Generation is the exception —
see prepare_generation.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FUEL_GROUP_MAP = {
    "CCGT":          "Gas",
    "OCGT":          "Gas",
    "NUCLEAR":       "Nuclear",
    "WIND":          "Wind",
    "NPSHYD":        "Hydro",
    "BIOMASS":       "Biomass",
    "COAL":          "Coal",
    "OIL":           "Oil",
    "PS":            "Pumped Storage",
    "INTFR":         "Interconnectors",
    "INTIRL":        "Interconnectors",
    "INTNED":        "Interconnectors",
    "INTNEM":        "Interconnectors",
    "INTNSL":        "Interconnectors",
    "INTVKL":        "Interconnectors",
    "INTIFA2":       "Interconnectors",
    "INTEW":         "Interconnectors",
    "INTELEC":       "Interconnectors",
    "OTHER":         "Other",
    # Embedded generation from NESO Historic Demand Data
    "SOLAR":         "Solar",
    "EMBEDDED_WIND": "Wind",
}


def _kb(path: Path) -> int:
    return path.stat().st_size // 1024


def _read_base(processed: Path, name: str, append: bool) -> pd.DataFrame | None:
    """Committed Parquet to merge into, or None for a full rebuild."""
    if not append:
        return None
    path = processed / name
    if not path.exists():
        print(f"  no existing {name} — building from raw only")
        return None
    return pd.read_parquet(path)


def _merge(new: pd.DataFrame | None, base: pd.DataFrame | None, subset: list[str]) -> pd.DataFrame | None:
    """
    Combine freshly read rows with the committed base on a natural key.

    The base goes first and `keep="last"` resolves collisions, so newly collected
    rows win: an upstream revision to an already-committed settlement period
    replaces the stale value rather than being discarded. Column order follows the
    committed file, which keeps append output comparable with a full rebuild even
    when the delta happens to contain only one of the two auction sources.
    """
    frames = [f for f in (base, new) if f is not None and not f.empty]
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=subset, keep="last")
    if base is not None:
        ordered = list(base.columns) + [c for c in merged.columns if c not in base.columns]
        merged = merged[ordered]
    return merged


def _write(df: pd.DataFrame, processed: Path, name: str, unit: str = "rows") -> None:
    out = processed / name
    df.to_parquet(out, index=False)
    print(f"  {len(df):,} {unit}  →  {_kb(out)} KB  ({out.name})")


_RANGE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


def _pull_order(path: Path) -> tuple[str, str]:
    """
    Sort key placing the oldest pull first.

    Raw filenames carry the range they cover — market_index_2019-01-01_2026-03-18.csv —
    and the end of that range is a good proxy for when the pull ran. Ordering by it
    rather than alphabetically matters because pulls overlap and the trailing days of
    any pull are provisional: NESO revises embedded solar and wind, and Elexon moves
    system prices through several settlement runs. Sorting by name put
    2019-01-01_2026-03-18 ahead of 2026-02-15_2026-08-17, so a first-wins dedup kept
    the older pull's unsettled tail. Oldest-first plus keep="last" means the most
    recent pull is authoritative, which is what the data actually warrants.
    """
    m = _RANGE_RE.search(path.name)
    return (m.group(2), path.name) if m else ("", path.name)


def _read_csvs(paths: list[Path], **kwargs) -> pd.DataFrame | None:
    """Read and concatenate a glob's worth of CSVs, oldest pull first; None if empty."""
    if not paths:
        return None
    return pd.concat(
        [pd.read_csv(p, **kwargs) for p in sorted(paths, key=_pull_order)],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Auctions — merge legacy auction_results + EAC into one file
# ---------------------------------------------------------------------------
def prepare_auctions(raw: Path, processed: Path, append: bool) -> None:
    print("Processing auctions...")
    new = _read_csvs(
        list(raw.glob("auction_results_*.csv")) + list(raw.glob("eac_results_*.csv")),
        parse_dates=["EFA Date", "Delivery Start", "Delivery End"],
    )
    base = _read_base(processed, "auctions.parquet", append)
    merged = _merge(new, base, subset=["Service", "EFA Date", "EFA"])
    if merged is None:
        print("  SKIP: no auction CSVs and no existing Parquet")
        return
    _write(merged.sort_values("EFA Date").reset_index(drop=True), processed, "auctions.parquet")


# ---------------------------------------------------------------------------
# Market index — half-hourly APXMIDP + N2EX spot prices
# ---------------------------------------------------------------------------
def prepare_market_index(raw: Path, processed: Path, append: bool) -> None:
    print("Processing market index...")
    new = _read_csvs(
        list(raw.glob("market_index_*.csv")),
        parse_dates=["settlementDate", "startTime"],
    )
    base = _read_base(processed, "market_index.parquet", append)
    merged = _merge(new, base, subset=["settlementDate", "settlementPeriod", "dataProvider"])
    if merged is None:
        print("  SKIP: no market index CSVs and no existing Parquet")
        return
    _write(merged, processed, "market_index.parquet")


# ---------------------------------------------------------------------------
# System prices — keep only the four columns the dashboard uses
# ---------------------------------------------------------------------------
def prepare_system_prices(raw: Path, processed: Path, append: bool) -> None:
    print("Processing system prices...")
    new = _read_csvs(
        list(raw.glob("system_prices_*.csv")),
        parse_dates=["settlementDate"],
        usecols=["settlementDate", "settlementPeriod", "systemSellPrice", "systemBuyPrice"],
    )
    base = _read_base(processed, "system_prices.parquet", append)
    merged = _merge(new, base, subset=["settlementDate", "settlementPeriod"])
    if merged is None:
        print("  SKIP: no system price CSVs and no existing Parquet")
        return
    _write(merged, processed, "system_prices.parquet")


# ---------------------------------------------------------------------------
# Generation — pre-aggregate to daily totals by fuel group
# The dashboard only plots daily stacked areas, so half-hourly + 15-fuel-type
# granularity (69 MB CSV) can be collapsed to ~14 k rows here.
# ---------------------------------------------------------------------------
def prepare_generation(raw: Path, processed: Path, append: bool) -> None:
    print("Processing generation (aggregating to daily by fuel group)...")

    # The raw pulls overlap: generation_by_fuel_2019-01-01_2026-03-18.csv re-covers
    # almost everything the earlier files hold, so a plain concat counts most of
    # history twice (and Feb 2026 three times). Deduplicate per source before
    # summing, exactly as the auction and market-index sections above do.
    #
    # Keys differ by source because the columns do. FUELHH carries startTime, the
    # true UTC half-hour, which is unique per fuel and survives the BST/GMT clock
    # changes that make settlementPeriod ambiguous twice a year. The embedded
    # series has no startTime, so it falls back to the settlement key.
    gen_by_fuel = _read_csvs(
        list(raw.glob("generation_by_fuel_*.csv")),
        parse_dates=["settlementDate", "startTime"],
        usecols=["settlementDate", "startTime", "fuelType", "generation"],
    )
    if gen_by_fuel is not None:
        gen_by_fuel = gen_by_fuel.drop_duplicates(subset=["startTime", "fuelType"], keep="last")

    embedded = _read_csvs(
        list(raw.glob("embedded_solar_wind_*.csv")),
        parse_dates=["settlementDate"],
        usecols=["settlementDate", "settlementPeriod", "fuelType", "generation"],
    )
    if embedded is not None:
        embedded = embedded.drop_duplicates(
            subset=["settlementDate", "settlementPeriod", "fuelType"], keep="last"
        )

    base = _read_base(processed, "generation_daily.parquet", append)

    frames = [
        f[["settlementDate", "fuelType", "generation"]]
        for f in (gen_by_fuel, embedded)
        if f is not None
    ]
    if not frames:
        if base is None:
            print("  SKIP: no generation CSVs and no existing Parquet")
        else:
            print("  no new generation CSVs — existing Parquet left as is")
        return

    gen = pd.concat(frames, ignore_index=True)
    gen["fuelGroup"] = gen["fuelType"].map(FUEL_GROUP_MAP).fillna("Other")
    gen_daily = (
        gen.groupby(["settlementDate", "fuelGroup"])["generation"]
        .sum()
        .reset_index()
    )

    # Generation is the one dataset a key-merge cannot handle: the daily total is
    # a sum over half-hourly rows, and once collapsed there is no way to tell a
    # re-pulled day from a double count. So append replaces whole days instead —
    # drop every date the new slice touches, then concatenate. The collector pulls
    # complete days, so this is a clean swap and stays idempotent on re-runs.
    #
    # A day still in progress upstream lands as a partial total. That is also true
    # of a full rebuild, and the downstream guard already exists: compute_kpis.py
    # only counts days carrying >= 46 settlement periods.
    if base is not None:
        touched = set(gen_daily["settlementDate"].unique())
        kept = base[~base["settlementDate"].isin(touched)]
        print(f"  replacing {len(touched):,} day(s), keeping {len(kept):,} existing rows")
        gen_daily = pd.concat([kept, gen_daily], ignore_index=True)

    gen_daily = gen_daily.sort_values(["settlementDate", "fuelGroup"]).reset_index(drop=True)
    _write(gen_daily, processed, "generation_daily.parquet")


# ---------------------------------------------------------------------------
# BESS fleet capacity — monthly cumulative from REPD
# ---------------------------------------------------------------------------
def prepare_bess_fleet(raw: Path, processed: Path, append: bool) -> None:
    print("Processing BESS fleet capacity (REPD)...")
    repd_raw = raw / "bess_fleet_capacity_raw.csv"

    # REPD is a quarterly Excel drop rather than an API, so there is no delta to
    # collect. The whole series is rebuilt from the one CSV when it is present,
    # and left untouched when it is not — which is the normal case on a runner.
    if not repd_raw.exists():
        if append and (processed / "bess_fleet_capacity.parquet").exists():
            print("  no REPD extract in this slice — existing Parquet left as is")
            return
        print(
            f"  SKIP: {repd_raw.name} not found.\n"
            "  Run REPDCollector.collect() to generate it:\n"
            "    python src/data_collection/repd_collector.py <repd_url_or_local_path>"
        )
        return

    bess = pd.read_csv(repd_raw)
    bess["month"] = pd.to_datetime(bess["month"])
    # is_extrapolated marks months projected past the end of the REPD extract
    # (REPD is quarterly, so the tail is always projected). Carried through so
    # the app and methodology can distinguish measured from projected capacity.
    cols = ["month", "bess_fleet_mw"]
    if "is_extrapolated" in bess.columns:
        bess["is_extrapolated"] = bess["is_extrapolated"].astype(bool)
        cols.append("is_extrapolated")
    bess = bess[cols].sort_values("month").reset_index(drop=True)
    _write(bess, processed, "bess_fleet_capacity.parquet", unit="months")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw CSVs into the committed processed Parquets."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge raw CSVs into the existing Parquets instead of rebuilding "
             "from scratch. Use when only a recent slice of raw data is present.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW,
        help=f"Directory of raw CSVs (default: {RAW.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED,
        help=f"Output directory for Parquets (default: {PROCESSED.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'Appending to' if args.append else 'Rebuilding'} {args.processed_dir} "
          f"from {args.raw_dir}\n")

    for step in (
        prepare_auctions,
        prepare_market_index,
        prepare_system_prices,
        prepare_generation,
        prepare_bess_fleet,
    ):
        step(args.raw_dir, args.processed_dir, args.append)

    print("\nDone. Commit the files in data/processed/ to git.")


if __name__ == "__main__":
    main()
