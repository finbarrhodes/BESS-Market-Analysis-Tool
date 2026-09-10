"""
Merge semantics for scripts/prepare_data.py.

The append path is what lets a CI runner extend the dataset without the ~450 MB
of raw history it can never have: it treats the committed Parquet as the base and
folds a freshly collected slice into it. These tests pin the two properties that
makes safe — a full rebuild and an incremental append must agree, and re-running
the same slice must change nothing.

All synthetic, no network, no committed data files.
"""

import pandas as pd
import pytest

from scripts.prepare_data import (
    _pull_order,
    prepare_generation,
    prepare_system_prices,
)


def _write_system_prices(raw, start, end, rows):
    """rows: list of (settlementPeriod, sellPrice)."""
    pd.DataFrame(
        {
            "settlementDate": [start] * len(rows),
            "settlementPeriod": [sp for sp, _ in rows],
            "systemSellPrice": [p for _, p in rows],
            "systemBuyPrice": [p for _, p in rows],
        }
    ).to_csv(raw / f"system_prices_{start}_{end}.csv", index=False)


def _read(processed, name):
    return pd.read_parquet(processed / name)


# ---------------------------------------------------------------------------
# Pull ordering
# ---------------------------------------------------------------------------

def test_pull_order_sorts_by_range_end_not_filename(tmp_path):
    """
    The bug this guards: sorting by name puts a 2019-01-01_2026-03-18 pull ahead
    of a 2026-02-15_2026-08-17 one, so a first-wins dedup keeps the older pull's
    provisional tail. Ordering by range end puts the most recent pull last, where
    keep="last" makes it authoritative.
    """
    # Both of these are real filenames from data/raw. The backfill was collected
    # in March 2026 and covers the narrow pull entirely, so it is the authoritative
    # one — but it sorts first by name because it *starts* in 2019.
    narrow = tmp_path / "market_index_2023-07-01_2023-10-31.csv"   # ran ~Oct 2023
    backfill = tmp_path / "market_index_2019-01-01_2026-03-18.csv"  # ran ~Mar 2026

    assert sorted([narrow, backfill]) == [backfill, narrow], "name order puts the older pull last"
    assert sorted([narrow, backfill], key=_pull_order) == [narrow, backfill], (
        "pull order must put the most recent collection last, where keep='last' wins"
    )


def test_pull_order_tolerates_unparseable_names(tmp_path):
    """A file with no date range sorts first rather than raising."""
    plain = tmp_path / "dm_requirements.csv"
    dated = tmp_path / "market_index_2019-01-01_2020-01-01.csv"
    assert sorted([dated, plain], key=_pull_order) == [plain, dated]


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------

def test_full_rebuild_prefers_the_most_recent_pull(tmp_path):
    """
    Overlapping pulls disagree because the trailing days of any pull are
    provisional — Elexon moves system prices through several settlement runs.
    The later pull holds the settled value and must win.
    """
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir(), processed.mkdir()
    _write_system_prices(raw, "2026-01-01", "2026-03-18", [(1, 90.76)])   # provisional
    _write_system_prices(raw, "2026-01-01", "2026-08-17", [(1, 151.90)])  # settled

    prepare_system_prices(raw, processed, append=False)

    out = _read(processed, "system_prices.parquet")
    assert len(out) == 1
    assert out["systemSellPrice"].iloc[0] == pytest.approx(151.90)


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------

def test_append_matches_a_full_rebuild(tmp_path):
    """The property the whole pipeline rests on: incremental == from scratch."""
    both, split = tmp_path / "both", tmp_path / "split"
    for d in (both / "raw", both / "processed", split / "old", split / "delta", split / "processed"):
        d.mkdir(parents=True)

    # Same two pulls, once side by side and once as base-then-delta.
    for raw in (both / "raw", split / "old"):
        _write_system_prices(raw, "2026-01-01", "2026-03-18", [(1, 90.76), (2, 40.0)])
    for raw in (both / "raw", split / "delta"):
        _write_system_prices(raw, "2026-01-01", "2026-08-17", [(1, 151.90), (3, 55.0)])

    prepare_system_prices(both / "raw", both / "processed", append=False)
    prepare_system_prices(split / "old", split / "processed", append=False)
    prepare_system_prices(split / "delta", split / "processed", append=True)

    a = _read(both / "processed", "system_prices.parquet").sort_values("settlementPeriod")
    b = _read(split / "processed", "system_prices.parquet").sort_values("settlementPeriod")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    # The revision landed, and the period only the base knew about survived.
    assert set(b["settlementPeriod"]) == {1, 2, 3}
    assert b.loc[b.settlementPeriod == 1, "systemSellPrice"].iloc[0] == pytest.approx(151.90)


def test_append_is_idempotent(tmp_path):
    """A re-run — a retried CI job, an overlapping slice — must be a no-op."""
    raw, delta, processed = tmp_path / "raw", tmp_path / "delta", tmp_path / "processed"
    for d in (raw, delta, processed):
        d.mkdir()
    _write_system_prices(raw, "2026-01-01", "2026-03-18", [(1, 90.76)])
    _write_system_prices(delta, "2026-01-01", "2026-08-17", [(1, 151.90), (2, 55.0)])

    prepare_system_prices(raw, processed, append=False)
    prepare_system_prices(delta, processed, append=True)
    first = _read(processed, "system_prices.parquet")
    prepare_system_prices(delta, processed, append=True)
    second = _read(processed, "system_prices.parquet")

    pd.testing.assert_frame_equal(first, second)


def test_append_with_no_new_csvs_leaves_the_parquet_alone(tmp_path):
    """An empty slice must not blank the committed file."""
    raw, empty, processed = tmp_path / "raw", tmp_path / "empty", tmp_path / "processed"
    for d in (raw, empty, processed):
        d.mkdir()
    _write_system_prices(raw, "2026-01-01", "2026-03-18", [(1, 90.76)])

    prepare_system_prices(raw, processed, append=False)
    before = _read(processed, "system_prices.parquet")
    prepare_system_prices(empty, processed, append=True)

    pd.testing.assert_frame_equal(before, _read(processed, "system_prices.parquet"))


# ---------------------------------------------------------------------------
# Generation — the one dataset a key-merge cannot handle
# ---------------------------------------------------------------------------

def _write_generation(raw, start, end, rows):
    """rows: list of (settlementDate, 'HH:MM', fuelType, generation)."""
    pd.DataFrame(
        {
            "settlementDate": [d for d, _, _, _ in rows],
            "startTime": [f"{d}T{t}:00Z" for d, t, _, _ in rows],
            "fuelType": [f for _, _, f, _ in rows],
            "generation": [g for _, _, _, g in rows],
        }
    ).to_csv(raw / f"generation_by_fuel_{start}_{end}.csv", index=False)


def test_generation_append_replaces_whole_days(tmp_path):
    """
    Daily totals are sums over half-hourly rows, so once collapsed there is no key
    to dedupe on — a re-pulled day would double-count. Append drops every date the
    new slice touches and re-adds it, which keeps re-runs idempotent and lets a
    revised day overwrite cleanly.
    """
    raw, delta, processed = tmp_path / "raw", tmp_path / "delta", tmp_path / "processed"
    for d in (raw, delta, processed):
        d.mkdir()

    _write_generation(raw, "2026-01-01", "2026-01-02", [
        ("2026-01-01", "00:00", "WIND", 100),
        ("2026-01-01", "00:30", "WIND", 100),
        ("2026-01-02", "00:00", "WIND", 500),
    ])
    prepare_generation(raw, processed, append=False)
    base = _read(processed, "generation_daily.parquet")
    assert base.loc[base.settlementDate == "2026-01-01", "generation"].iloc[0] == 200

    # Day 1 revised downward, day 3 is new. Day 2 is untouched and must survive.
    _write_generation(delta, "2026-01-01", "2026-01-03", [
        ("2026-01-01", "00:00", "WIND", 60),
        ("2026-01-01", "00:30", "WIND", 60),
        ("2026-01-03", "00:00", "WIND", 900),
    ])
    prepare_generation(delta, processed, append=True)
    out = _read(processed, "generation_daily.parquet")

    by_date = out.set_index(out.settlementDate.astype(str))["generation"]
    assert by_date["2026-01-01"] == 120, "revised day must replace, not add"
    assert by_date["2026-01-02"] == 500, "untouched day must survive"
    assert by_date["2026-01-03"] == 900, "new day must land"

    prepare_generation(delta, processed, append=True)
    pd.testing.assert_frame_equal(out, _read(processed, "generation_daily.parquet"))
