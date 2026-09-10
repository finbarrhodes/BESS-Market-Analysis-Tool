"""REPD trailing extrapolation.

REPD is published quarterly and always lags the half-hourly price data, so the
trailing months of the fleet series are projected. Evaluating the fitted OLS
line directly could place the first projected month *below* the last measured
one, which is meaningless for a cumulative capacity series — the projection is
anchored at the last measured value instead.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_collection.repd_collector import EXTRAP_FIT_MONTHS, REPDCollector


def _projects(n_months=30, start="2023-01-01", per_month=100.0):
    """Synthetic battery projects: a steady build-out, one per month."""
    months = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame({
        "operational_date": months,
        "installed_capacity_mw": [per_month] * n_months,
    })


def _series(n_months=30, end=None):
    c = REPDCollector(local_path="unused")
    return c.build_monthly_capacity_series(
        _projects(n_months), start_date="2023-01-01", end_date=end
    )


def test_fit_window_is_a_named_constant():
    assert isinstance(EXTRAP_FIT_MONTHS, int) and EXTRAP_FIT_MONTHS > 0


def test_series_is_monotonic_through_the_projection():
    """Installed capacity is cumulative — it cannot decrease, projected or not."""
    s = _series(n_months=30, end="2026-06-01")
    diffs = s["bess_fleet_mw"].diff().dropna()
    assert (diffs >= -1e-6).all(), (
        f"series decreases by up to {diffs.min():.1f} MW — the projection is not "
        "anchored at the last measured value"
    )


def test_projection_starts_at_or_above_last_measured_value():
    s = _series(n_months=30, end="2026-06-01")
    if "is_extrapolated" not in s.columns or not s["is_extrapolated"].any():
        pytest.skip("no months were projected for this range")
    last_measured = s.loc[~s["is_extrapolated"], "bess_fleet_mw"].iloc[-1]
    first_projected = s.loc[s["is_extrapolated"], "bess_fleet_mw"].iloc[0]
    assert first_projected >= last_measured - 1e-6


def test_extrapolation_flag_is_present_and_typed():
    s = _series(n_months=30, end="2026-06-01")
    assert "is_extrapolated" in s.columns, (
        "downstream needs to distinguish measured from projected capacity"
    )
    assert s["is_extrapolated"].dtype == bool


def test_measured_months_are_not_flagged():
    s = _series(n_months=30, end="2026-06-01")
    measured = s.loc[~s["is_extrapolated"]]
    assert len(measured) > 0
    assert measured["month"].max() < s.loc[s["is_extrapolated"], "month"].min()


def test_no_projection_when_range_ends_at_measured_data():
    s = _series(n_months=30, end="2025-06-01")
    assert not s["is_extrapolated"].any()


def test_capacity_never_negative():
    s = _series(n_months=30, end="2026-06-01")
    assert (s["bess_fleet_mw"] >= 0).all()


# ---------------------------------------------------------------------------
# Projected-tail freshness check (scripts/check_repd_freshness.py)
# ---------------------------------------------------------------------------

from scripts.check_repd_freshness import trailing_extrapolated_months


def _flagged(flags):
    """Fleet series with the given is_extrapolated flags, oldest first."""
    return pd.DataFrame({
        "month": pd.date_range("2026-01-01", periods=len(flags), freq="MS"),
        "bess_fleet_mw": [100.0 * (i + 1) for i in range(len(flags))],
        "is_extrapolated": flags,
    })


def test_trailing_run_is_counted_not_the_total():
    """Only the tail reflects a missed REPD drop; a mid-series flag does not."""
    assert trailing_extrapolated_months(_flagged([True, False, False, True, True])) == 2


def test_no_projected_tail_counts_zero():
    assert trailing_extrapolated_months(_flagged([False, False, False])) == 0


def test_a_fully_projected_series_counts_every_month():
    assert trailing_extrapolated_months(_flagged([True, True, True, True])) == 4


def test_order_is_by_month_not_row_order():
    """A shuffled frame must still count the chronological tail."""
    df = _flagged([False, False, True, True]).sample(frac=1, random_state=0)
    assert trailing_extrapolated_months(df) == 2


def test_missing_flag_column_is_treated_as_no_projection():
    """prepare_data.py omits the column for extracts predating the flag."""
    df = _flagged([True, True]).drop(columns=["is_extrapolated"])
    assert trailing_extrapolated_months(df) == 0


def test_empty_series_does_not_raise():
    assert trailing_extrapolated_months(_flagged([])) == 0
