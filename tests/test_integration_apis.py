"""Live API contract tests — deselected by default.

    pytest -m integration

These guard against upstream changes that unit tests cannot see: NESO rotating
its EAC resources each April, and Elexon's date-parameter semantics. Both were
sources of silent data loss in this project.
"""
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

requests = pytest.importorskip("requests")

NESO_SQL = "https://api.neso.energy/api/3/action/datastore_search_sql"
ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"


def _neso_bounds(resource_id):
    sql = f'SELECT MIN("deliveryStart") AS lo, MAX("deliveryStart") AS hi FROM "{resource_id}"'
    r = requests.get(NESO_SQL, params={"sql": sql}, timeout=60)
    r.raise_for_status()
    rec = r.json()["result"]["records"][0]
    return pd.Timestamp(rec["lo"]).date(), pd.Timestamp(rec["hi"]).date()


@pytest.mark.parametrize("resource_id,declared_start,declared_end",
                         [(s[0], s[1], s[2]) for s in
                          __import__("src.data_collection.neso_collector",
                                     fromlist=["_EAC_SEGMENTS"])._EAC_SEGMENTS])
def test_eac_segment_still_covers_its_declared_range(resource_id, declared_start, declared_end):
    """Each configured resource must still serve the range we claim it does.

    When NESO rotates the live feed into a new fiscal-year archive, the live
    resource stops serving older data and this fails — which is the point.
    """
    lo, hi = _neso_bounds(resource_id)
    fix = (
        f"\nResource {resource_id} now serves {lo} – {hi}, but _EAC_SEGMENTS declares "
        f"{declared_start} – {declared_end or 'present'}."
        "\nMost likely cause: NESO's April fiscal-year rotation. See "
        "test_eac_naming_convention_is_unchanged for the current resource list."
    )
    assert lo <= pd.Timestamp(declared_start).date() + pd.Timedelta(days=1).to_pytimedelta(), (
        "EAC segment no longer reaches back to its declared start." + fix
    )
    if declared_end is not None:
        assert hi >= pd.Timestamp(declared_end).date() - pd.Timedelta(days=1).to_pytimedelta(), (
            "EAC segment no longer reaches its declared end." + fix
        )


def test_eac_naming_convention_is_unchanged():
    """
    The premise behind _EAC_SEGMENTS: the live resource has one exact name, and each
    archived fiscal year is published as "... FY<year> (Archive)".

    This is the test that turns the April rotation from a silent truncation into an
    actionable failure. It fails the moment a new FY archive appears upstream that no
    segment covers, and prints the new resource id — which is the whole fix.

    It fails differently, and says so, if NESO renames the resources instead. That
    matters because a rename would break any name-based discovery in exactly the way
    that caused the original ~2-year data loss: by finding nothing and carrying on.
    """
    from src.data_collection.neso_collector import (
        _EAC_ARCHIVE_NAME_RE,
        _EAC_LIVE_RESOURCE_NAME,
        _EAC_PACKAGE_ID,
        _EAC_SEGMENTS,
    )

    r = requests.get(
        "https://api.neso.energy/api/3/action/package_show",
        params={"id": _EAC_PACKAGE_ID}, timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    assert body.get("success"), (
        f"package_show failed for '{_EAC_PACKAGE_ID}' — the package may have been "
        f"renamed or withdrawn. Re-discover with package_search?q=enduring+auction+capability"
    )
    resources = {res["name"]: res["id"] for res in body["result"]["resources"]}

    # --- The live resource still exists under exactly the expected name ---
    assert _EAC_LIVE_RESOURCE_NAME in resources, (
        f"No resource named exactly '{_EAC_LIVE_RESOURCE_NAME}'.\n"
        "NESO has renamed the live EAC resource. Update _EAC_LIVE_RESOURCE_NAME and "
        "_EAC_ARCHIVE_NAME_RE in neso_collector.py to match.\n"
        f"Names currently published: {sorted(resources)}"
    )
    live_segments = [s for s in _EAC_SEGMENTS if s[2] is None]
    assert len(live_segments) == 1, "_EAC_SEGMENTS must have exactly one open-ended segment"
    assert live_segments[0][0] == resources[_EAC_LIVE_RESOURCE_NAME], (
        "The open-ended segment does not point at the live resource.\n"
        f"  _EAC_SEGMENTS live id: {live_segments[0][0]}\n"
        f"  portal live id:        {resources[_EAC_LIVE_RESOURCE_NAME]}"
    )

    # --- Every archived fiscal year upstream is routed by a segment ---
    archives = {
        int(m.group(1)): rid
        for name, rid in resources.items()
        if (m := _EAC_ARCHIVE_NAME_RE.match(name))
    }
    assert archives, (
        "No resources matched the FY archive naming pattern.\n"
        f"  pattern: {_EAC_ARCHIVE_NAME_RE.pattern}\n"
        "NESO has changed the archive naming convention. This is the failure mode that "
        "silently truncates collection — fix the pattern before the next refresh runs.\n"
        f"Names currently published: {sorted(resources)}"
    )

    known = {s[0] for s in _EAC_SEGMENTS}
    missing = {fy: rid for fy, rid in archives.items() if rid not in known}
    assert not missing, (
        "NESO has archived a fiscal year that _EAC_SEGMENTS does not route — this is the "
        "April rotation, and collection will silently return short until it is added.\n"
        + "".join(
            f'\n  ("{rid}", "{fy}-03-31", "{fy + 1}-03-31"),  # FY{fy}'
            for fy, rid in sorted(missing.items())
        )
        + "\n\nAdd the line(s) above to _EAC_SEGMENTS, close the previously-live segment's "
          "end date, and open a new live segment on the live resource id."
    )


def test_live_eac_resource_is_current():
    """The open-ended segment should be serving data from the last few days."""
    from src.data_collection.neso_collector import _EAC_SEGMENTS

    live = [s for s in _EAC_SEGMENTS if s[2] is None][0]
    _, hi = _neso_bounds(live[0])
    age = (pd.Timestamp.utcnow().date() - hi).days
    assert age <= 7, f"live EAC resource is {age} days stale — check for an April rotation"


def test_elexon_to_parameter_is_exclusive_at_midnight():
    """Documents the boundary semantics the collector depends on.

    A bare `to` date returns only the first few settlement periods of that day.
    If Elexon ever changes this to be inclusive, the end-of-day pin becomes
    redundant and this test tells us.
    """
    def count(to_value, day="2026-04-04"):
        r = requests.get(f"{ELEXON}/balancing/pricing/market-index",
                         params={"from": "2026-03-29", "to": to_value}, timeout=60)
        r.raise_for_status()
        return sum(1 for x in r.json().get("data", [])
                   if x.get("settlementDate") == day and x.get("dataProvider") == "APXMIDP")

    assert count("2026-04-04") < 10, "bare date no longer truncates — revisit the collector"
    assert count("2026-04-04T23:59:59Z") >= 46, "end-of-day pin no longer returns a full day"


def test_elexon_full_day_has_all_settlement_periods():
    r = requests.get(f"{ELEXON}/balancing/pricing/market-index",
                     params={"from": "2026-06-10", "to": "2026-06-10T23:59:59Z"}, timeout=60)
    r.raise_for_status()
    periods = {x["settlementPeriod"] for x in r.json()["data"]
               if x["dataProvider"] == "APXMIDP"}
    assert len(periods) == 48
