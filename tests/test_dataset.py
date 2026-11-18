"""Checks against the real warehouse, skipped when the dataset has not been built.

These are the tests that would catch an upstream schema change or a transform that is
subtly wrong at scale but fine on four synthetic rows. CI runs them only in the job that
builds a season; locally they run as soon as `make build` has been done once.
"""

from __future__ import annotations

import pytest

from fourthdown.data import modeling, warehouse

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def connection(project_paths):
    if not project_paths.database.exists():
        pytest.skip("warehouse not built; run `make build`")
    with warehouse.connect(project_paths) as connection:
        yield connection


def test_the_last_build_ran_and_passed_every_data_test(project_paths, connection):
    tests = modeling.load_test_results(project_paths)
    assert tests, "the warehouse was built without its dbt data tests"
    failures = [test for test in tests if not test.passed]
    assert not failures, "; ".join(f"{test.name}: {test.message}" for test in failures)


def test_semantic_plays_reassembles_every_processed_play(project_paths, connection):
    # The star schema split must be lossless: plays (fact_plays joined back to dim_game)
    # holds exactly the processed rows.
    glob = (project_paths.processed / "season=*" / "*.parquet").as_posix()
    differences = connection.execute(
        f"SELECT count(*) FROM (SELECT * FROM read_parquet('{glob}') "
        "EXCEPT ALL SELECT * FROM plays)"
    ).fetchone()
    assert differences == (0,)


def test_season_play_counts_are_plausible(connection):
    rows = connection.execute(
        "SELECT season, count(*) FROM plays GROUP BY season ORDER BY season"
    ).fetchall()
    for season, plays in rows:
        assert 40_000 <= plays <= 55_000, f"{season} has {plays} plays"


def test_neutral_pass_rate_is_below_overall_pass_rate(connection):
    """Trailing teams throw. Removing game script should lower the league pass rate."""
    overall, neutral = connection.execute(
        "SELECT avg(is_pass_call::INT), "
        "avg(is_pass_call::INT) FILTER (WHERE is_neutral_script) "
        "FROM plays WHERE is_designed_play"
    ).fetchone()
    assert neutral < overall


def test_pass_rate_rises_over_the_period(connection):
    """The league passed more in the 2020s than in 2009-2012; the data should show it."""
    early, late = connection.execute(
        "SELECT avg(is_pass_call::INT) FILTER (WHERE season <= 2012), "
        "avg(is_pass_call::INT) FILTER (WHERE season >= 2021) "
        "FROM plays WHERE is_designed_play AND is_neutral_script"
    ).fetchone()
    if early is None or late is None:
        pytest.skip("needs both 2009-2012 and 2021+ built")
    assert late > early
