from __future__ import annotations

import shutil
import sys

import polars as pl
import pytest
from conftest import open_warehouse

from fourthdown.config import data_paths
from fourthdown.data import audit, etl, modeling, schema, warehouse
from fourthdown.sql.guard import UnsafeSQLError, validate

STAR_SCHEMA = (
    ("core", "fact_plays"),
    ("core", "dim_game"),
    ("core", "dim_team"),
    ("core", "dim_player"),
    ("core", "dim_season"),
    ("marts", "mart_team_season_tendencies"),
)


@pytest.fixture()
def connection(views_connection):
    return views_connection


def test_every_semantic_table_is_queryable(connection):
    for view in warehouse.VIEW_NAMES:
        connection.execute(f"SELECT * FROM {view} LIMIT 1").fetchall()


def test_the_star_schema_is_built_beside_the_semantic_layer(connection):
    for schema_name, table in STAR_SCHEMA:
        connection.execute(f"SELECT * FROM {schema_name}.{table} LIMIT 1").fetchall()


def test_plays_keeps_the_processed_column_contract_in_order(connection):
    # The schema card lists columns in ordinal order; the remodel must not reshuffle them.
    columns = connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = 'plays' ORDER BY ordinal_position"
    ).fetchall()
    assert tuple(name for (name,) in columns) == schema.PROCESSED_COLUMNS


def test_fact_plays_holds_no_game_level_attributes(connection):
    columns = {
        name
        for (name,) in connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'core' AND table_name = 'fact_plays'"
        ).fetchall()
    }
    assert {"game_id", "season", "posteam", "epa"} <= columns
    assert not {"home_team", "away_team", "week", "roof", "home_score"} & columns


def test_generated_sql_cannot_reach_the_star_schema():
    # The guard admits the semantic layer only; the model is never shown core or marts.
    with pytest.raises(UnsafeSQLError):
        validate("SELECT * FROM core.fact_plays")
    with pytest.raises(UnsafeSQLError):
        validate("SELECT * FROM mart_team_season_tendencies")


def test_games_view_has_one_row_per_game(connection):
    assert connection.execute("SELECT count(*) FROM games").fetchone() == (1,)


def test_team_game_pass_rate_counts_designed_plays_only(connection):
    row = connection.execute(
        "SELECT plays, designed_plays, pass_rate FROM team_game WHERE team = 'BBB'"
    ).fetchone()
    plays, designed_plays, pass_rate = row
    # Three plays for BBB, of which the kneel is not a call: one pass, one run.
    assert (plays, designed_plays) == (3, 2)
    assert pass_rate == pytest.approx(0.5)


def test_tendency_mart_matches_team_game(connection):
    mart = connection.execute(
        "SELECT designed_plays, pass_rate FROM marts.mart_team_season_tendencies "
        "WHERE team = 'BBB' AND season = 2023"
    ).fetchone()
    assert (
        mart
        == connection.execute(
            "SELECT designed_plays, pass_rate FROM team_game WHERE team = 'BBB'"
        ).fetchone()
    )


def test_player_game_follows_box_score_conventions(raw_frame, tmp_paths):
    # One passer: a 20-yard TD, a pick-six, an 8-yard sack, and a two-point try.
    plays = [
        {"yards_gained": 20.0, "touchdown": 1.0, "pass_touchdown": 1.0},
        {"yards_gained": 0.0, "touchdown": 1.0, "interception": 1.0},
        {"yards_gained": -8.0, "sack": 1.0},
        {"yards_gained": 2.0, "two_point_attempt": 1.0},
    ]
    template = raw_frame.head(1)
    rows = []
    for play_id, play in enumerate(plays, start=1):
        values = {"touchdown": 0.0, "pass_touchdown": 0.0, "interception": 0.0, "sack": 0.0}
        values |= {"two_point_attempt": 0.0, **play, "play_id": float(play_id)}
        columns = [pl.lit(value).alias(name) for name, value in values.items()]
        rows.append(template.with_columns(*columns, passer_player_id=pl.lit("QB1")))
    pl.concat(rows, how="vertical_relaxed").write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    with open_warehouse(tmp_paths) as connection:
        row = connection.execute(
            "SELECT plays, yards, touchdowns FROM player_game WHERE role = 'passer'"
        ).fetchone()
    # The sack is still a dropback (it counts toward plays and EPA), not passing yards.
    assert row == (3, 20, 1)


def test_player_game_carries_the_full_name_from_the_players_file(raw_frame, tmp_paths):
    frame = raw_frame.head(1).with_columns(
        passer_player_id=pl.lit("00-0033873"), passer_player_name=pl.lit("P.Mahomes")
    )
    frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    players = {"gsis_id": ["00-0033873"], "display_name": ["Patrick Mahomes"], "position": ["QB"]}
    pl.DataFrame(players).write_parquet(tmp_paths.players)
    with open_warehouse(tmp_paths) as connection:
        row = connection.execute("SELECT player_name, full_name FROM player_game").fetchone()
        dim = connection.execute(
            "SELECT full_name, short_name FROM core.dim_player WHERE player_id = '00-0033873'"
        ).fetchone()
    assert row == ("P.Mahomes", "Patrick Mahomes")
    assert dim == ("Patrick Mahomes", "P.Mahomes")


def test_drives_view_aggregates_by_drive(connection):
    row = connection.execute("SELECT plays, start_yardline_100 FROM drives WHERE posteam = 'BBB'")
    assert row.fetchone() == (3, 75)


def test_build_requires_processed_partitions(tmp_paths):
    with pytest.raises(FileNotFoundError, match="no processed partitions"):
        warehouse.build(tmp_paths)


def test_connect_requires_a_built_warehouse(tmp_paths):
    with pytest.raises(FileNotFoundError, match="warehouse missing"):
        warehouse.connect(tmp_paths)


def test_a_failing_data_test_leaves_the_live_warehouse_alone(raw_frame, tmp_paths):
    # The synthetic season has 2 teams, not 32: the league-wide tests fail, so the build
    # must raise and the previously built warehouse must still be the one on disk.
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    warehouse.build(tmp_paths, run_tests=False)
    before = tmp_paths.database.stat().st_mtime_ns
    with pytest.raises(modeling.DbtError, match="every_season_has_32_teams|32 teams"):
        warehouse.build(tmp_paths)
    assert tmp_paths.database.stat().st_mtime_ns == before
    assert modeling.load_test_results(tmp_paths) is None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot replace an open file")
def test_rebuild_succeeds_while_a_reader_holds_the_warehouse(raw_frame, tmp_paths):
    # The pipeline Job rebuilds while the warehouse service has the file open read-only.
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    warehouse.build(tmp_paths, run_tests=False)
    with warehouse.connect(tmp_paths) as reader:
        warehouse.build(tmp_paths, run_tests=False)
        assert reader.execute("SELECT count(*) FROM games").fetchone() == (1,)
    with warehouse.connect(tmp_paths) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone() == (1,)


def test_warehouse_survives_moving_the_data_root(raw_frame, tmp_path):
    # The container mounts the host's data/ at /data; every layer must follow, including
    # the staging views, which read Parquet by a path relative to the root.
    paths = data_paths(tmp_path / "host")
    paths.ensure()
    raw_frame.write_parquet(paths.season_raw(2023))
    etl.transform_season(2023, paths)
    warehouse.build(paths, run_tests=False)
    moved = data_paths(paths.root.rename(tmp_path / "container"))
    with warehouse.connect(moved) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone() == (1,)
        plays = connection.execute("SELECT count(*) FROM staging.stg_plays").fetchone()
        assert plays == (raw_frame.height,)


def test_audit_report_lists_the_data_tests(connection):
    tests = [
        modeling.TestResult("unique fact_plays.play_key", "pass", 0, ""),
        modeling.TestResult("every season has 32 teams", "fail", 1, "Got 1 result"),
    ]
    report = audit.render_report(connection, tests)
    assert "## Coverage by season" in report
    assert "1 of 2 dbt data tests passed" in report
    # Failures first.
    assert report.index("every season has 32 teams") < report.index("unique fact_plays")


def test_audit_report_says_when_no_tests_ran(connection):
    assert "ran without tests" in audit.render_report(connection, None)


def test_drives_scored_touchdown_ignores_opponent_return_touchdowns(raw_frame, tmp_paths):
    # Three one-play drives: a touchdown, an opponent's return touchdown, and a punt.
    template = raw_frame.head(1)
    rows = [
        template.with_columns(
            play_id=pl.lit(float(drive)),
            fixed_drive=pl.lit(float(drive)),
            fixed_drive_result=pl.lit(result),
            touchdown=pl.lit(touchdown),
        )
        for drive, result, touchdown in (
            (1, "Touchdown", 1.0),
            (2, "Opp touchdown", 1.0),
            (3, "Punt", 0.0),
        )
    ]
    pl.concat(rows, how="vertical_relaxed").write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    with open_warehouse(tmp_paths) as connection:
        rows = connection.execute(
            "SELECT drive_number, scored_touchdown FROM drives ORDER BY drive_number"
        ).fetchall()
    assert rows == [(1, True), (2, False), (3, False)]


def _week(raw_frame, week, game_id):
    return raw_frame.with_columns(
        week=pl.lit(week, dtype=raw_frame["week"].dtype), game_id=pl.lit(game_id)
    )


def _fact_rows(paths):
    with warehouse.connect(paths) as connection:
        return connection.execute(
            "SELECT * FROM core.fact_plays ORDER BY game_id, play_id"
        ).fetchall()


def test_plan_is_full_without_a_live_warehouse(raw_frame, tmp_paths):
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    assert warehouse.plan(tmp_paths) == warehouse.BuildPlan.full("no live warehouse")


def test_a_new_week_builds_incrementally_and_matches_a_full_build(raw_frame, tmp_paths):
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    warehouse.build(tmp_paths, run_tests=False)
    assert warehouse.plan(tmp_paths).weeks == ()

    pl.concat([raw_frame, _week(raw_frame, 2, "2023_02_BBB_AAA")]).write_parquet(
        tmp_paths.season_raw(2023)
    )
    etl.transform_week(2023, 2, tmp_paths)
    planned = warehouse.plan(tmp_paths)
    assert (planned.incremental, planned.weeks) == (True, ("2023-02",))
    warehouse.build(tmp_paths, run_tests=False, incremental=True)
    incremental = _fact_rows(tmp_paths)
    assert len(incremental) == 2 * raw_frame.height
    assert warehouse.plan(tmp_paths).weeks == ()

    warehouse.build(tmp_paths, run_tests=False)
    assert _fact_rows(tmp_paths) == incremental


def test_a_corrected_week_replaces_its_games_including_deleted_plays(raw_frame, tmp_paths):
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    warehouse.build(tmp_paths, run_tests=False)
    # nflverse republishes week 1 with the last play removed.
    raw_frame.head(raw_frame.height - 1).write_parquet(tmp_paths.season_raw(2023))
    etl.transform_week(2023, 1, tmp_paths)
    assert warehouse.plan(tmp_paths).weeks == ("2023-01",)
    warehouse.build(tmp_paths, run_tests=False, incremental=True)
    assert len(_fact_rows(tmp_paths)) == raw_frame.height - 1


def test_changed_dbt_code_forces_a_full_build(raw_frame, tmp_paths, tmp_path, monkeypatch):
    project = tmp_path / "dbt_project"
    shutil.copytree(modeling.project_dir(), project)
    monkeypatch.setenv(modeling.PROJECT_ENV, str(project))
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    warehouse.build(tmp_paths, run_tests=False)
    fact = project / "models" / "core" / "fact_plays.sql"
    fact.write_text(fact.read_text() + "\n-- edited\n")
    assert warehouse.plan(tmp_paths) == warehouse.BuildPlan.full("dbt model fact_plays changed")


def test_a_removed_partition_forces_a_full_build(raw_frame, tmp_paths):
    pl.concat([raw_frame, _week(raw_frame, 2, "2023_02_BBB_AAA")]).write_parquet(
        tmp_paths.season_raw(2023)
    )
    etl.transform_season(2023, tmp_paths)
    warehouse.build(tmp_paths, run_tests=False)
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    planned = warehouse.plan(tmp_paths)
    assert not planned.incremental and "2023-02" in planned.reason
