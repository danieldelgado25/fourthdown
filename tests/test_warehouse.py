from __future__ import annotations

import duckdb
import pytest

from fourthdown.data import audit, etl, warehouse


@pytest.fixture()
def connection(raw_frame, tmp_paths):
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    with duckdb.connect(":memory:") as connection:
        warehouse.create_views(connection, tmp_paths)
        yield connection


def test_every_view_is_queryable(connection):
    for view in warehouse.VIEW_NAMES:
        connection.execute(f"SELECT * FROM {view} LIMIT 1").fetchall()


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


def test_drives_view_aggregates_by_drive(connection):
    row = connection.execute("SELECT plays, start_yardline_100 FROM drives WHERE posteam = 'BBB'")
    assert row.fetchone() == (3, 75)


def test_build_requires_processed_partitions(tmp_paths):
    with pytest.raises(FileNotFoundError, match="no processed partitions"):
        warehouse.build(tmp_paths)


def test_connect_requires_a_built_warehouse(tmp_paths):
    with pytest.raises(FileNotFoundError, match="warehouse missing"):
        warehouse.connect(tmp_paths)


def test_audit_report_renders_from_a_tiny_warehouse(connection):
    report = audit.render_report(connection)
    assert "## Coverage by season" in report
    assert "unique play keys" in report
