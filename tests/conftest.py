from __future__ import annotations

import duckdb
import polars as pl
import pytest

from fourthdown.config import Paths, data_paths
from fourthdown.data import etl, schema, warehouse

RAW_ROWS: list[dict] = [
    # 1st and 10 at own 25, tied, first quarter: a designed pass on a neutral script.
    {
        "game_id": "2023_01_AAA_BBB",
        "play_id": 1.0,
        "posteam": "BBB",
        "down": 1.0,
        "ydstogo": 10.0,
        "yardline_100": 75.0,
        "score_differential": 0.0,
        "half_seconds_remaining": 1700.0,
        "qtr": 1.0,
        "wp": 0.5,
        "pass": 1.0,
        "rush": 0.0,
        "play_type": "pass",
    },
    # 3rd and 1 in the red zone, trailing by 21 late: a designed run in garbage time.
    {
        "game_id": "2023_01_AAA_BBB",
        "play_id": 2.0,
        "posteam": "BBB",
        "down": 3.0,
        "ydstogo": 1.0,
        "yardline_100": 12.0,
        "score_differential": -21.0,
        "half_seconds_remaining": 90.0,
        "qtr": 4.0,
        "wp": 0.01,
        "pass": 0.0,
        "rush": 1.0,
        "play_type": "run",
    },
    # Kneel-down: a rush by the box score, not a play call.
    {
        "game_id": "2023_01_AAA_BBB",
        "play_id": 3.0,
        "posteam": "BBB",
        "down": 1.0,
        "ydstogo": 10.0,
        "yardline_100": 60.0,
        "score_differential": 3.0,
        "half_seconds_remaining": 40.0,
        "qtr": 4.0,
        "wp": 0.99,
        "pass": 0.0,
        "rush": 1.0,
        "qb_kneel": 1.0,
        "play_type": "qb_kneel",
    },
    # Punt: no down-and-distance decision to model.
    {
        "game_id": "2023_01_AAA_BBB",
        "play_id": 4.0,
        "posteam": "AAA",
        "down": 4.0,
        "ydstogo": 12.0,
        "yardline_100": 88.0,
        "score_differential": -3.0,
        "half_seconds_remaining": 600.0,
        "qtr": 3.0,
        "wp": 0.4,
        "fixed_drive": 2.0,
        "pass": 0.0,
        "rush": 0.0,
        "special": 1.0,
        "punt_attempt": 1.0,
        "play_type": "punt",
    },
]

DEFAULTS: dict[str, object] = {
    "season": 2023,
    "season_type": "REG",
    "week": 1,
    "game_date": "2023-09-10",
    "home_team": "BBB",
    "away_team": "AAA",
    "defteam": "AAA",
    "posteam_type": "home",
    "fixed_drive": 1.0,
    "result": 7,
    "epa": 0.1,
    "wp": 0.5,
    "desc": "(15:00) test play",
}


def _raw_frame() -> pl.DataFrame:
    """A four-play frame shaped like an nflverse season file, with the rest nulled out."""
    rows = []
    for row in RAW_ROWS:
        merged: dict[str, object] = {}
        for column in schema.SOURCE_COLUMNS:
            merged[column] = row.get(column, DEFAULTS.get(column))
        rows.append(merged)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    numeric = ("play_id", "down", "ydstogo", "yardline_100", "score_differential")
    return frame.with_columns([pl.col(column).cast(pl.Float64) for column in numeric])


@pytest.fixture()
def raw_frame() -> pl.DataFrame:
    return _raw_frame()


@pytest.fixture()
def tmp_paths(tmp_path) -> Paths:
    paths = data_paths(tmp_path)
    paths.ensure()
    return paths


@pytest.fixture(scope="session")
def project_paths() -> Paths:
    return data_paths()


@pytest.fixture()
def views_connection(raw_frame, tmp_paths):
    """The semantic views over the synthetic season, in memory."""
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    with duckdb.connect(":memory:") as connection:
        warehouse.create_views(connection, tmp_paths)
        yield connection
