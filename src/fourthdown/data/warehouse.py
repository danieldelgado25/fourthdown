"""DuckDB semantic layer over the processed Parquet partitions.

These five views are the only surface the query layer (and later the text-to-SQL prompt)
sees. Keeping the vocabulary small and pre-joined is what makes generated SQL correct
often enough to be useful: a model asked to reason about ``team_game`` rarely invents a
join, whereas a model handed 372 raw columns invents one constantly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from fourthdown.config import Paths, data_paths

LOGGER = logging.getLogger(__name__)

VIEW_NAMES: tuple[str, ...] = ("plays", "games", "drives", "team_game", "player_game")

PLAYS_VIEW = """
CREATE OR REPLACE VIEW plays AS
SELECT * FROM read_parquet('{glob}')
"""

GAMES_VIEW = """
CREATE OR REPLACE VIEW games AS
SELECT
    game_id,
    any_value(season) AS season,
    any_value(season_type) AS season_type,
    any_value(week) AS week,
    any_value(game_date) AS game_date,
    any_value(home_team) AS home_team,
    any_value(away_team) AS away_team,
    any_value(home_score) AS home_score,
    any_value(away_score) AS away_score,
    any_value(result) AS home_margin,
    any_value(total) AS total_points,
    any_value(spread_line) AS spread_line,
    any_value(total_line) AS total_line,
    any_value(div_game) AS div_game,
    any_value(roof) AS roof,
    any_value(surface) AS surface,
    any_value(temp) AS temp,
    any_value(wind) AS wind,
    any_value(stadium) AS stadium,
    any_value(home_coach) AS home_coach,
    any_value(away_coach) AS away_coach,
    count(*) AS plays
FROM plays
GROUP BY game_id
"""

DRIVES_VIEW = """
CREATE OR REPLACE VIEW drives AS
SELECT
    drive_id,
    any_value(game_id) AS game_id,
    any_value(season) AS season,
    any_value(week) AS week,
    any_value(posteam) AS posteam,
    any_value(defteam) AS defteam,
    any_value(fixed_drive) AS drive_number,
    any_value(fixed_drive_result) AS drive_result,
    min(qtr) AS start_quarter,
    arg_min(yardline_100, play_id) AS start_yardline_100,
    arg_max(yardline_100, play_id) AS end_yardline_100,
    arg_min(game_seconds_remaining, play_id) AS end_game_seconds_remaining,
    count(*) AS plays,
    count(*) FILTER (WHERE is_designed_play) AS designed_plays,
    sum(yards_gained) AS yards_gained,
    sum(first_down::INT) AS first_downs,
    sum(epa) AS epa,
    avg(epa) AS epa_per_play,
    max(touchdown::INT) = 1 AS scored_touchdown
FROM plays
WHERE drive_id IS NOT NULL AND posteam IS NOT NULL
GROUP BY drive_id
"""

TEAM_GAME_VIEW = """
CREATE OR REPLACE VIEW team_game AS
SELECT
    game_id,
    posteam AS team,
    any_value(defteam) AS opponent,
    any_value(season) AS season,
    any_value(week) AS week,
    any_value(posteam_is_home) AS is_home,
    count(*) AS plays,
    count(*) FILTER (WHERE is_designed_play) AS designed_plays,
    avg(is_pass_call::INT) FILTER (WHERE is_designed_play) AS pass_rate,
    avg(is_pass_call::INT) FILTER (WHERE is_designed_play AND is_neutral_script)
        AS neutral_pass_rate,
    avg(epa) FILTER (WHERE is_designed_play) AS epa_per_play,
    avg(epa) FILTER (WHERE is_designed_play AND pass_play) AS pass_epa_per_play,
    avg(epa) FILTER (WHERE is_designed_play AND rush_play) AS rush_epa_per_play,
    avg(success::INT) FILTER (WHERE is_designed_play) AS success_rate,
    sum(yards_gained) FILTER (WHERE is_designed_play) AS yards_gained,
    sum(touchdown::INT) AS touchdowns,
    sum(interception::INT) AS interceptions,
    sum(fumble_lost::INT) AS fumbles_lost,
    sum(third_down_converted::INT) AS third_down_conversions,
    count(*) FILTER (WHERE down = 3 AND is_designed_play) AS third_down_attempts,
    sum(fourth_down_converted::INT) AS fourth_down_conversions,
    count(*) FILTER (WHERE down = 4 AND is_designed_play) AS fourth_down_attempts
FROM plays
WHERE posteam IS NOT NULL
GROUP BY game_id, posteam
"""

PLAYER_GAME_VIEW = """
CREATE OR REPLACE VIEW player_game AS
SELECT
    game_id, season, week, posteam AS team, 'passer' AS role,
    passer_player_id AS player_id, passer_player_name AS player_name,
    count(*) AS plays,
    sum(yards_gained) AS yards,
    sum(touchdown::INT) AS touchdowns,
    avg(epa) AS epa_per_play,
    avg(cpoe) AS cpoe
FROM plays
WHERE passer_player_id IS NOT NULL
GROUP BY ALL
UNION ALL
SELECT
    game_id, season, week, posteam AS team, 'rusher' AS role,
    rusher_player_id AS player_id, rusher_player_name AS player_name,
    count(*) AS plays,
    sum(yards_gained) AS yards,
    sum(touchdown::INT) AS touchdowns,
    avg(epa) AS epa_per_play,
    NULL AS cpoe
FROM plays
WHERE rusher_player_id IS NOT NULL
GROUP BY ALL
UNION ALL
SELECT
    game_id, season, week, posteam AS team, 'receiver' AS role,
    receiver_player_id AS player_id, receiver_player_name AS player_name,
    count(*) AS plays,
    sum(yards_gained) AS yards,
    sum(touchdown::INT) AS touchdowns,
    avg(epa) AS epa_per_play,
    NULL AS cpoe
FROM plays
WHERE receiver_player_id IS NOT NULL
GROUP BY ALL
"""

VIEW_STATEMENTS: tuple[str, ...] = (
    GAMES_VIEW,
    DRIVES_VIEW,
    TEAM_GAME_VIEW,
    PLAYER_GAME_VIEW,
)


def processed_glob(paths: Paths) -> str:
    return str(paths.processed / "season=*" / "*.parquet")


def create_views(connection: duckdb.DuckDBPyConnection, paths: Paths) -> None:
    """Define every view against the processed partitions of ``paths``."""
    connection.execute(PLAYS_VIEW.format(glob=processed_glob(paths)))
    for statement in VIEW_STATEMENTS:
        connection.execute(statement)


def build(paths: Paths | None = None, *, database: Path | None = None) -> Path:
    """Create (or refresh) the DuckDB database holding the semantic layer."""
    resolved = paths or data_paths()
    resolved.ensure()
    partitions = sorted(resolved.processed.glob("season=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(
            f"no processed partitions under {resolved.processed}. Run `fourthdown etl` first."
        )
    target = database or resolved.database
    target.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(target)) as connection:
        create_views(connection, resolved)
        rows = connection.execute("SELECT count(*) FROM plays").fetchone()
        LOGGER.info("warehouse at %s covers %s plays in %d seasons", target, rows, len(partitions))
    return target


def connect(paths: Paths | None = None, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, defaulting to read-only because every query path is a reader."""
    resolved = paths or data_paths()
    if not resolved.database.exists():
        raise FileNotFoundError(
            f"warehouse missing at {resolved.database}. Run `fourthdown warehouse` first."
        )
    return duckdb.connect(str(resolved.database), read_only=read_only)
