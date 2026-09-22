"""Column selection for the processed play table.

The nflverse release carries 372 columns per season. Most are per-player credit
columns (four assisted-tackler slots, lateral recovery slots) that no question in this
project asks about. The subset below is the contract the rest of the codebase codes
against: everything the SQL layer exposes, the narrative generator reads, or a model
trains on.
"""

from __future__ import annotations

IDENTITY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "posteam_type",
    "fixed_drive",
    "fixed_drive_result",
    "series",
    "series_result",
    "qtr",
    "game_half",
)

SITUATION_COLUMNS: tuple[str, ...] = (
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "score_differential",
    "posteam_score",
    "defteam_score",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "shotgun",
    "no_huddle",
)

PLAY_COLUMNS: tuple[str, ...] = (
    "play_type",
    "desc",
    "yards_gained",
    "pass",
    "rush",
    "qb_dropback",
    "qb_scramble",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "play_deleted",
    "special",
    "penalty",
    "pass_length",
    "pass_location",
    "run_location",
    "run_gap",
    "air_yards",
    "yards_after_catch",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "sack",
    "fumble_lost",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "third_down_converted",
    "fourth_down_converted",
    "fourth_down_failed",
    "field_goal_attempt",
    "field_goal_result",
    "kick_distance",
    "punt_attempt",
    "extra_point_attempt",
    "two_point_attempt",
)

PLAYER_COLUMNS: tuple[str, ...] = (
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "kicker_player_name",
    "punter_player_name",
)

GAME_CONTEXT_COLUMNS: tuple[str, ...] = (
    "home_score",
    "away_score",
    "result",
    "total",
    "spread_line",
    "total_line",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium",
    "home_coach",
    "away_coach",
)

MODEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "ep",
    "epa",
    "qb_epa",
    "wp",
    "def_wp",
    "wpa",
    "vegas_wp",
    "vegas_wpa",
    "cp",
    "cpoe",
    "success",
    "xpass",
    "pass_oe",
    "xyac_epa",
    "xyac_mean_yardage",
)
"""Outputs of nflfastR's own fitted models, kept for analytics and as baselines."""

LEAKAGE_COLUMNS: frozenset[str] = frozenset(MODEL_OUTPUT_COLUMNS)
"""Never train on these.

They are produced by models fitted on the same plays (and, for ``wp``/``epa``, on
outcome-adjacent information), so including them as features yields an inflated score
that says nothing about the model we built. They belong in the comparison column of the
results table, not in the feature matrix.
"""

SOURCE_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS
    + SITUATION_COLUMNS
    + PLAY_COLUMNS
    + PLAYER_COLUMNS
    + GAME_CONTEXT_COLUMNS
    + MODEL_OUTPUT_COLUMNS
)

RENAMES: dict[str, str] = {
    "desc": "play_desc",
    "pass": "pass_play",
    "rush": "rush_play",
    "special": "special_teams",
}
"""``desc`` collides with SQL's ``DESC``; ``pass`` and ``rush`` read as verbs in queries."""


def rename(column: str) -> str:
    return RENAMES.get(column, column)


DERIVED_COLUMNS: tuple[str, ...] = (
    "drive_id",
    "posteam_is_home",
    "posteam_won",
    "distance_bucket",
    "field_zone",
    "score_state",
    "is_early_down",
    "is_two_minute",
    "is_neutral_script",
    "is_garbage_time",
    "is_designed_play",
    "is_pass_call",
)

PROCESSED_COLUMNS: tuple[str, ...] = (
    tuple(rename(column) for column in SOURCE_COLUMNS) + DERIVED_COLUMNS
)

INT_COLUMNS: tuple[str, ...] = (
    "play_id",
    "season",
    "week",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "fixed_drive",
    "series",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "kick_distance",
    "posteam_score",
    "defteam_score",
    "score_differential",
)

BOOL_COLUMNS: tuple[str, ...] = (
    "goal_to_go",
    "shotgun",
    "no_huddle",
    "pass_play",
    "rush_play",
    "qb_dropback",
    "qb_scramble",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "play_deleted",
    "special_teams",
    "penalty",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "sack",
    "fumble_lost",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "third_down_converted",
    "fourth_down_converted",
    "fourth_down_failed",
    "field_goal_attempt",
    "punt_attempt",
    "extra_point_attempt",
    "two_point_attempt",
    "div_game",
    "success",
)
