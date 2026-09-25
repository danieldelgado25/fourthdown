"""The schema card: what the model is told about the warehouse before writing SQL.

Column names and types come from DuckDB itself, so the card cannot drift from the views.
Everything a name does not convey — that `yardline_100` counts down to the opponent's
goal line, that `is_pass_call` is NULL on kneels, that there is no `SD` — is curated here,
because those are exactly the things a model gets wrong when left to guess.

The card is deliberately one screen per view rather than a dump of all 90 `plays` columns:
prompt length is the budget, and an unused column is a hallucination waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from fourthdown.data.warehouse import VIEW_NAMES

VIEW_PURPOSE: dict[str, str] = {
    "plays": ("One row per play, the grain everything else aggregates. Regular and post season."),
    "games": "One row per game with final score, venue, weather, and betting lines.",
    "drives": "One row per drive with starting field position, result, and EPA.",
    "team_game": "One row per team per game: pass rate, EPA splits, situational conversions.",
    "player_game": "One row per player per game per role ('passer', 'rusher', 'receiver').",
}

# Only the columns whose meaning is not obvious from the name, or that are easy to misuse.
COLUMN_NOTES: dict[str, str] = {
    "yardline_100": "yards from the opponent's goal line: 1 = about to score, 99 = backed up",
    "ydstogo": "yards needed for a first down",
    "posteam": "team with the ball (offense); defteam is the other one",
    "epa": "expected points added by this play, nflfastR's model; higher is better for posteam",
    "wp": "posteam win probability before the snap",
    "success": "true when epa > 0",
    "is_designed_play": (
        "true for a genuine run/pass decision: excludes kneels, spikes, aborted snaps, and "
        "special teams. Filter on this for any tendency or efficiency question"
    ),
    "is_pass_call": (
        "true when the designed play was a pass (sacks and scrambles count as passes); "
        "NULL when is_designed_play is false"
    ),
    "is_neutral_script": (
        "win probability 0.2-0.8, first three quarters, outside two minutes. Use it when the "
        "question is about what a team likes to do, rather than what the scoreboard forced"
    ),
    "is_garbage_time": "fourth quarter with win probability outside 0.05-0.95",
    "is_early_down": "down is 1 or 2",
    "is_two_minute": "120 seconds or fewer left in the half",
    "distance_bucket": "'short' (<=2), 'medium' (3-6), 'long' (7-10), 'very_long' (>10)",
    "field_zone": "'red_zone', 'opponent_territory', 'midfield', 'own_territory', 'backed_up'",
    "score_state": (
        "'trailing_big', 'trailing_two_scores', 'trailing_one_score', 'tied', "
        "'leading_one_score', 'leading_two_scores', 'leading_big'"
    ),
    "season_type": "'REG' or 'POST'",
    "posteam_won": "true when the team with the ball won the game; NULL on ties",
    "play_desc": "free-text description of the play",
    "play_type": (
        "raw nflverse type: 'pass', 'run', 'punt', 'field_goal', 'kickoff', 'extra_point', "
        "'qb_kneel', 'qb_spike', 'no_play' (penalty), or NULL for administrative rows"
    ),
    "home_margin": "home_score - away_score",
    "neutral_pass_rate": "pass rate restricted to neutral game script",
    "role": "'passer', 'rusher', or 'receiver'",
    "cpoe": "completion percentage over expected, passers only",
    "drive_result": "how the drive ended, e.g. 'Touchdown', 'Punt', 'Field goal', 'Turnover'",
}

# Same name, different meaning once aggregated.
VIEW_COLUMN_NOTES: dict[tuple[str, str], str] = {
    ("drives", "epa"): "total EPA of the drive",
    ("drives", "posteam"): "team on offense for the drive",
    ("player_game", "plays"): "plays in which the player filled this role",
    ("player_game", "yards"): "yards gained on those plays, not official passing/rushing yards",
    ("team_game", "plays"): "all plays including special teams; designed_plays excludes them",
}

# Columns worth listing for `plays`, in the order a human would read them. The other ~60
# exist and are queryable, but naming them all costs prompt budget and invites misuse.
PLAYS_HIGHLIGHT: tuple[str, ...] = (
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "play_type",
    "play_desc",
    "yards_gained",
    "epa",
    "wp",
    "success",
    "touchdown",
    "first_down",
    "shotgun",
    "no_huddle",
    "passer_player_name",
    "rusher_player_name",
    "receiver_player_name",
    "is_designed_play",
    "is_pass_call",
    "is_neutral_script",
    "is_garbage_time",
    "is_early_down",
    "is_two_minute",
    "distance_bucket",
    "field_zone",
    "score_state",
    "posteam_won",
)

RULES = """\
Rules:
- DuckDB SQL, one SELECT statement, no semicolon, no DDL or DML.
- Only these views exist. Do not invent tables, columns, or a `player_season` view.
- Any question about tendency, play calling, or efficiency per play must filter
  `is_designed_play`, and should filter `is_neutral_script` when it asks what a team
  prefers rather than what it did.
- Team abbreviations are the current ones for all history: use LAC (not SD), LA (not
  STL), LV (not OAK), JAX, WAS, ARI.
- The season is the year it started: the February 2024 Super Bowl is season 2023.
- Rates are averages of booleans: `avg(is_pass_call::INT)`.
- Add ORDER BY and LIMIT when the question asks for the most, fewest, or a ranking."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    note: str | None


def _columns(connection: duckdb.DuckDBPyConnection, view: str) -> list[Column]:
    rows = connection.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [view],
    ).fetchall()
    return [
        Column(name, _short_type(kind), VIEW_COLUMN_NOTES.get((view, name), COLUMN_NOTES.get(name)))
        for name, kind in rows
    ]


def _short_type(duckdb_type: str) -> str:
    mapping = {
        "BIGINT": "int",
        "INTEGER": "int",
        "HUGEINT": "int",
        "DOUBLE": "float",
        "FLOAT": "float",
        "VARCHAR": "text",
        "BOOLEAN": "bool",
        "DATE": "date",
    }
    return mapping.get(duckdb_type.upper(), duckdb_type.lower())


def _render_view(view: str, columns: list[Column]) -> str:
    lines = [f"{view} — {VIEW_PURPOSE[view]}"]
    for column in columns:
        note = f"  # {column.note}" if column.note else ""
        lines.append(f"  {column.name} {column.type}{note}")
    return "\n".join(lines)


def view_columns(connection: duckdb.DuckDBPyConnection) -> dict[str, tuple[str, ...]]:
    """Column names per view, for repair feedback. `plays` is trimmed to the card's subset."""
    columns: dict[str, tuple[str, ...]] = {}
    for view in VIEW_NAMES:
        names = [column.name for column in _columns(connection, view)]
        if view == "plays":
            names = [name for name in PLAYS_HIGHLIGHT if name in set(names)]
        columns[view] = tuple(names)
    return columns


def build(connection: duckdb.DuckDBPyConnection) -> str:
    """Render the schema card from the live warehouse catalogue."""
    blocks = []
    for view in VIEW_NAMES:
        columns = _columns(connection, view)
        if view == "plays":
            highlighted = {column.name: column for column in columns}
            columns = [highlighted[name] for name in PLAYS_HIGHLIGHT if name in highlighted]
            extra = len(highlighted) - len(columns)
            block = _render_view(view, columns)
            block += f"\n  ...and {extra} more raw nflverse columns (penalties, kicking, ids)"
        else:
            block = _render_view(view, columns)
        blocks.append(block)
    coverage = connection.execute("SELECT min(season), max(season), count(*) FROM plays").fetchone()
    if coverage is None:  # pragma: no cover - an aggregate always returns a row
        raise RuntimeError("could not read warehouse coverage")
    first_season, last_season, plays = coverage
    header = f"Warehouse: {plays:,} plays, seasons {first_season}-{last_season}."
    return "\n\n".join([header, *blocks, RULES])
