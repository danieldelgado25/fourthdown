"""Pandera contracts for the raw nflverse file and the processed weekly partitions.

The raw contract is structural, because the raw file is someone else's: the columns the ETL
cannot do without are present, the ones it reads as numbers are numbers (a cast would turn
text into silent nulls), every play has a key and no key repeats, and the file holds the
season it is named for.

The processed contract is the table every later stage trusts: exactly the contracted
columns, in order, with their types; values inside the ranges the game allows (and that
2009-2025 never leaves); labels from known sets; and only the partition's own season and
week. The bounds come from profiling all 816,856 plays, with room to spare: they catch a
corrupted or misparsed file, not an unusual game.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from fourthdown.data import schema
from fourthdown.quality import DataQualityError

GAME_ID_PATTERN = r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$"

RANGES: dict[str, tuple[float, float]] = {
    "season": (1999, 2100),
    "week": (1, 22),
    "play_id": (1, 10_000),
    "qtr": (1, 6),
    "down": (1, 4),
    "ydstogo": (0, 99),
    "yardline_100": (0, 100),
    "fixed_drive": (1, 60),
    "series": (1, 150),
    "quarter_seconds_remaining": (0, 900),
    "half_seconds_remaining": (0, 1800),
    "game_seconds_remaining": (0, 3600),
    "score_differential": (-80, 80),
    "posteam_score": (0, 100),
    "defteam_score": (0, 100),
    "home_score": (0, 100),
    "away_score": (0, 100),
    "posteam_timeouts_remaining": (0, 3),
    "defteam_timeouts_remaining": (0, 3),
    "yards_gained": (-100, 110),
    "air_yards": (-100, 110),
    "yards_after_catch": (-100, 110),
    "kick_distance": (-10, 100),
    "temp": (-40, 130),
    "wind": (0, 100),
    "spread_line": (-40, 40),
    "total_line": (20, 90),
    "wp": (0, 1),
    "def_wp": (0, 1),
    "vegas_wp": (0, 1),
    "cp": (0, 1),
    "xpass": (0, 1),
}
"""Inclusive bounds. Observed 2009-2025: down 1-4, qtr 1-6, ydstogo 0-50, yards_gained
-38 to 99, score_differential -59 to 59, temp -6 to 97, spread -18 to 27."""

ALLOWED: dict[str, tuple[str, ...]] = {
    "season_type": ("REG", "POST"),
    "posteam_type": ("home", "away"),
    "game_half": ("Half1", "Half2", "Overtime"),
    "play_type": (
        "pass",
        "run",
        "no_play",
        "kickoff",
        "punt",
        "extra_point",
        "field_goal",
        "qb_kneel",
        "qb_spike",
    ),
    "field_goal_result": ("made", "missed", "blocked"),
    "roof": ("closed", "dome", "open", "outdoors"),
    "distance_bucket": ("short", "medium", "long", "very_long"),
    "field_zone": ("red_zone", "opponent_territory", "midfield", "own_territory", "backed_up"),
    "score_state": (
        "trailing_big",
        "trailing_two_scores",
        "trailing_one_score",
        "tied",
        "leading_one_score",
        "leading_two_scores",
        "leading_big",
    ),
}

NOT_NULL = (
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "drive_id",
    "qtr",
    "is_designed_play",
)

DTYPES: dict[str, type[pl.DataType]] = {
    **{column: pl.Int32 for column in schema.INT_COLUMNS},
    **{column: pl.Boolean for column in (*schema.BOOL_COLUMNS, *schema.DERIVED_BOOL_COLUMNS)},
    **{column: pl.Float64 for column in schema.FLOAT_COLUMNS},
    **{column: pl.Date for column in schema.DATE_COLUMNS},
    **{column: pl.String for column in schema.STRING_COLUMNS},
}

RAW_NUMERIC = tuple(
    column
    for column in schema.SOURCE_COLUMNS
    if schema.rename(column) in {*schema.INT_COLUMNS, *schema.FLOAT_COLUMNS, *schema.BOOL_COLUMNS}
)
"""Raw columns the ETL reads as numbers (nflverse stores its flags as 0/1 numbers)."""


def _checks(column: str) -> list[pa.Check]:
    checks: list[pa.Check] = []
    if column in RANGES:
        low, high = RANGES[column]
        checks.append(pa.Check.in_range(low, high, error=f"between {low:g} and {high:g}"))
    if column in ALLOWED:
        allowed = ALLOWED[column]
        checks.append(pa.Check.isin(allowed, error=f"one of {', '.join(allowed)}"))
    if column == "game_id":
        checks.append(pa.Check.str_matches(GAME_ID_PATTERN, error="shaped like 2026_10_KC_BUF"))
    return checks


def processed_schema(season: int, week: int) -> pa.DataFrameSchema:
    """The contract for the partition ``season``/``week``."""
    columns = {
        column: pa.Column(
            DTYPES[column],
            checks=[
                *_checks(column),
                *(
                    [pa.Check.equal_to(season, error=f"equal to {season}")]
                    if column == "season"
                    else []
                ),
                *([pa.Check.equal_to(week, error=f"equal to {week}")] if column == "week" else []),
            ],
            nullable=column not in NOT_NULL,
        )
        for column in schema.PROCESSED_COLUMNS
    }
    return pa.DataFrameSchema(
        columns,
        strict=True,
        ordered=True,
        unique=["game_id", "play_id"],
        name=f"processed {season}-{week:02d}",
    )


def raw_schema(season: int) -> pa.DataFrameSchema:
    """The structural contract for a raw nflverse season file (or a week of it)."""
    columns = {
        column: pa.Column(
            checks=[pa.Check.equal_to(season, error=f"equal to {season}")]
            if column == "season"
            else [],
            nullable=column not in ("game_id", "play_id", "season", "week"),
            required=column in schema.REQUIRED_SOURCE_COLUMNS,
        )
        for column in schema.SOURCE_COLUMNS
    }
    return pa.DataFrameSchema(columns, strict=False, unique=["game_id", "play_id"])


def _numeric_types(frame: pl.DataFrame) -> list[str]:
    return [
        f"{column}: expected numbers, got {frame.schema[column]}; it would load as nulls"
        for column in RAW_NUMERIC
        if column in frame.columns
        and not (frame.schema[column].is_numeric() or frame.schema[column] == pl.Null)
    ]


def _examples(values: Iterable[object], limit: int = 3) -> str:
    shown: list[str] = []
    for value in values:
        text = str(value)
        if text not in shown:
            shown.append(text)
        if len(shown) == limit:
            break
    return ", ".join(shown)


def _describe(error: SchemaErrors | SchemaError) -> list[str]:
    """Pandera's failure cases as one line per (column, check), with example values."""
    cases = error.failure_cases
    if not isinstance(cases, pl.DataFrame) or cases.is_empty():
        return [str(error).splitlines()[0]]
    problems: list[str] = []
    grouped = cases.with_columns(pl.col("failure_case").cast(pl.String)).group_by(
        "column", "check", maintain_order=True
    )
    for (column, check), group in grouped:
        examples = _examples(group["failure_case"].to_list())
        rows = group.height
        if check == "column_in_dataframe":
            problems.append(f"missing column {examples}; nflverse may have changed its layout")
        elif check == "column_in_schema":
            problems.append(f"unexpected column {examples}")
        elif check == "column_ordered":
            problems.append(f"columns out of contract order at {examples}")
        elif check.startswith("dtype("):
            problems.append(f"{column}: expected {check[6:-1].strip(chr(39))}, got {examples}")
        elif check == "not_nullable":
            problems.append(f"{column}: {rows:,} null value(s) where one is required")
        elif check in {"multiple_fields_uniqueness", "field_uniqueness"}:
            problems.append(f"(game_id, play_id): {rows:,} rows share a key with another row")
        else:
            problems.append(f"{column}: {rows:,} value(s) not {check} (e.g. {examples})")
    return problems


def _validate(contract: pa.DataFrameSchema, frame: pl.DataFrame) -> list[str]:
    try:
        contract.validate(frame, lazy=True)
    except (SchemaErrors, SchemaError) as error:
        return _describe(error)
    return []


def validate_raw(frame: pl.DataFrame, *, season: int, subject: str) -> None:
    """Raise ``DataQualityError`` if the raw frame breaks the structural contract."""
    problems = _numeric_types(frame) + _validate(raw_schema(season), frame)
    if problems:
        raise DataQualityError(subject, problems)


def validate_processed(frame: pl.DataFrame, *, season: int, week: int) -> None:
    """Raise ``DataQualityError`` if a processed partition breaks the contract."""
    problems = _validate(processed_schema(season, week), frame)
    if problems:
        raise DataQualityError(f"processed partition {season}-{week:02d}", problems)
