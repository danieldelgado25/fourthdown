"""Polars transform from a raw nflverse season file to the processed play table.

The transform is a single lazy plan per season: project the columns we contracted to,
normalise types, and derive the situational buckets that the SQL layer, the narrative
generator, and the models all share. Deriving them once here is what keeps a bucket
definition from drifting between a chart, a query, and a feature matrix.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import polars as pl

from fourthdown.config import Paths, data_paths
from fourthdown.data import schema

LOGGER = logging.getLogger(__name__)

SHORT_YARDAGE = 2
MEDIUM_YARDAGE = 6
LONG_YARDAGE = 10
TWO_MINUTE_SECONDS = 120
NEUTRAL_WP_RANGE = (0.2, 0.8)
GARBAGE_WP_RANGE = (0.05, 0.95)


def _project_source(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Select the contracted columns, filling in any a given season does not publish."""
    available = set(frame.collect_schema().names())
    missing = [column for column in schema.SOURCE_COLUMNS if column not in available]
    if missing:
        LOGGER.info("season file is missing %d contracted columns: %s", len(missing), missing)
    projection = [
        pl.col(column) if column in available else pl.lit(None).alias(column)
        for column in schema.SOURCE_COLUMNS
    ]
    return frame.select(projection).rename(
        {old: new for old, new in schema.RENAMES.items() if old in schema.SOURCE_COLUMNS}
    )


def _normalise_types(frame: pl.LazyFrame) -> pl.LazyFrame:
    casts = [pl.col(column).cast(pl.Int32, strict=False) for column in schema.INT_COLUMNS]
    casts += [(pl.col(column) != 0).alias(column) for column in schema.BOOL_COLUMNS]
    game_date = pl.col("game_date")
    if frame.collect_schema()["game_date"] == pl.Utf8:
        game_date = game_date.str.to_date(strict=False)
    casts.append(game_date.cast(pl.Date, strict=False).alias("game_date"))
    return frame.with_columns(casts)


def _distance_bucket() -> pl.Expr:
    return (
        pl.when(pl.col("down").is_null())
        .then(None)
        .when(pl.col("ydstogo") <= SHORT_YARDAGE)
        .then(pl.lit("short"))
        .when(pl.col("ydstogo") <= MEDIUM_YARDAGE)
        .then(pl.lit("medium"))
        .when(pl.col("ydstogo") <= LONG_YARDAGE)
        .then(pl.lit("long"))
        .otherwise(pl.lit("very_long"))
        .alias("distance_bucket")
    )


def _field_zone() -> pl.Expr:
    """Bucket field position. ``yardline_100`` counts yards to the opponent's goal line."""
    return (
        pl.when(pl.col("yardline_100").is_null())
        .then(None)
        .when(pl.col("yardline_100") <= 20)
        .then(pl.lit("red_zone"))
        .when(pl.col("yardline_100") <= 40)
        .then(pl.lit("opponent_territory"))
        .when(pl.col("yardline_100") <= 60)
        .then(pl.lit("midfield"))
        .when(pl.col("yardline_100") <= 80)
        .then(pl.lit("own_territory"))
        .otherwise(pl.lit("backed_up"))
        .alias("field_zone")
    )


def _score_state() -> pl.Expr:
    differential = pl.col("score_differential")
    return (
        pl.when(differential.is_null())
        .then(None)
        .when(differential <= -17)
        .then(pl.lit("trailing_big"))
        .when(differential <= -9)
        .then(pl.lit("trailing_two_scores"))
        .when(differential <= -1)
        .then(pl.lit("trailing_one_score"))
        .when(differential == 0)
        .then(pl.lit("tied"))
        .when(differential <= 8)
        .then(pl.lit("leading_one_score"))
        .when(differential <= 16)
        .then(pl.lit("leading_two_scores"))
        .otherwise(pl.lit("leading_big"))
        .alias("score_state")
    )


def _posteam_won() -> pl.Expr:
    """Game outcome from the offense's perspective; null on ties and missing results.

    ``result`` is the final home margin, so the offense won when the sign of the margin
    matches whether the offense was the home team.
    """
    home_won = pl.col("result") > 0
    return (
        pl.when(pl.col("result").is_null() | (pl.col("result") == 0))
        .then(None)
        .otherwise(home_won == pl.col("posteam_is_home"))
        .alias("posteam_won")
    )


def _is_designed_play() -> pl.Expr:
    """A pre-snap run/pass decision by the offense, which is what the play-call model predicts.

    Kneels, spikes, aborted snaps, deleted plays, and special teams are excluded: they are
    not a coordinator choosing between run and pass.
    """
    return (
        (pl.col("pass_play") | pl.col("rush_play"))
        & pl.col("down").is_not_null()
        & ~pl.col("qb_kneel").fill_null(False)  # noqa: FBT003
        & ~pl.col("qb_spike").fill_null(False)  # noqa: FBT003
        & ~pl.col("aborted_play").fill_null(False)  # noqa: FBT003
        & ~pl.col("play_deleted").fill_null(False)  # noqa: FBT003
        & ~pl.col("special_teams").fill_null(False)  # noqa: FBT003
    ).alias("is_designed_play")


def _derive(frame: pl.LazyFrame) -> pl.LazyFrame:
    frame = frame.with_columns(
        (pl.col("game_id") + pl.lit("-") + pl.col("fixed_drive").cast(pl.Utf8)).alias("drive_id"),
        (pl.col("posteam") == pl.col("home_team")).alias("posteam_is_home"),
        _distance_bucket(),
        _field_zone(),
        _score_state(),
        pl.col("down").is_in([1, 2]).alias("is_early_down"),
        (pl.col("half_seconds_remaining") <= TWO_MINUTE_SECONDS).alias("is_two_minute"),
        pl.col("wp").is_between(*NEUTRAL_WP_RANGE).alias("_wp_neutral"),
        (~pl.col("wp").is_between(*GARBAGE_WP_RANGE)).alias("_wp_lopsided"),
    )
    frame = frame.with_columns(
        _posteam_won(),
        _is_designed_play(),
        (
            pl.col("_wp_neutral")
            & (pl.col("qtr") <= 3)
            & (pl.col("half_seconds_remaining") > TWO_MINUTE_SECONDS)
        ).alias("is_neutral_script"),
        (pl.col("_wp_lopsided") & (pl.col("qtr") >= 4)).alias("is_garbage_time"),
    )
    return frame.with_columns(
        pl.when(pl.col("is_designed_play"))
        .then(pl.col("pass_play"))
        .otherwise(None)
        .alias("is_pass_call")
    ).drop("_wp_neutral", "_wp_lopsided")


def transform(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Raw season frame to processed play table, ordered and column-stable."""
    plan = _derive(_normalise_types(_project_source(frame)))
    return plan.select(schema.PROCESSED_COLUMNS).sort("game_id", "play_id")


def transform_season(season: int, paths: Paths | None = None) -> Path:
    """Run the transform for one season and write its processed Parquet partition."""
    resolved = paths or data_paths()
    source = resolved.season_raw(season)
    if not source.exists():
        raise FileNotFoundError(
            f"raw season file missing: {source}. Run `fourthdown ingest` first."
        )
    destination = resolved.season_processed(season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("transforming season %s", season)
    table = transform(pl.scan_parquet(source)).collect()
    table.write_parquet(destination, compression="zstd", statistics=True)
    LOGGER.info("wrote %s rows to %s", table.height, destination)
    return destination


def transform_seasons(seasons: Iterable[int], paths: Paths | None = None) -> Iterator[Path]:
    resolved = paths or data_paths()
    resolved.ensure()
    for season in seasons:
        yield transform_season(season, resolved)
