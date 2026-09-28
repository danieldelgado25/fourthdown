"""Pre-snap feature frames and the season splits every model in this package trains on.

Two rules hold everywhere here.

*Pre-snap only.* A feature is admissible if a coach standing on the sideline knows it
before the ball is snapped. Yards gained, EPA, and the outcome of the play are not
features, they are what we are predicting the consequences of.

*No borrowed models.* ``schema.LEAKAGE_COLUMNS`` (``wp``, ``vegas_wp``, ``epa``,
``xpass``, ...) are nflfastR's own fitted outputs over these same plays. Training on
them would produce a model that scores well by copying another model. They are loaded
anyway, but only ever as the baseline column in a results table --
:func:`feature_matrix` raises if one reaches the feature list.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

from fourthdown.data.schema import LEAKAGE_COLUMNS

LOGGER = logging.getLogger(__name__)

WP_FEATURES: tuple[str, ...] = (
    "yardline_100",
    "down",
    "ydstogo",
    "goal_to_go",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_is_home",
    "posteam_spread",
)
"""State of the game from the offense's point of view.

``posteam_spread`` is the closing point spread signed for the offense, the one piece of
information here that is not on the scoreboard. It is pre-game rather than in-play, so
it leaks nothing about how this drive ends, and it is what separates a win-probability
model from a score-differential lookup table in week 1.
"""

PLAYCALL_FEATURES: tuple[str, ...] = (
    "yardline_100",
    "down",
    "ydstogo",
    "goal_to_go",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_is_home",
    "qtr",
    "shotgun",
    "no_huddle",
)
"""Everything the defense can also see before the snap, including formation.

``shotgun`` and ``no_huddle`` are observable pre-snap -- a linebacker reads them -- so
they belong in a model that asks "what can the defense anticipate?".
"""

CONVERSION_FEATURES: tuple[str, ...] = (
    "yardline_100",
    "ydstogo",
    "goal_to_go",
    "down",
    "posteam_spread",
)

DEFAULT_TRAIN_SEASONS: range = range(2009, 2019)
DEFAULT_VALID_SEASONS: range = range(2019, 2021)
DEFAULT_TEST_SEASONS: range = range(2021, 2100)
"""Split by season, not at random.

Randomly splitting plays would put the first and third quarter of the same game on both
sides of the line, and the outcome of a game is shared by all ~150 of its plays, so a
random split measures memorisation. Splitting by season also matches how the model would
be used: fit on the past, asked about a season it has never seen.
"""

BASE_SQL = """
SELECT
    game_id,
    play_id,
    season,
    week,
    season_type,
    posteam,
    defteam,
    qtr,
    down,
    ydstogo,
    yardline_100,
    goal_to_go::INT AS goal_to_go,
    game_seconds_remaining,
    half_seconds_remaining,
    score_differential,
    posteam_timeouts_remaining,
    defteam_timeouts_remaining,
    posteam_is_home::INT AS posteam_is_home,
    shotgun,
    no_huddle,
    CASE WHEN posteam_is_home THEN spread_line ELSE -spread_line END AS posteam_spread,
    posteam_won::INT AS posteam_won,
    is_designed_play,
    is_neutral_script,
    is_pass_call::INT AS is_pass_call,
    pass_play,
    first_down::INT AS first_down,
    touchdown::INT AS touchdown,
    field_goal_attempt,
    field_goal_result,
    kick_distance,
    punt_attempt,
    epa,
    wp,
    vegas_wp,
    xpass
FROM plays
WHERE {where}
"""


@dataclass(frozen=True)
class Split:
    """One season-contiguous slice of the feature frame."""

    name: str
    frame: pl.DataFrame

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.frame["season"].to_list())))

    def __len__(self) -> int:
        return self.frame.height


@dataclass(frozen=True)
class Dataset:
    """Train/validation/test frames plus the feature list they were built for."""

    features: tuple[str, ...]
    target: str
    train: Split
    valid: Split
    test: Split

    def summary(self) -> str:
        parts = [
            f"{split.name} {len(split):,} rows ({split.seasons[0]}-{split.seasons[-1]})"
            if len(split)
            else f"{split.name} empty"
            for split in (self.train, self.valid, self.test)
        ]
        return ", ".join(parts)


def load_plays(
    connection: duckdb.DuckDBPyConnection,
    *,
    where: str = "posteam IS NOT NULL",
    seasons: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Pull the modelling columns out of the warehouse as a Polars frame."""
    clause = where
    if seasons is not None:
        listed = ", ".join(str(int(season)) for season in seasons)
        clause = f"({clause}) AND season IN ({listed})"
    frame = connection.execute(BASE_SQL.format(where=clause)).pl()
    LOGGER.info("loaded %s plays for modelling", f"{frame.height:,}")
    return frame


def split_by_season(
    frame: pl.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    train: Sequence[int] = DEFAULT_TRAIN_SEASONS,
    valid: Sequence[int] = DEFAULT_VALID_SEASONS,
    test: Sequence[int] = DEFAULT_TEST_SEASONS,
) -> Dataset:
    """Cut the frame into three season-disjoint splits, dropping rows with gaps."""
    needed = [*dict.fromkeys([*features, target])]
    usable = frame.drop_nulls(subset=needed)
    dropped = frame.height - usable.height
    if dropped:
        LOGGER.info("dropped %s rows with missing features or target", f"{dropped:,}")

    def slice_for(name: str, seasons: Sequence[int]) -> Split:
        wanted = set(int(season) for season in seasons)
        return Split(name=name, frame=usable.filter(pl.col("season").is_in(wanted)))

    return Dataset(
        features=tuple(features),
        target=target,
        train=slice_for("train", train),
        valid=slice_for("valid", valid),
        test=slice_for("test", test),
    )


def feature_matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Float32 design matrix, with a loud failure if a fitted-model column sneaks in."""
    leaking = sorted(set(features) & LEAKAGE_COLUMNS)
    if leaking:
        raise ValueError(f"{leaking} are nflfastR model outputs; they are baselines, not features")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame.select(features).to_numpy().astype(np.float32)


def target_vector(frame: pl.DataFrame, target: str) -> np.ndarray:
    return frame.select(target).to_numpy().ravel().astype(np.float32)
