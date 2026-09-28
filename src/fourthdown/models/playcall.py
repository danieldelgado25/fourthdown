"""Run or pass, and what being readable costs you.

Predicting the call is the easy half; a gradient-boosted model on pre-snap state beats
the "always pass" baseline comfortably. The half worth putting in front of someone is
the follow-up question: *when the defense can see it coming, does the offense actually
lose anything?*

The measurement is per team-season, on early-down neutral-script plays only (leading by
three scores in the fourth quarter makes everybody run, and that is game state, not
tendency):

    predictability = share of plays where the model's call was the call

That is the number a defensive coordinator cares about -- how often guessing gets you
the right answer -- and it is reported next to mean confidence ``max(p, 1 - p)``, which
says how sure the guess was. Pair either with the same team-season's EPA per play and
the correlation is a finding rather than a metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from fourthdown.models.features import Dataset, feature_matrix, target_vector
from fourthdown.models.metrics import ProbabilityScore, score

LOGGER = logging.getLogger(__name__)

MAX_ITER = 400
LEARNING_RATE = 0.06
MAX_DEPTH = 6
SEED = 7
MIN_PLAYS = 100
ARTIFACT = "playcall.joblib"


@dataclass(frozen=True)
class TeamTendency:
    """One team-season's readability and what it scored."""

    season: int
    team: str
    plays: int
    pass_rate: float
    predictability: float
    confidence: float
    epa_per_play: float


@dataclass(frozen=True)
class PredictabilityReport:
    """Team-season tendencies plus the correlation everyone asks about."""

    rows: tuple[TeamTendency, ...]
    correlation: float

    def most_predictable(self, limit: int = 5) -> tuple[TeamTendency, ...]:
        return tuple(sorted(self.rows, key=lambda row: -row.predictability)[:limit])

    def least_predictable(self, limit: int = 5) -> tuple[TeamTendency, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.predictability)[:limit])


class PlayCallModel:
    """P(pass) on a designed play, given what the defense can see pre-snap."""

    def __init__(self, features: tuple[str, ...], estimator: HistGradientBoostingClassifier):
        self.features = features
        self.estimator = estimator

    @classmethod
    def fit(cls, dataset: Dataset) -> PlayCallModel:
        estimator = HistGradientBoostingClassifier(
            max_iter=MAX_ITER,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )
        estimator.fit(
            feature_matrix(dataset.train.frame, dataset.features),
            target_vector(dataset.train.frame, dataset.target),
        )
        LOGGER.info("play-call model fit on %s plays", f"{len(dataset.train):,}")
        return cls(dataset.features, estimator)

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = feature_matrix(frame, self.features)
        return self.estimator.predict_proba(matrix)[:, 1].astype(np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": list(self.features), "estimator": self.estimator}, path)

    @classmethod
    def load(cls, path: Path) -> PlayCallModel:
        payload = joblib.load(path)
        return cls(tuple(payload["features"]), payload["estimator"])


def majority_baseline(dataset: Dataset) -> ProbabilityScore:
    """Always guess the training set's most common call, at its base rate.

    This is the number the model has to beat to be interesting: the NFL passes on about
    three of every five designed plays, so 58-62% accuracy is free.
    """
    rate = float(target_vector(dataset.train.frame, dataset.target).mean())
    truth = target_vector(dataset.test.frame, dataset.target)
    return score(truth, np.full(truth.shape, rate))


def xpass_baseline(dataset: Dataset) -> ProbabilityScore | None:
    """nflfastR's own ``xpass``, scored on the same rows, when it is available."""
    frame = dataset.test.frame.drop_nulls(subset=["xpass"])
    if frame.is_empty():
        return None
    return score(target_vector(frame, dataset.target), frame["xpass"].to_numpy())


def predictability(
    model: PlayCallModel, frame: pl.DataFrame, *, min_plays: int = MIN_PLAYS
) -> PredictabilityReport:
    """Score every play, then roll up to team-seasons with at least ``min_plays``."""
    if frame.is_empty():
        return PredictabilityReport(rows=(), correlation=float("nan"))
    probability = model.predict(frame)
    scored = frame.with_columns(
        pl.Series("pass_probability", probability),
        pl.Series("call_confidence", np.maximum(probability, 1.0 - probability)),
    ).with_columns(
        ((pl.col("pass_probability") >= 0.5).cast(pl.Int8) == pl.col("is_pass_call"))
        .cast(pl.Float64)
        .alias("call_guessed")
    )
    grouped = (
        scored.group_by(["season", "posteam"])
        .agg(
            pl.len().alias("plays"),
            pl.col("is_pass_call").mean().alias("pass_rate"),
            pl.col("call_guessed").mean().alias("predictability"),
            pl.col("call_confidence").mean().alias("confidence"),
            pl.col("epa").mean().alias("epa_per_play"),
        )
        .filter(pl.col("plays") >= min_plays)
        .sort(["season", "posteam"])
    )
    rows = tuple(
        TeamTendency(
            season=int(row["season"]),
            team=str(row["posteam"]),
            plays=int(row["plays"]),
            pass_rate=float(row["pass_rate"]),
            predictability=float(row["predictability"]),
            confidence=float(row["confidence"]),
            epa_per_play=float(row["epa_per_play"] if row["epa_per_play"] is not None else 0.0),
        )
        for row in grouped.iter_rows(named=True)
    )
    if len(rows) < 3:
        return PredictabilityReport(rows=rows, correlation=float("nan"))
    readable = np.array([row.predictability for row in rows])
    epa = np.array([row.epa_per_play for row in rows])
    if readable.std() == 0.0 or epa.std() == 0.0:
        return PredictabilityReport(rows=rows, correlation=float("nan"))
    correlation = float(np.corrcoef(readable, epa)[0, 1])
    return PredictabilityReport(rows=rows, correlation=correlation)


def tendency_table(rows: tuple[TeamTendency, ...]) -> str:
    lines = [
        "| season | team | plays | pass rate | predictability | confidence | EPA/play |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {row.season} | {row.team} | {row.plays:,} | {row.pass_rate:.1%} | "
        f"{row.predictability:.3f} | {row.confidence:.3f} | {row.epa_per_play:+.3f} |"
        for row in rows
    ]
    return "\n".join(lines)
