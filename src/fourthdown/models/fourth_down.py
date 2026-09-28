"""Fourth-down decisions: go for it, kick the field goal, or punt.

There is no "fourth-down model" to train. The recommendation is a small decision tree
whose leaves are the win-probability model evaluated at the state each choice leads to,
weighted by how often each choice works:

    WP(go)   = p_convert * WP(1st and 10 at the sticks)
             + (1 - p_convert) * (1 - WP(opponent takes over at the spot))

    WP(kick) = p_make * (1 - WP(opponent at their 25, three points behind))
             + (1 - p_make) * (1 - WP(opponent takes over at the spot of the kick))

    WP(punt) = 1 - WP(opponent takes over where punts from here land)

So three things have to be estimated from the data -- conversion probability, field-goal
probability, and where a punt leaves the opponent -- and the fourth, the value of each
resulting state, comes from :mod:`fourthdown.models.winprob`. Writing it this way keeps
the recommendation explainable: every number in the output is a probability someone can
argue with, not a coefficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from fourthdown.models.winprob import GameState, WinProbabilityModel

LOGGER = logging.getLogger(__name__)

KICK_SNAP_YARDS = 17
"""Seven yards of snap and hold plus ten yards of end zone."""

TOUCHBACK_YARDLINE = 75.0
"""After a score, the receiving team starts at its own 25, which is 75 from our goal."""

MISSED_KICK_FLOOR = 80.0
"""A missed kick gives the ball at the spot, but never worse than the opponent's 20."""

PLAY_SECONDS = 6.0
OPTIONS: tuple[str, ...] = ("go", "field_goal", "punt")

CONVERSION_SQL = """
SELECT
    yardline_100,
    ydstogo,
    coalesce(goal_to_go, false)::INT AS goal_to_go,
    down,
    greatest(coalesce(first_down, false)::INT, coalesce(touchdown, false)::INT) AS converted
FROM plays
WHERE is_designed_play AND down IN (3, 4) AND ydstogo BETWEEN 1 AND 30
  AND yardline_100 IS NOT NULL AND season < {holdout}
"""

FIELD_GOAL_SQL = """
SELECT
    yardline_100 + {snap} AS attempt_distance,
    (field_goal_result = 'made')::INT AS made
FROM plays
WHERE field_goal_attempt AND field_goal_result IS NOT NULL
  AND yardline_100 IS NOT NULL AND season < {holdout}
"""

PUNT_SQL = """
WITH ordered AS (
    SELECT
        game_id,
        play_id,
        posteam,
        yardline_100,
        punt_attempt,
        lead(posteam) OVER (PARTITION BY game_id ORDER BY play_id) AS next_posteam,
        lead(yardline_100) OVER (PARTITION BY game_id ORDER BY play_id) AS next_yardline_100
    FROM plays
    WHERE season < {holdout} AND posteam IS NOT NULL
)
SELECT yardline_100, next_yardline_100 AS opponent_yardline_100
FROM ordered
WHERE punt_attempt AND next_posteam IS NOT NULL AND next_posteam <> posteam
  AND next_yardline_100 IS NOT NULL
"""


@dataclass(frozen=True)
class Option:
    """One choice, its probability of working, and what it is worth."""

    name: str
    win_probability: float
    success_probability: float | None
    detail: str


@dataclass(frozen=True)
class Recommendation:
    """The ranked options for one fourth down."""

    state: GameState
    options: tuple[Option, ...]

    @property
    def best(self) -> Option:
        return self.options[0]

    @property
    def edge(self) -> float:
        """Win-probability gap between the recommendation and the next-best option."""
        if len(self.options) < 2:
            return 0.0
        return self.options[0].win_probability - self.options[1].win_probability

    def render(self) -> str:
        lines = [
            f"4th and {self.state.ydstogo:g} at the {self.state.yardline_100:g}, "
            f"{self.state.score_differential:+g} with "
            f"{self.state.game_seconds_remaining / 60:.1f} minutes left",
            f"-> {self.best.name.replace('_', ' ')} "
            f"(+{self.edge * 100:.1f} win probability points)",
            "",
            "| option | win probability | success | detail |",
            "| --- | --- | --- | --- |",
        ]
        for option in self.options:
            chance = option.success_probability
            success = "-" if chance is None else f"{chance:.1%}"
            lines.append(
                f"| {option.name.replace('_', ' ')} | {option.win_probability:.1%} | "
                f"{success} | {option.detail} |"
            )
        return "\n".join(lines)


class ConversionModel:
    """P(pick up the first down) from distance, field position, and down."""

    FEATURES: tuple[str, ...] = ("yardline_100", "ydstogo", "goal_to_go", "down")

    def __init__(self, estimator: HistGradientBoostingClassifier) -> None:
        self.estimator = estimator

    @classmethod
    def fit(cls, frame: pl.DataFrame) -> ConversionModel:
        estimator = HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.08, random_state=7
        )
        estimator.fit(frame.select(cls.FEATURES).to_numpy(), frame["converted"].to_numpy())
        return cls(estimator)

    def probability(self, state: GameState) -> float:
        row = np.array(
            [[state.yardline_100, state.ydstogo, float(state.goal_to_go), 4.0]], dtype=np.float64
        )
        return float(self.estimator.predict_proba(row)[0, 1])


class FieldGoalModel:
    """P(make) as a function of attempt distance."""

    def __init__(self, estimator: HistGradientBoostingClassifier) -> None:
        self.estimator = estimator

    @classmethod
    def fit(cls, frame: pl.DataFrame) -> FieldGoalModel:
        estimator = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.08, random_state=7
        )
        estimator.fit(frame.select("attempt_distance").to_numpy(), frame["made"].to_numpy())
        return cls(estimator)

    def probability(self, yardline_100: float) -> float:
        distance = np.array([[yardline_100 + KICK_SNAP_YARDS]], dtype=np.float64)
        return float(self.estimator.predict_proba(distance)[0, 1])


class PuntModel:
    """Where a punt from here leaves the other team, taken straight from history.

    A parametric punt model would be a worse version of the empirical average: punts
    from the opponent's 40 are a different animal (coffin corner, high risk of a
    touchback) than punts from your own 20, and the data already knows the shape.
    """

    def __init__(self, table: dict[int, float], fallback: float) -> None:
        self.table = table
        self.fallback = fallback

    @classmethod
    def fit(cls, frame: pl.DataFrame, *, bucket: int = 5) -> PuntModel:
        if frame.is_empty():
            raise ValueError("no punts to fit on")
        grouped = (
            frame.with_columns((pl.col("yardline_100") // bucket).cast(pl.Int64).alias("bucket"))
            .group_by("bucket")
            .agg(pl.col("opponent_yardline_100").mean().alias("landing"), pl.len().alias("punts"))
            .filter(pl.col("punts") >= 25)
            .sort("bucket")
        )
        table = {
            int(row["bucket"]) * bucket: float(row["landing"])
            for row in grouped.iter_rows(named=True)
        }
        average = frame["opponent_yardline_100"].mean()
        fallback = float(average) if isinstance(average, int | float) else TOUCHBACK_YARDLINE
        return cls(table, fallback)

    def opponent_yardline(self, yardline_100: float) -> float:
        if not self.table:
            return self.fallback
        key = min(self.table, key=lambda edge: abs(edge - yardline_100))
        return self.table[key]


@dataclass(frozen=True)
class FourthDownAdvisor:
    """The win-probability model plus the three estimators it needs at the leaves."""

    win_probability: WinProbabilityModel
    conversion: ConversionModel
    field_goal: FieldGoalModel
    punt: PuntModel

    def recommend(self, state: GameState) -> Recommendation:
        options = [self._go(state), self._kick(state), self._punt(state)]
        options.sort(key=lambda option: option.win_probability, reverse=True)
        return Recommendation(state=state, options=tuple(options))

    def _go(self, state: GameState) -> Option:
        convert = self.conversion.probability(state)
        gained = replace(
            state,
            down=1,
            ydstogo=min(10.0, max(state.yardline_100 - state.ydstogo, 1.0)),
            yardline_100=max(state.yardline_100 - state.ydstogo, 1.0),
            goal_to_go=(state.yardline_100 - state.ydstogo) <= 10,
            game_seconds_remaining=max(state.game_seconds_remaining - PLAY_SECONDS, 0.0),
            half_seconds_remaining=max(state.half_clock() - PLAY_SECONDS, 0.0),
        )
        stopped = state.possession_flip(100.0 - state.yardline_100)
        value = convert * self.win_probability.probability(gained) + (1 - convert) * (
            1 - self.win_probability.probability(stopped)
        )
        return Option(
            name="go",
            win_probability=value,
            success_probability=convert,
            detail=f"convert to 1st and {gained.ydstogo:g}, else they take over at their "
            f"{100 - stopped.yardline_100:g}",
        )

    def _kick(self, state: GameState) -> Option:
        make = self.field_goal.probability(state.yardline_100)
        scored = state.possession_flip(TOUCHBACK_YARDLINE)
        scored = replace(scored, score_differential=-(state.score_differential + 3))
        missed_spot = min(100.0 - (state.yardline_100 + 7.0), MISSED_KICK_FLOOR)
        missed = state.possession_flip(missed_spot)
        value = make * (1 - self.win_probability.probability(scored)) + (1 - make) * (
            1 - self.win_probability.probability(missed)
        )
        distance = state.yardline_100 + KICK_SNAP_YARDS
        return Option(
            name="field_goal",
            win_probability=value,
            success_probability=make,
            detail=f"{distance:g}-yard attempt; a miss hands them the ball at their "
            f"{100 - missed_spot:g}",
        )

    def _punt(self, state: GameState) -> Option:
        landing = self.punt.opponent_yardline(state.yardline_100)
        received = state.possession_flip(landing)
        value = 1 - self.win_probability.probability(received)
        return Option(
            name="punt",
            win_probability=value,
            success_probability=None,
            detail=f"they start at their {100 - landing:.0f} on average",
        )


def training_frames(
    connection: duckdb.DuckDBPyConnection, *, holdout_season: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Conversion, field-goal, and punt frames, all from seasons before the holdout."""
    conversions = connection.execute(CONVERSION_SQL.format(holdout=holdout_season)).pl()
    goals = connection.execute(
        FIELD_GOAL_SQL.format(snap=KICK_SNAP_YARDS, holdout=holdout_season)
    ).pl()
    punts = connection.execute(PUNT_SQL.format(holdout=holdout_season)).pl()
    LOGGER.info(
        "fourth-down components: %s conversions, %s kicks, %s punts",
        f"{conversions.height:,}",
        f"{goals.height:,}",
        f"{punts.height:,}",
    )
    return conversions, goals, punts


def build(
    connection: duckdb.DuckDBPyConnection,
    win_probability: WinProbabilityModel,
    *,
    holdout_season: int,
) -> FourthDownAdvisor:
    conversions, goals, punts = training_frames(connection, holdout_season=holdout_season)
    return FourthDownAdvisor(
        win_probability=win_probability,
        conversion=ConversionModel.fit(conversions),
        field_goal=FieldGoalModel.fit(goals),
        punt=PuntModel.fit(punts),
    )


def save_components(advisor: FourthDownAdvisor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "conversion": advisor.conversion.estimator,
            "field_goal": advisor.field_goal.estimator,
            "punt_table": advisor.punt.table,
            "punt_fallback": advisor.punt.fallback,
        },
        path,
    )


def load_components(path: Path, win_probability: WinProbabilityModel) -> FourthDownAdvisor:
    payload = joblib.load(path)
    return FourthDownAdvisor(
        win_probability=win_probability,
        conversion=ConversionModel(payload["conversion"]),
        field_goal=FieldGoalModel(payload["field_goal"]),
        punt=PuntModel(payload["punt_table"], payload["punt_fallback"]),
    )
