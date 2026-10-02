"""Fit every phase-04 model and write the report that justifies them.

One entry point, because the three models share a feature frame and a split, and
because a results table is only honest if every row in it was produced by the same run
against the same held-out seasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from fourthdown.models import features, fourth_down, metrics, playcall, winprob
from fourthdown.models.features import DEFAULT_SPLIT, Dataset, SeasonSplit
from fourthdown.models.metrics import ProbabilityScore
from fourthdown.models.winprob import GameState

LOGGER = logging.getLogger(__name__)

WP_ARTIFACT = "winprob.pt"
PLAYCALL_ARTIFACT = "playcall.joblib"
FOURTH_DOWN_ARTIFACT = "fourth_down.joblib"
HISTORY_ARTIFACT = "winprob_history.json"

FOURTH_DOWN_SAMPLE = 4000
SAMPLE_SEED = 11


@dataclass(frozen=True)
class DecisionAudit:
    """How often coaches chose what the advisor would, on held-out fourth downs."""

    plays: int
    agreement: float
    coach_go_rate: float
    model_go_rate: float
    lost_win_probability: float

    def render(self) -> str:
        return "\n".join(
            [
                f"- fourth downs audited: {self.plays:,}",
                f"- coach and advisor agree: {self.agreement:.1%}",
                f"- coaches went for it: {self.coach_go_rate:.1%}; "
                f"advisor would: {self.model_go_rate:.1%}",
                "- mean win probability surrendered by the actual choice: "
                f"{self.lost_win_probability * 100:.2f} points",
            ]
        )


@dataclass(frozen=True)
class Results:
    """Everything one training run produced."""

    win_probability: ProbabilityScore
    vegas_baseline: ProbabilityScore | None
    nflfastr_baseline: ProbabilityScore | None
    history: winprob.TrainingHistory
    play_call: ProbabilityScore
    play_call_majority: ProbabilityScore
    play_call_xpass: ProbabilityScore | None
    predictability: playcall.PredictabilityReport
    audit: DecisionAudit
    wp_dataset: Dataset
    playcall_dataset: Dataset


def win_probability_dataset(frame: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> Dataset:
    """Every play with a decided game, from the offense's perspective."""
    usable = frame.filter(pl.col("posteam_won").is_not_null() & pl.col("down").is_not_null())
    return features.split_by_season(
        usable,
        features.WP_FEATURES,
        "posteam_won",
        train=split.train,
        valid=split.valid,
        test=split.test,
    )


def play_call_dataset(frame: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> Dataset:
    """Designed run/pass plays only: kneels and punts are not a coordinator's choice."""
    usable = frame.filter(pl.col("is_designed_play") & pl.col("is_pass_call").is_not_null())
    return features.split_by_season(
        usable,
        features.PLAYCALL_FEATURES,
        "is_pass_call",
        train=split.train,
        valid=split.valid,
        test=split.test,
    )


def _baseline(frame: pl.DataFrame, column: str, target: str) -> ProbabilityScore | None:
    available = frame.drop_nulls(subset=[column])
    if available.is_empty():
        return None
    return metrics.score(
        available[target].to_numpy().astype(float), available[column].to_numpy().astype(float)
    )


def _states(frame: pl.DataFrame) -> list[GameState]:
    return [
        GameState(
            yardline_100=float(row["yardline_100"]),
            down=int(row["down"]),
            ydstogo=float(row["ydstogo"]),
            game_seconds_remaining=float(row["game_seconds_remaining"]),
            score_differential=float(row["score_differential"]),
            posteam_timeouts_remaining=int(row["posteam_timeouts_remaining"]),
            defteam_timeouts_remaining=int(row["defteam_timeouts_remaining"]),
            posteam_is_home=bool(row["posteam_is_home"]),
            posteam_spread=float(row["posteam_spread"]),
            goal_to_go=bool(row["goal_to_go"]),
            half_seconds_remaining=float(row["half_seconds_remaining"]),
        )
        for row in frame.iter_rows(named=True)
    ]


def audit_decisions(
    advisor: fourth_down.FourthDownAdvisor,
    frame: pl.DataFrame,
    *,
    sample: int = FOURTH_DOWN_SAMPLE,
    seed: int = SAMPLE_SEED,
) -> DecisionAudit:
    """Replay held-out fourth downs and compare the advisor with what actually happened.

    A play counts as "went for it" when the offense ran a designed play; a field-goal or
    punt attempt counts as the kick. Sampling keeps the audit to a few thousand states
    because each one costs three network evaluations.
    """
    fourth = frame.filter(
        (pl.col("down") == 4)
        & pl.col("yardline_100").is_not_null()
        & (pl.col("is_designed_play") | pl.col("field_goal_attempt") | pl.col("punt_attempt"))
    )
    if fourth.is_empty():
        return DecisionAudit(0, float("nan"), float("nan"), float("nan"), float("nan"))
    if fourth.height > sample:
        fourth = fourth.sample(n=sample, seed=seed)
    choices = (
        pl.when(pl.col("is_designed_play"))
        .then(pl.lit("go"))
        .when(pl.col("field_goal_attempt"))
        .then(pl.lit("field_goal"))
        .otherwise(pl.lit("punt"))
    )
    actual = fourth.select(choices.alias("choice"))["choice"].to_list()
    agreed = 0
    surrendered: list[float] = []
    model_go = 0
    for state, chosen in zip(_states(fourth), actual, strict=True):
        recommendation = advisor.recommend(state)
        best = recommendation.best
        if best.name == "go":
            model_go += 1
        if best.name == chosen:
            agreed += 1
        taken = next(option for option in recommendation.options if option.name == chosen)
        surrendered.append(best.win_probability - taken.win_probability)
    plays = len(actual)
    return DecisionAudit(
        plays=plays,
        agreement=agreed / plays,
        coach_go_rate=actual.count("go") / plays,
        model_go_rate=model_go / plays,
        lost_win_probability=float(np.mean(surrendered)),
    )


def run(
    connection: duckdb.DuckDBPyConnection,
    *,
    model_dir: Path,
    max_epochs: int = winprob.MAX_EPOCHS,
    audit_sample: int = FOURTH_DOWN_SAMPLE,
    split: SeasonSplit = DEFAULT_SPLIT,
) -> Results:
    """Train win probability, the fourth-down components, and the play-call model."""
    frame = features.load_plays(connection)
    wp_data = win_probability_dataset(frame, split)
    LOGGER.info("win probability splits: %s", wp_data.summary())
    model, history = winprob.train(wp_data, max_epochs=max_epochs)

    test_frame = wp_data.test.frame
    predicted = model.predict(features.feature_matrix(test_frame, wp_data.features))
    wp_score = metrics.score(features.target_vector(test_frame, wp_data.target), predicted)

    call_data = play_call_dataset(frame, split)
    LOGGER.info("play-call splits: %s", call_data.summary())
    call_model = playcall.PlayCallModel.fit(call_data)
    call_score = metrics.score(
        features.target_vector(call_data.test.frame, call_data.target),
        call_model.predict(call_data.test.frame),
    )
    neutral = call_data.test.frame.filter(pl.col("is_neutral_script") & (pl.col("down") <= 2))
    tendencies = playcall.predictability(call_model, neutral)

    holdout = wp_data.test.seasons[0] if len(wp_data.test) else min(split.test)
    advisor = fourth_down.build(connection, model, holdout_season=holdout)
    audit = audit_decisions(advisor, test_frame, sample=audit_sample)

    model.save(model_dir / WP_ARTIFACT)
    winprob.write_history(history, model_dir / HISTORY_ARTIFACT)
    call_model.save(model_dir / PLAYCALL_ARTIFACT)
    fourth_down.save_components(advisor, model_dir / FOURTH_DOWN_ARTIFACT)
    LOGGER.info("artifacts written to %s", model_dir)

    return Results(
        win_probability=wp_score,
        vegas_baseline=_baseline(test_frame, "vegas_wp", wp_data.target),
        nflfastr_baseline=_baseline(test_frame, "wp", wp_data.target),
        history=history,
        play_call=call_score,
        play_call_majority=playcall.majority_baseline(call_data),
        play_call_xpass=playcall.xpass_baseline(call_data),
        predictability=tendencies,
        audit=audit,
        wp_dataset=wp_data,
        playcall_dataset=call_data,
    )


def summary_metrics(results: Results) -> dict[str, float]:
    """The headline numbers of a run: what the scorecard gates and MLflow records."""
    wp, call = results.win_probability, results.play_call
    metrics = {
        "wp_log_loss": wp.log_loss,
        "wp_brier": wp.brier,
        "wp_auc": wp.auc,
        "wp_ece": wp.calibration_error,
        "playcall_accuracy": call.accuracy,
        "playcall_auc": call.auc,
        "playcall_lift_over_majority": call.accuracy - results.play_call_majority.accuracy,
        "advisor_agreement": results.audit.agreement,
        "advisor_go_rate": results.audit.model_go_rate,
        "coach_go_rate": results.audit.coach_go_rate,
    }
    if results.vegas_baseline is not None:
        metrics["wp_log_loss_gap_vs_vegas"] = wp.log_loss - results.vegas_baseline.log_loss
    if results.play_call_xpass is not None:
        metrics["playcall_lift_over_xpass"] = call.accuracy - results.play_call_xpass.accuracy
    return metrics


def report(results: Results) -> str:
    """The Markdown that lands in ``docs/model_eval.md``."""
    wp = results.wp_dataset
    calls = results.playcall_dataset
    lines = [
        "# Model evaluation",
        "",
        "Every number below is measured on held-out seasons "
        f"({wp.test.seasons[0]}-{wp.test.seasons[-1]}); models were fit on "
        f"{wp.train.seasons[0]}-{wp.train.seasons[-1]} and early-stopped on "
        f"{wp.valid.seasons[0]}-{wp.valid.seasons[-1]}.",
        "",
        "## Win probability",
        "",
        f"Splits: {wp.summary()}.",
        f"Best epoch {results.history.best_epoch + 1} of {len(results.history.valid_loss)} "
        f"(validation log loss {results.history.best_valid_loss:.4f}).",
        "",
        metrics.HEADER,
        results.win_probability.as_row("fourthdown MLP"),
    ]
    if results.vegas_baseline is not None:
        lines.append(results.vegas_baseline.as_row("nflfastR vegas_wp"))
    if results.nflfastr_baseline is not None:
        lines.append(results.nflfastr_baseline.as_row("nflfastR wp"))
    lines += [
        "",
        "The baselines are nflfastR's own fitted win-probability columns, scored on the "
        "same plays. They are never features -- `features.feature_matrix` raises if one "
        "reaches the design matrix -- so this is a comparison, not a leak.",
        "",
        "### Calibration",
        "",
        metrics.reliability_table(results.win_probability.bins),
        "",
        "## Fourth-down advisor",
        "",
        results.audit.render(),
        "",
        "The advisor has no parameters of its own: it evaluates the win-probability "
        "model at the states each option leads to, weighted by a conversion model, a "
        "field-goal model, and the empirical punt landing spot.",
        "",
        "## Play call (run or pass)",
        "",
        f"Splits: {calls.summary()}.",
        "",
        metrics.HEADER,
        results.play_call.as_row("fourthdown GBM"),
        results.play_call_majority.as_row("base rate"),
    ]
    if results.play_call_xpass is not None:
        lines.append(results.play_call_xpass.as_row("nflfastR xpass"))
    report_rows = results.predictability
    lines += [
        "",
        "### Predictability",
        "",
        "How often the model's call was the call, on early-down neutral-script plays, "
        "by team-season. 0.5 is a coin flip; `confidence` is how sure the model was.",
        "",
        "Most predictable:",
        "",
        playcall.tendency_table(report_rows.most_predictable()),
        "",
        "Least predictable:",
        "",
        playcall.tendency_table(report_rows.least_predictable()),
        "",
        f"Correlation between predictability and EPA per play: "
        f"{report_rows.correlation:+.3f} across {len(report_rows.rows)} team-seasons.",
        "",
    ]
    return "\n".join(lines)
