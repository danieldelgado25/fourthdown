"""The leakage contract: what a model may learn from, checked before anything trains.

Two layers, because each catches what the other cannot:

* **Declared.** Every feature of every model is in ``schema.PRE_SNAP_COLUMNS`` (known on
  the sideline before the snap) and in none of ``schema.LEAKAGE_COLUMNS`` (nflfastR's own
  fitted outputs). This is what stops ``wp``: as a single feature it predicts the winner
  about as well as the score does (AUC 0.84 against 0.83), so no data test could tell it
  apart, but it is another model's answer, and a model that copies it measures nothing.
* **Measured.** On the warehouse, no single feature separates its model's target almost
  perfectly (AUC above 0.95). This is what stops a post-snap outcome mislabelled as a
  pre-snap column: ``pass_play`` predicts the play call with AUC 1.0. The real features
  top out at 0.83 (``score_differential`` for the winner, ``shotgun`` 0.71 for the call).

`fourthdown train` and the orchestrator's ``trained_models`` asset refuse to train when
the contract is broken; in Dagster it is a blocking check on ``plays``, so the training
step is skipped and the run fails with the violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import polars as pl

from fourthdown.data.schema import LEAKAGE_COLUMNS, PRE_SNAP_COLUMNS
from fourthdown.quality import DataQualityError

PROXY_AUC = 0.95
SAMPLE_ROWS = 200_000
SEED = 0


def model_features() -> dict[str, tuple[tuple[str, ...], str | None]]:
    """Each model's feature list and the target its features are measured against."""
    from fourthdown.models import features
    from fourthdown.models.fourth_down import ConversionModel

    return {
        "win probability": (features.WP_FEATURES, "posteam_won"),
        "play call": (features.PLAYCALL_FEATURES, "is_pass_call"),
        "fourth-down conversion": (ConversionModel.FEATURES, None),
        "conversion features": (features.CONVERSION_FEATURES, None),
    }


@dataclass
class LeakageReport:
    violations: list[str] = field(default_factory=list)
    auc: dict[str, dict[str, float]] = field(default_factory=dict)
    """model -> feature -> single-feature AUC (folded, so 0.5 is no signal)."""

    @property
    def passed(self) -> bool:
        return not self.violations

    def render(self) -> str:
        lines = ["| model | feature | AUC alone |", "| --- | --- | --- |"]
        for model, scores in self.auc.items():
            for name, score in sorted(scores.items(), key=lambda item: -item[1]):
                lines.append(f"| {model} | {name} | {score:.3f} |")
        return "\n".join(lines)


def declared_violations() -> list[str]:
    problems: list[str] = []
    for model, (names, _) in model_features().items():
        leaking = sorted(set(names) & LEAKAGE_COLUMNS)
        if leaking:
            problems.append(
                f"{model}: {', '.join(leaking)} are nflfastR model outputs, never features"
            )
        undeclared = sorted(set(names) - PRE_SNAP_COLUMNS - LEAKAGE_COLUMNS)
        if undeclared:
            problems.append(
                f"{model}: {', '.join(undeclared)} not declared pre-snap "
                "(schema.PRE_SNAP_COLUMNS); add it there only if a coach knows it before the snap"
            )
    return problems


def _auc(target: pl.Series, scores: pl.Series) -> float:
    """Mann-Whitney AUC with tied ranks averaged, folded to >= 0.5."""
    frame = pl.DataFrame({"y": target.cast(pl.Int8), "s": scores.cast(pl.Float64)})
    ranked = frame.with_columns(pl.col("s").rank("average").alias("r"))
    positives = int(ranked["y"].sum())
    negatives = ranked.height - positives
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = float(ranked.filter(pl.col("y") == 1)["r"].sum())
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return max(auc, 1 - auc)


def measured(frame: pl.DataFrame, report: LeakageReport) -> None:
    """Single-feature AUCs against each model's target, on the training frame."""
    rows = {
        "win probability": frame.filter(
            pl.col("posteam_won").is_not_null() & pl.col("down").is_not_null()
        ),
        "play call": frame.filter(
            pl.col("is_designed_play") & pl.col("is_pass_call").is_not_null()
        ),
    }
    for model, (names, target) in model_features().items():
        if target is None or model not in rows:
            continue
        usable = rows[model]
        if usable.height > SAMPLE_ROWS:
            usable = usable.sample(n=SAMPLE_ROWS, seed=SEED)
        scores: dict[str, float] = {}
        for name in names:
            if name not in usable.columns:
                report.violations.append(f"{model}: feature {name} is not in the warehouse")
                continue
            pair = usable.select(name, target).drop_nulls()
            if pair.is_empty():
                report.violations.append(f"{model}: feature {name} is null on every row")
                continue
            scores[name] = round(_auc(pair[target], pair[name]), 3)
            if scores[name] > PROXY_AUC:
                report.violations.append(
                    f"{model}: {name} alone predicts {target} with AUC {scores[name]:.3f} "
                    f"(> {PROXY_AUC}); it carries the outcome"
                )
        report.auc[model] = scores


def check(connection: duckdb.DuckDBPyConnection) -> LeakageReport:
    """The whole contract against the warehouse."""
    from fourthdown.models import features

    report = LeakageReport(violations=declared_violations())
    measured(features.load_plays(connection), report)
    return report


def enforce(connection: duckdb.DuckDBPyConnection) -> LeakageReport:
    """``check``, raising ``DataQualityError`` when the contract is broken."""
    report = check(connection)
    if not report.passed:
        raise DataQualityError("the leakage contract", report.violations)
    return report
