"""Scoring for probability models.

Accuracy is the wrong headline for a win-probability model: a model that says 50% for
every play and a model that is confidently right half the time are both "50% accurate",
and only one of them is useful. These are the numbers that separate them -- proper
scoring rules (log loss, Brier) and calibration, which asks whether the plays labelled
70% actually won 70% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPSILON = 1e-7


@dataclass(frozen=True)
class Bin:
    """One reliability-diagram bucket."""

    lower: float
    upper: float
    count: int
    predicted: float
    observed: float


@dataclass(frozen=True)
class ProbabilityScore:
    """How good a set of probabilities is, from three angles."""

    rows: int
    log_loss: float
    brier: float
    auc: float
    accuracy: float
    calibration_error: float
    bins: tuple[Bin, ...]

    def as_row(self, name: str) -> str:
        return (
            f"| {name} | {self.rows:,} | {self.log_loss:.4f} | {self.brier:.4f} | "
            f"{self.auc:.4f} | {self.accuracy:.3f} | {self.calibration_error:.4f} |"
        )


HEADER = (
    "| model | rows | log loss | Brier | AUC | accuracy | ECE |\n"
    "| --- | --- | --- | --- | --- | --- | --- |"
)


def reliability(
    truth: np.ndarray, probability: np.ndarray, *, bins: int = 10
) -> tuple[tuple[Bin, ...], float]:
    """Bucket predictions and report how far each bucket drifts from its own claim.

    The scalar is expected calibration error: the count-weighted mean gap between what
    was predicted and what happened.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    buckets: list[Bin] = []
    error = 0.0
    for position in range(bins):
        mask = index == position
        count = int(mask.sum())
        if not count:
            continue
        predicted = float(probability[mask].mean())
        observed = float(truth[mask].mean())
        error += count * abs(predicted - observed)
        buckets.append(
            Bin(
                lower=float(edges[position]),
                upper=float(edges[position + 1]),
                count=count,
                predicted=predicted,
                observed=observed,
            )
        )
    total = int(probability.size)
    return tuple(buckets), error / total if total else 0.0


def score(truth: np.ndarray, probability: np.ndarray, *, bins: int = 10) -> ProbabilityScore:
    """Log loss, Brier, AUC, accuracy at 0.5, and calibration error in one pass."""
    truth = np.asarray(truth).astype(np.float64).ravel()
    probability = np.clip(np.asarray(probability).astype(np.float64).ravel(), EPSILON, 1 - EPSILON)
    if truth.shape != probability.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {probability.shape}")
    buckets, calibration_error = reliability(truth, probability, bins=bins)
    single_class = len(np.unique(truth)) < 2
    return ProbabilityScore(
        rows=int(truth.size),
        log_loss=float(log_loss(truth, probability, labels=[0, 1])),
        brier=float(brier_score_loss(truth, probability)),
        auc=float("nan") if single_class else float(roc_auc_score(truth, probability)),
        accuracy=float(((probability >= 0.5) == (truth >= 0.5)).mean()),
        calibration_error=float(calibration_error),
        bins=buckets,
    )


def reliability_table(buckets: tuple[Bin, ...]) -> str:
    lines = ["| bucket | plays | predicted | observed |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {bucket.lower:.1f}-{bucket.upper:.1f} | {bucket.count:,} | "
        f"{bucket.predicted:.3f} | {bucket.observed:.3f} |"
        for bucket in buckets
    ]
    return "\n".join(lines)
