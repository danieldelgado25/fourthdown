"""Train every model into a candidate directory, register it, and serve it if it earns it.

`fourthdown train` and the orchestrator's ``trained_models`` asset both run this. Models
are trained into ``models/.candidate`` so nothing is served until it is judged. With
tracking on, the run becomes a new registered version (``fourthdown.models.registry``) and
is promoted into the served directory per ``promote``:

* ``gated``: only if the registry's gates pass (what the weekly retrain does);
* ``never``: register it, keep serving what serves now;
* ``always``: serve it regardless (CI smoke runs on tiny data).

With tracking off, the candidate is copied in directly (when ``promote`` allows).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from fourthdown.config import Paths
from fourthdown.data import provenance, warehouse
from fourthdown.models import registry, tracking
from fourthdown.models import train as training
from fourthdown.models.card import MODEL_CARD, ModelCard
from fourthdown.models.features import DEFAULT_SPLIT, SeasonSplit
from fourthdown.models.train import FOURTH_DOWN_ARTIFACT, WP_ARTIFACT
from fourthdown.quality import leakage

PROMOTE_MODES = ("gated", "never", "always")
CANDIDATE_DIR = ".candidate"
"""Where models are trained before anything is served; under the models dir, so on one volume."""
LAST_TRAINED = "last_trained_card.json"
"""The card of the latest run, served or not, under ``paths.reports``."""


class DataDriftError(RuntimeError):
    """The files on disk do not match the provenance manifest; training would be untraceable."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(
            "data does not match data/manifest.json; rerun `fourthdown build`:\n  "
            + "\n  ".join(problems)
        )
        self.problems = problems


@dataclass
class Release:
    """What one training run produced and what became of it."""

    card: ModelCard
    results: training.Results
    report: Path
    serving: Served


def is_current(paths: Paths, split: SeasonSplit, *, epochs: int, audit_sample: int) -> bool:
    """True when the last training run used exactly the data on disk and these settings.

    Checked against the last trained card as well as the served one: a candidate that
    failed its gates was still trained on this data, and retraining would fail the same way.
    """
    if provenance.verify(paths):
        return False
    manifest = provenance.load(paths)
    present = set(manifest.seasons)
    try:
        # A card records the seasons each split actually had rows for.
        effective = SeasonSplit(
            train=tuple(s for s in split.train if s in present),
            valid=tuple(s for s in split.valid if s in present),
            test=tuple(s for s in split.test if s in present),
        )
    except ValueError:
        return False
    params = tracking.training_params(effective, epochs=epochs, audit_sample=audit_sample)
    for card_path in (paths.reports / LAST_TRAINED, paths.models / MODEL_CARD):
        if card_path.exists():
            card = ModelCard.load(card_path)
            if card.data_version == manifest.data_version and card.params == params:
                return True
    return False


def train_and_release(
    paths: Paths,
    *,
    report: Path,
    split: SeasonSplit = DEFAULT_SPLIT,
    epochs: int = 60,
    audit_sample: int = 4000,
    track: bool = True,
    promote: str = "gated",
) -> Release:
    """Train, write the evaluation report, register, and serve per ``promote``.

    Raises:
        DataDriftError: the data on disk is not what the manifest records.
        DataQualityError: the leakage contract is broken (``fourthdown.quality.leakage``);
            nothing is trained.
        ValueError: ``promote`` is not one of ``PROMOTE_MODES``.
    """
    if promote not in PROMOTE_MODES:
        raise ValueError(f"promote must be one of {', '.join(PROMOTE_MODES)}")
    problems = provenance.verify(paths)
    if problems:
        raise DataDriftError(problems)
    manifest = provenance.load(paths)
    candidate = paths.models / CANDIDATE_DIR
    shutil.rmtree(candidate, ignore_errors=True)
    candidate.mkdir(parents=True)
    try:
        with warehouse.connect(paths) as connection:
            leakage.enforce(connection)
            results = training.run(
                connection,
                model_dir=candidate,
                max_epochs=epochs,
                audit_sample=audit_sample,
                split=split,
            )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(training.report(results))
        card = ModelCard.build(
            artifact=candidate / WP_ARTIFACT,
            manifest=manifest,
            params=tracking.training_params(
                SeasonSplit(
                    train=results.wp_dataset.train.seasons,
                    valid=results.wp_dataset.valid.seasons,
                    test=results.wp_dataset.test.seasons,
                ),
                epochs=epochs,
                audit_sample=audit_sample,
            ),
            metrics=training.summary_metrics(results),
        )
        card_path = candidate / MODEL_CARD
        if track:
            card = tracking.log_run(
                card,
                paths=paths,
                card_path=card_path,
                manifest=manifest,
                history=results.history,
                artifacts=[
                    candidate / name
                    for name in (
                        WP_ARTIFACT,
                        training.HISTORY_ARTIFACT,
                        training.PLAYCALL_ARTIFACT,
                        FOURTH_DOWN_ARTIFACT,
                    )
                ]
                + [report],
            )
        else:
            card.save(card_path)
        paths.reports.mkdir(parents=True, exist_ok=True)
        card.save(paths.reports / LAST_TRAINED)
        served = serve_candidate(card, paths, candidate, promote=promote, track=track)
    finally:
        shutil.rmtree(candidate, ignore_errors=True)
    return Release(card=card, results=results, report=report, serving=served)


@dataclass
class Served:
    """What became of a candidate: its registered version, and whether it now serves."""

    gate_failures: list[str]
    served: bool = False
    version: int | None = None
    notes: list[str] = field(default_factory=list)


def serve_candidate(
    card: ModelCard, paths: Paths, candidate: Path, *, promote: str, track: bool
) -> Served:
    """Register the candidate, and install it for serving if ``promote`` allows."""
    outcome = Served(gate_failures=registry.gate_failures(card.metrics))
    failures = outcome.gate_failures
    serve = promote == "always" or (promote == "gated" and not failures)
    if track:
        version = registry.register(card, paths)
        outcome.version = version.version
        outcome.notes.append(
            f"registered {registry.MODEL_NAME} version {version.version} ({version.gates})"
        )
        if serve:
            registry.promote(paths, version.version, ungated=bool(failures))
            role = "serving it, not as champion" if failures else f"now {registry.ALIAS}"
            outcome.notes.append(f"installed version {version.version}; {role}")
    elif serve:
        registry.install_unregistered(candidate, paths.models)
        outcome.notes.append(f"installed the untracked candidate into {paths.models}")
    outcome.served = serve
    if not serve and promote != "never":
        outcome.notes.append(
            f"not promoted: the models gates failed; still serving what is in {paths.models}"
        )
    return outcome
