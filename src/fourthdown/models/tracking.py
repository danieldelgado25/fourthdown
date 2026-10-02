"""Experiment tracking with MLflow, and the model card that travels with an artifact.

Every `fourthdown train` run becomes an MLflow run holding the data version it read,
the season split and hyperparameters it used, the headline metrics, the per-epoch
loss curve, and the artifacts it wrote. Tracking defaults to a SQLite store under the
data directory, so it works offline; point ``FOURTHDOWN_MLFLOW_URI`` at a server to
share runs.

The model card is the same provenance in a file that ships next to ``winprob.pt``. The
serving container has no MLflow and no warehouse, so the card is how it knows which run
and which data produced the weights it loaded, and its SHA-256 is how it knows the
weights are the ones the card describes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from fourthdown.config import Paths
from fourthdown.data import provenance
from fourthdown.models import features, playcall, winprob
from fourthdown.models.card import ModelCard, ParamValue
from fourthdown.models.features import SeasonSplit

EXPERIMENT = "fourthdown-models"
URI_ENV = "FOURTHDOWN_MLFLOW_URI"


def tracking_uri(paths: Paths) -> str:
    override = os.environ.get(URI_ENV)
    if override:
        return override
    return f"sqlite:///{paths.mlflow / 'mlflow.db'}"


def _seasons(values: Sequence[int]) -> str:
    ordered = sorted(values)
    return f"{ordered[0]}-{ordered[-1]}" if len(ordered) > 1 else str(ordered[0])


def training_params(split: SeasonSplit, *, epochs: int, audit_sample: int) -> dict[str, ParamValue]:
    """Everything that decides what a training run produces, apart from the data."""
    return {
        "split.train": _seasons(split.train),
        "split.valid": _seasons(split.valid),
        "split.test": _seasons(split.test),
        "wp.features": ",".join(features.WP_FEATURES),
        "wp.hidden": ",".join(str(width) for width in winprob.HIDDEN),
        "wp.dropout": winprob.DROPOUT,
        "wp.batch_size": winprob.BATCH_SIZE,
        "wp.max_epochs": epochs,
        "wp.patience": winprob.PATIENCE,
        "wp.learning_rate": winprob.LEARNING_RATE,
        "wp.weight_decay": winprob.WEIGHT_DECAY,
        "wp.seed": winprob.SEED,
        "playcall.max_iter": playcall.MAX_ITER,
        "playcall.learning_rate": playcall.LEARNING_RATE,
        "playcall.max_depth": playcall.MAX_DEPTH,
        "playcall.seed": playcall.SEED,
        "advisor.audit_sample": audit_sample,
    }


def _experiment_id(client: MlflowClient, paths: Paths) -> str:
    existing = client.get_experiment_by_name(EXPERIMENT)
    if existing is not None:
        return str(existing.experiment_id)
    artifacts = (paths.mlflow / "artifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    return str(client.create_experiment(EXPERIMENT, artifact_location=artifacts.as_uri()))


def log_run(
    card: ModelCard,
    *,
    paths: Paths,
    card_path: Path,
    manifest: provenance.Manifest,
    history: winprob.TrainingHistory,
    artifacts: Sequence[Path],
) -> ModelCard:
    """Record a training run in MLflow and write the card, stamped with the run id."""
    uri = tracking_uri(paths)
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=uri)
    tags = {
        "data_version": card.data_version,
        "seasons": _seasons(manifest.seasons) if manifest.seasons else "none",
        "git_commit": card.git_commit or "unknown",
        "package_version": card.package_version,
        "artifact_sha256": card.artifact_sha256,
    }
    run = client.create_run(
        _experiment_id(client, paths),
        run_name=f"train-{card.data_version}",
        tags=tags,
    )
    run_id = run.info.run_id
    try:
        stamped = replace(card, mlflow_run_id=run_id, mlflow_tracking_uri=uri)
        stamped.save(card_path)
        now = int(time.time() * 1000)
        metrics = [Metric(key, float(value), now, 0) for key, value in card.metrics.items()]
        for step, (train_loss, valid_loss) in enumerate(
            zip(history.train_loss, history.valid_loss, strict=True)
        ):
            metrics.append(Metric("wp_train_loss", train_loss, now, step))
            metrics.append(Metric("wp_valid_loss", valid_loss, now, step))
        params = [Param(key, str(value)) for key, value in card.params.items()]
        params.append(Param("data_version", card.data_version))
        client.log_batch(run_id, metrics=metrics, params=params, tags=[RunTag("stage", "train")])
        for path in (*artifacts, card_path, paths.manifest):
            if path.exists():
                client.log_artifact(run_id, str(path))
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id)
    return stamped
