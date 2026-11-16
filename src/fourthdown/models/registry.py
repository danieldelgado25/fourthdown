"""The MLflow model registry: every trained model is a version; only gated ones serve.

`fourthdown train` used to write straight into ``data/models``, the directory the models
service and the win-probability container load, so whatever the last run produced was
served whether or not it was any good. Now:

1. Training writes into a candidate directory and logs the run to MLflow (`tracking`).
2. `register` makes the run a new version of `MODEL_NAME`, tagged with the verdict of the
   ``models.*`` scorecard gates (`gates.json`, the same bounds CI enforces) on its metrics.
3. `promote` refuses a version whose gates failed. Otherwise it points the `ALIAS` alias
   at it and exports that version's artifacts from MLflow into ``data/models``, with the
   card stamped with the registered version.

The serving side is unchanged and needs no MLflow: it loads ``data/models``, which only
`promote` writes. The card's ``registered_version`` says which version that is.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fourthdown.config import Paths
from fourthdown.data import provenance
from fourthdown.evaluation import scorecard
from fourthdown.models import tracking
from fourthdown.models.card import MODEL_CARD, ModelCard
from fourthdown.models.train import (
    FOURTH_DOWN_ARTIFACT,
    HISTORY_ARTIFACT,
    PLAYCALL_ARTIFACT,
    WP_ARTIFACT,
)

MODEL_NAME = "fourthdown-win-probability"
ALIAS = "champion"
GATES_TAG = "gates"
"""``pass`` or ``fail``: the models-suite verdict recorded on each version."""

SERVED_FILES: tuple[str, ...] = (
    WP_ARTIFACT,
    HISTORY_ARTIFACT,
    PLAYCALL_ARTIFACT,
    FOURTH_DOWN_ARTIFACT,
    MODEL_CARD,
)
"""What `promote` exports into the served directory: the files the services load."""


class RegistryError(RuntimeError):
    """A version that does not exist, failed its gates, or lacks its artifacts."""


def gate_failures(metrics: Mapping[str, float]) -> list[str]:
    """Every ``models.*`` gate the metrics miss, as ``key=value (bound)``; empty if all pass.

    A gated metric the run did not produce is a failure, as in the scorecard: a renamed
    metric must not quietly pass a gate.
    """
    failures = []
    for gate in scorecard.load_gates():
        if gate.suite != "models":
            continue
        value = metrics.get(gate.metric)
        if value is None:
            failures.append(f"{gate.key} missing")
        elif not gate.admits(float(value)):
            failures.append(f"{gate.key}={float(value):.4g} (needs {gate.bound()})")
    return failures


def client_for(paths: Paths) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking.tracking_uri(paths))


@dataclass(frozen=True)
class Version:
    """One registered model version, as `versions` lists it."""

    version: int
    run_id: str
    gates: str
    champion: bool
    data_version: str | None
    metrics: dict[str, float]
    failures: str


def register(card: ModelCard, paths: Paths) -> Version:
    """Register ``card``'s MLflow run as a new version, tagged with its gate verdict."""
    if card.mlflow_run_id is None:
        raise RegistryError("the card has no MLflow run; train with tracking on to register")
    client = client_for(paths)
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(
            MODEL_NAME, description="Win probability, the fourth-down advisor, and play call."
        )
    failures = gate_failures(card.metrics)
    run = client.get_run(card.mlflow_run_id)
    created = client.create_model_version(
        MODEL_NAME,
        source=run.info.artifact_uri,
        run_id=card.mlflow_run_id,
        tags={
            GATES_TAG: "fail" if failures else "pass",
            "gate_failures": "; ".join(failures),
            "data_version": card.data_version,
            "artifact_sha256": card.artifact_sha256,
        },
    )
    return _version(client, created.version, champion=None)


def _champion(client: MlflowClient) -> int | None:
    try:
        return int(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)
    except MlflowException:
        return None


def _version(client: MlflowClient, number: str | int, champion: int | None) -> Version:
    found = client.get_model_version(MODEL_NAME, str(number))
    if found.run_id is None:
        raise RegistryError(f"version {number} was not registered from a run")
    run = client.get_run(found.run_id)
    tags = found.tags or {}
    return Version(
        version=int(found.version),
        run_id=found.run_id,
        gates=tags.get(GATES_TAG, "unknown"),
        champion=champion is not None and int(found.version) == champion,
        data_version=tags.get("data_version"),
        metrics=dict(run.data.metrics),
        failures=tags.get("gate_failures", ""),
    )


def versions(paths: Paths) -> list[Version]:
    """Every registered version, newest first; empty when nothing is registered."""
    client = client_for(paths)
    try:
        found = client.search_model_versions(f"name = '{MODEL_NAME}'")
    except MlflowException:
        return []
    champion = _champion(client)
    numbers = sorted((int(version.version) for version in found), reverse=True)
    return [_version(client, number, champion) for number in numbers]


def promote(paths: Paths, version: int, *, ungated: bool = False) -> ModelCard:
    """Make ``version`` the champion and install its artifacts where the services load them.

    Raises `RegistryError` if the version does not exist, failed its gates, or its
    artifacts do not match its card. ``ungated`` (``train --promote always``, for CI smoke
    runs) installs a version that failed its gates, but never makes it the champion.
    """
    client = client_for(paths)
    try:
        found = _version(client, version, _champion(client))
    except MlflowException as error:
        raise RegistryError(f"{MODEL_NAME} has no version {version}") from error
    passed = found.gates == "pass"
    if not passed and not ungated:
        detail = f": {found.failures}" if found.failures else ""
        raise RegistryError(f"version {version} did not pass the models gates{detail}")
    card = _install(client, found, paths.models)
    if passed:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, str(version))
    return card


def _install(client: MlflowClient, version: Version, target: Path) -> ModelCard:
    """Download the version's files beside ``target``, verify them, then swap them in.

    Each file is replaced with `os.replace`, so a reader never sees a half-written one, and
    the card goes last: the services key their reload on it.
    """
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".promote-") as scratch:
        staging = Path(scratch)
        for name in SERVED_FILES:
            try:
                client.download_artifacts(version.run_id, name, str(staging))
            except (MlflowException, OSError) as error:
                raise RegistryError(f"version {version.version} has no {name}") from error
        card = ModelCard.load(staging / MODEL_CARD)
        digest = provenance.sha256_file(staging / card.artifact)
        if digest != card.artifact_sha256:
            raise RegistryError(
                f"version {version.version}'s {card.artifact} does not match its card"
            )
        card = replace(card, registered_model=MODEL_NAME, registered_version=version.version)
        card.save(staging / MODEL_CARD)
        for name in (*(name for name in SERVED_FILES if name != MODEL_CARD), MODEL_CARD):
            os.replace(staging / name, target / name)
    return card


def install_unregistered(source: Path, target: Path, names: Sequence[str] = SERVED_FILES) -> None:
    """Copy a candidate's files into the served directory without the registry.

    For ``fourthdown train --no-track``, which has no MLflow run to register.
    """
    target.mkdir(parents=True, exist_ok=True)
    for name in (*(name for name in names if name != MODEL_CARD), MODEL_CARD):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
