"""The model registry: register every run, serve only versions that pass the models gates."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from fourthdown.cli import _serve_candidate, app
from fourthdown.data import provenance
from fourthdown.models import features, registry, tracking
from fourthdown.models.card import MODEL_CARD, ModelCard
from fourthdown.models.features import SeasonSplit
from fourthdown.models.train import (
    FOURTH_DOWN_ARTIFACT,
    HISTORY_ARTIFACT,
    PLAYCALL_ARTIFACT,
    WP_ARTIFACT,
)
from fourthdown.models.winprob import TrainingHistory, WinProbabilityModel, WinProbabilityNet
from fourthdown.serving import winprob as service

SPLIT = SeasonSplit(train=(2009, 2010), valid=(2011,), test=(2012,))

PASSING = {
    "wp_auc": 0.86,
    "wp_log_loss": 0.455,
    "wp_ece": 0.01,
    "wp_log_loss_gap_vs_vegas": 0.005,
    "playcall_auc": 0.815,
    "playcall_lift_over_majority": 0.12,
    "advisor_agreement": 0.67,
    "advisor_go_rate": 0.42,
}
LEAKY = {**PASSING, "wp_auc": 0.97}  # too good: a leaked feature


@pytest.fixture()
def manifest():
    return provenance.Manifest(
        processed={
            2009: provenance.PartitionRecord(2009, "p", "a" * 64, 10, "b" * 64, "c" * 64, "t")
        }
    )


def train_candidate(directory: Path, manifest, metrics: dict[str, float], seed: int = 0):
    """What `fourthdown train` leaves in its candidate directory, with chosen metrics."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    net = WinProbabilityNet(len(features.WP_FEATURES))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(1e-3)
    WinProbabilityModel(features.WP_FEATURES, net).save(directory / WP_ARTIFACT)
    for name in (HISTORY_ARTIFACT, PLAYCALL_ARTIFACT, FOURTH_DOWN_ARTIFACT):
        (directory / name).write_text(f"{name} from seed {seed}")
    return ModelCard.build(
        artifact=directory / WP_ARTIFACT,
        manifest=manifest,
        params=tracking.training_params(SPLIT, epochs=3, audit_sample=10),
        metrics=metrics,
    )


def tracked(paths, manifest, metrics, seed=0) -> ModelCard:
    candidate = paths.models / f".candidate-{seed}"
    card = train_candidate(candidate, manifest, metrics, seed)
    return tracking.log_run(
        card,
        paths=paths,
        card_path=candidate / MODEL_CARD,
        manifest=manifest,
        history=TrainingHistory(train_loss=[0.6], valid_loss=[0.62], best_epoch=0),
        artifacts=[
            candidate / name
            for name in (WP_ARTIFACT, HISTORY_ARTIFACT, PLAYCALL_ARTIFACT, FOURTH_DOWN_ARTIFACT)
        ],
    )


# --- gates -----------------------------------------------------------------------------


def test_metrics_inside_every_models_gate_pass():
    assert registry.gate_failures(PASSING) == []


def test_a_leaky_auc_and_a_missing_metric_both_fail():
    failures = registry.gate_failures({key: v for key, v in LEAKY.items() if key != "wp_ece"})
    assert any(failure.startswith("models.wp_auc=0.97") for failure in failures)
    assert "models.wp_ece missing" in failures


# --- register and promote ---------------------------------------------------------------


def test_register_records_the_gate_verdict(tmp_paths, manifest):
    good = registry.register(tracked(tmp_paths, manifest, PASSING, seed=1), tmp_paths)
    bad = registry.register(tracked(tmp_paths, manifest, LEAKY, seed=2), tmp_paths)
    assert (good.version, good.gates) == (1, "pass")
    assert (bad.version, bad.gates) == (2, "fail")
    assert "models.wp_auc" in bad.failures
    listed = registry.versions(tmp_paths)
    assert [version.version for version in listed] == [2, 1]
    assert not any(version.champion for version in listed)


def test_promote_installs_the_version_where_the_services_load(tmp_paths, manifest):
    card = tracked(tmp_paths, manifest, PASSING)
    version = registry.register(card, tmp_paths)
    served = registry.promote(tmp_paths, version.version)
    assert served.registered_version == 1
    assert served.registered_model == registry.MODEL_NAME
    for name in registry.SERVED_FILES:
        assert (tmp_paths.models / name).exists()
    # The model container verifies the weights against the card and reports the version.
    loaded = service.load(tmp_paths.models)
    assert loaded.provenance()["registered_version"] == 1
    assert loaded.provenance()["mlflow_run_id"] == card.mlflow_run_id
    assert [version.champion for version in registry.versions(tmp_paths)] == [True]


def test_a_failed_version_is_never_promoted(tmp_paths, manifest):
    registry.register(tracked(tmp_paths, manifest, PASSING, seed=1), tmp_paths)
    registry.promote(tmp_paths, 1)
    registry.register(tracked(tmp_paths, manifest, LEAKY, seed=2), tmp_paths)
    with pytest.raises(registry.RegistryError, match="did not pass the models gates"):
        registry.promote(tmp_paths, 2)
    assert ModelCard.load(tmp_paths.models / MODEL_CARD).registered_version == 1


def test_ungated_installs_but_keeps_the_champion(tmp_paths, manifest):
    registry.register(tracked(tmp_paths, manifest, PASSING, seed=1), tmp_paths)
    registry.promote(tmp_paths, 1)
    registry.register(tracked(tmp_paths, manifest, LEAKY, seed=2), tmp_paths)
    registry.promote(tmp_paths, 2, ungated=True)
    assert ModelCard.load(tmp_paths.models / MODEL_CARD).registered_version == 2
    champions = [version.version for version in registry.versions(tmp_paths) if version.champion]
    assert champions == [1]


def test_promoting_a_version_that_does_not_exist(tmp_paths, manifest):
    registry.register(tracked(tmp_paths, manifest, PASSING), tmp_paths)
    with pytest.raises(registry.RegistryError, match="no version 9"):
        registry.promote(tmp_paths, 9)


def test_register_needs_a_tracked_run(tmp_paths, manifest):
    card = train_candidate(tmp_paths.models / ".candidate", manifest, PASSING)
    with pytest.raises(registry.RegistryError, match="no MLflow run"):
        registry.register(card, tmp_paths)


# --- what `fourthdown train` does with a candidate --------------------------------------


def test_train_promotes_a_passing_candidate(tmp_paths, manifest):
    card = tracked(tmp_paths, manifest, PASSING)
    _serve_candidate(
        card, tmp_paths, tmp_paths.models / ".candidate-0", promote="gated", track=True
    )
    assert ModelCard.load(tmp_paths.models / MODEL_CARD).registered_version == 1


def test_train_keeps_serving_the_old_model_when_gates_fail(tmp_paths, manifest):
    good = tracked(tmp_paths, manifest, PASSING, seed=1)
    _serve_candidate(
        good, tmp_paths, tmp_paths.models / ".candidate-1", promote="gated", track=True
    )
    bad = tracked(tmp_paths, manifest, LEAKY, seed=2)
    _serve_candidate(bad, tmp_paths, tmp_paths.models / ".candidate-2", promote="gated", track=True)
    served = ModelCard.load(tmp_paths.models / MODEL_CARD)
    assert served.registered_version == 1
    assert served.artifact_sha256 == good.artifact_sha256
    assert [version.gates for version in registry.versions(tmp_paths)] == ["fail", "pass"]


def test_train_without_tracking_copies_a_passing_candidate(tmp_paths, manifest):
    candidate = tmp_paths.models / ".candidate"
    card = train_candidate(candidate, manifest, PASSING)
    card.save(candidate / MODEL_CARD)
    _serve_candidate(card, tmp_paths, candidate, promote="gated", track=False)
    assert ModelCard.load(tmp_paths.models / MODEL_CARD).registered_version is None
    assert (tmp_paths.models / WP_ARTIFACT).exists()


def test_train_with_promote_never_serves_nothing_new(tmp_paths, manifest):
    card = tracked(tmp_paths, manifest, PASSING)
    _serve_candidate(
        card, tmp_paths, tmp_paths.models / ".candidate-0", promote="never", track=True
    )
    assert not (tmp_paths.models / MODEL_CARD).exists()
    assert [version.gates for version in registry.versions(tmp_paths)] == ["pass"]


# --- the CLI -----------------------------------------------------------------------------


def test_registry_commands(tmp_paths, manifest):
    registry.register(tracked(tmp_paths, manifest, PASSING, seed=1), tmp_paths)
    registry.register(tracked(tmp_paths, manifest, LEAKY, seed=2), tmp_paths)
    runner = CliRunner()
    root = ["--data-dir", str(tmp_paths.root)]
    promoted = runner.invoke(app, ["registry", "promote", "1", *root])
    assert promoted.exit_code == 0, promoted.output
    refused = runner.invoke(app, ["registry", "promote", "2", *root])
    assert refused.exit_code == 1
    listed = runner.invoke(app, ["registry", "list", *root])
    assert "v1 [champion]  gates pass" in listed.output
    assert "v2  gates fail" in listed.output
    assert "models.wp_auc" in listed.output


def test_registry_list_when_empty(tmp_paths):
    result = CliRunner().invoke(app, ["registry", "list", "--data-dir", str(tmp_paths.root)])
    assert result.exit_code == 0
    assert "no fourthdown-win-probability versions" in result.output
