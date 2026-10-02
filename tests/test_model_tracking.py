from __future__ import annotations

import pytest
import torch
from mlflow.tracking import MlflowClient

from fourthdown.data import provenance
from fourthdown.models import features, tracking
from fourthdown.models.card import MODEL_CARD, ModelCard
from fourthdown.models.features import SeasonSplit
from fourthdown.models.winprob import TrainingHistory, WinProbabilityModel, WinProbabilityNet
from fourthdown.serving import winprob as service

SPLIT = SeasonSplit(train=(2009, 2010), valid=(2011,), test=(2012,))
STATE = {
    "yardline_100": 45,
    "down": 2,
    "ydstogo": 7,
    "game_seconds_remaining": 900,
    "score_differential": 3,
}


@pytest.fixture()
def manifest():
    return provenance.Manifest(
        processed={
            2009: provenance.PartitionRecord(2009, "p", "a" * 64, 10, "b" * 64, "c" * 64, "t")
        }
    )


@pytest.fixture()
def model_dir(tmp_path, manifest):
    directory = tmp_path / "models"
    torch.manual_seed(0)
    net = WinProbabilityNet(len(features.WP_FEATURES))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(1e-3)
    model = WinProbabilityModel(features.WP_FEATURES, net)
    model.save(directory / "winprob.pt")
    card = ModelCard.build(
        artifact=directory / "winprob.pt",
        manifest=manifest,
        params=tracking.training_params(SPLIT, epochs=3, audit_sample=10),
        metrics={"wp_log_loss": 0.45},
    )
    card.save(directory / MODEL_CARD)
    return directory


def test_training_params_capture_split_and_hyperparameters():
    params = tracking.training_params(SPLIT, epochs=5, audit_sample=10)
    assert params["split.train"] == "2009-2010"
    assert params["split.test"] == "2012"
    assert params["wp.max_epochs"] == 5
    assert "wp.learning_rate" in params


def test_model_card_round_trips_and_carries_the_data_version(model_dir, manifest):
    card = ModelCard.load(model_dir / MODEL_CARD)
    assert card.data_version == manifest.data_version
    assert card.seasons == {"train": "2009-2010", "valid": "2011", "test": "2012"}
    assert card.artifact_sha256 == provenance.sha256_file(model_dir / "winprob.pt")


def test_log_run_records_data_version_config_and_artifacts(model_dir, manifest, tmp_paths):
    card = ModelCard.load(model_dir / MODEL_CARD)
    history = TrainingHistory(train_loss=[0.6, 0.5], valid_loss=[0.62, 0.55], best_epoch=1)
    card_path = tmp_paths.models / MODEL_CARD
    stamped = tracking.log_run(
        card,
        paths=tmp_paths,
        card_path=card_path,
        manifest=manifest,
        history=history,
        artifacts=[model_dir / "winprob.pt"],
    )
    client = MlflowClient(tracking_uri=tracking.tracking_uri(tmp_paths))
    run = client.get_run(stamped.mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["data_version"] == manifest.data_version
    assert run.data.params["data_version"] == manifest.data_version
    assert run.data.params["split.train"] == "2009-2010"
    assert run.data.metrics["wp_log_loss"] == pytest.approx(0.45)
    losses = client.get_metric_history(run.info.run_id, "wp_valid_loss")
    assert [m.value for m in sorted(losses, key=lambda m: m.step)] == [0.62, 0.55]
    logged = {item.path for item in client.list_artifacts(run.info.run_id)}
    assert {"winprob.pt", MODEL_CARD} <= logged
    assert ModelCard.load(card_path).mlflow_run_id == run.info.run_id


def test_service_predicts_with_provenance(model_dir, manifest):
    client = service.create_app(model_dir).test_client()
    health = client.get("/health").get_json()
    assert health["model"]["data_version"] == manifest.data_version
    single = client.post("/v1/win-probability", json=STATE).get_json()
    assert 0.0 < single["win_probability"] < 1.0
    assert single["model"]["artifact_sha256"] == provenance.sha256_file(model_dir / "winprob.pt")
    batch = client.post("/v1/win-probability", json={"states": [STATE, STATE]}).get_json()
    assert batch["win_probabilities"] == pytest.approx([single["win_probability"]] * 2, abs=1e-6)
    card = client.get("/v1/model").get_json()
    assert card["features"] == list(features.WP_FEATURES)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"down": 5}, "down"),
        ({"down": 2.5}, "whole number"),
        ({"yardline_100": 0}, "yardline_100"),
        ({"score_differential": "3"}, "number"),
        ({"posteam_is_home": "yes"}, "true or false"),
    ],
)
def test_service_rejects_impossible_states(model_dir, change, message):
    client = service.create_app(model_dir).test_client()
    response = client.post("/v1/win-probability", json={**STATE, **change})
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_service_rejects_missing_fields_and_bad_batches(model_dir):
    client = service.create_app(model_dir).test_client()
    missing = {key: value for key, value in STATE.items() if key != "ydstogo"}
    assert client.post("/v1/win-probability", json=missing).status_code == 400
    assert client.post("/v1/win-probability", json={"states": []}).status_code == 400
    too_many = {"states": [STATE] * (service.MAX_BATCH + 1)}
    assert client.post("/v1/win-probability", json=too_many).status_code == 400


def test_service_refuses_weights_that_do_not_match_their_card(model_dir):
    other = WinProbabilityModel(features.WP_FEATURES, WinProbabilityNet(len(features.WP_FEATURES)))
    other.net.scale.fill_(2.0)
    other.save(model_dir / "winprob.pt")
    with pytest.raises(service.ProvenanceError, match="SHA-256"):
        service.create_app(model_dir)


def test_service_refuses_to_start_without_a_card(tmp_path):
    with pytest.raises(service.ProvenanceError, match="no model card"):
        service.create_app(tmp_path)
