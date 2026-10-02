"""A small HTTP service in front of the win-probability model, built to run in a container.

It loads one artifact and its model card at startup and refuses to start if the
artifact's SHA-256 does not match the card, so the provenance it reports is the
provenance of the weights it is actually serving. Every prediction carries the model's
data version and MLflow run id.

Run it with gunicorn (as the container does)::

    gunicorn "fourthdown.serving.winprob:create_app()"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request

from fourthdown.config import data_paths
from fourthdown.data.provenance import sha256_file
from fourthdown.models.card import MODEL_CARD, ModelCard
from fourthdown.models.winprob import ARTIFACT, GameState, WinProbabilityModel

LOGGER = logging.getLogger(__name__)

MODEL_DIR_ENV = "FOURTHDOWN_MODEL_DIR"
MAX_BATCH = 1000
GAME_SECONDS = 3600.0
HALF_SECONDS = 1800.0


class ProvenanceError(RuntimeError):
    """The artifact on disk is not the one its model card describes."""


@dataclass(frozen=True)
class LoadedModel:
    model: WinProbabilityModel
    card: ModelCard

    def provenance(self) -> dict[str, object]:
        return {
            "data_version": self.card.data_version,
            "mlflow_run_id": self.card.mlflow_run_id,
            "artifact_sha256": self.card.artifact_sha256,
            "git_commit": self.card.git_commit,
        }


def load(model_dir: Path) -> LoadedModel:
    """Load the artifact and its card, checking one against the other."""
    card_path = model_dir / MODEL_CARD
    if not card_path.exists():
        raise ProvenanceError(f"no model card at {card_path}; run `fourthdown train`")
    card = ModelCard.load(card_path)
    artifact = model_dir / card.artifact
    if not artifact.exists():
        raise ProvenanceError(f"model card names {card.artifact}, which is not in {model_dir}")
    digest = sha256_file(artifact)
    if digest != card.artifact_sha256:
        raise ProvenanceError(
            f"{artifact} has SHA-256 {digest[:12]}, but its card says "
            f"{card.artifact_sha256[:12]}; retrain or restore the matching card"
        )
    model = WinProbabilityModel.load(artifact)
    if model.features != card.features:
        raise ProvenanceError("artifact features do not match the model card")
    return LoadedModel(model, card)


def _number(payload: dict[str, object], key: str, *, low: float, high: float) -> float:
    if key not in payload:
        raise ValueError(f"{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    number = float(value)
    if not low <= number <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return number


def _optional_number(
    payload: dict[str, object], key: str, default: float, *, low: float, high: float
) -> float:
    return _number(payload, key, low=low, high=high) if key in payload else default


def _flag(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def parse_state(payload: object) -> GameState:
    """A pre-snap situation from JSON, rejecting anything outside a real game."""
    if not isinstance(payload, dict):
        raise ValueError("each state must be a JSON object")
    yardline = _number(payload, "yardline_100", low=1, high=99)
    down = _number(payload, "down", low=1, high=4)
    if not down.is_integer():
        raise ValueError("down must be a whole number")
    togo = _number(payload, "ydstogo", low=1, high=99)
    clock = _number(payload, "game_seconds_remaining", low=0, high=GAME_SECONDS)
    half = (
        _number(payload, "half_seconds_remaining", low=0, high=HALF_SECONDS)
        if "half_seconds_remaining" in payload
        else None
    )
    return GameState(
        yardline_100=yardline,
        down=int(down),
        ydstogo=togo,
        game_seconds_remaining=clock,
        score_differential=_number(payload, "score_differential", low=-100, high=100),
        posteam_timeouts_remaining=int(
            _optional_number(payload, "posteam_timeouts_remaining", 3, low=0, high=3)
        ),
        defteam_timeouts_remaining=int(
            _optional_number(payload, "defteam_timeouts_remaining", 3, low=0, high=3)
        ),
        posteam_is_home=_flag(payload, "posteam_is_home", True),
        posteam_spread=_optional_number(payload, "posteam_spread", 0.0, low=-50, high=50),
        goal_to_go=_flag(payload, "goal_to_go", yardline <= togo),
        half_seconds_remaining=half,
    )


def create_app(model_dir: Path | None = None) -> Flask:
    """Build the service. The model directory defaults to ``FOURTHDOWN_MODEL_DIR``."""
    resolved = model_dir or Path(os.environ.get(MODEL_DIR_ENV) or data_paths().models)
    loaded = load(resolved)
    LOGGER.info(
        "serving %s (data version %s, run %s)",
        ARTIFACT,
        loaded.card.data_version,
        loaded.card.mlflow_run_id,
    )
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok", "model": loaded.provenance()})

    @app.get("/v1/model")
    def model_card() -> Response:
        card = loaded.card
        return jsonify(
            {
                **loaded.provenance(),
                "model": card.model,
                "features": list(card.features),
                "seasons": card.seasons,
                "metrics": card.metrics,
                "trained_at": card.trained_at,
                "package_version": card.package_version,
            }
        )

    @app.post("/v1/win-probability")
    def predict() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        batch = isinstance(payload, dict) and "states" in payload
        raw = payload["states"] if isinstance(payload, dict) and batch else [payload]
        if not isinstance(raw, list) or not raw:
            return jsonify({"error": "states must be a non-empty list"}), 400
        if len(raw) > MAX_BATCH:
            return jsonify({"error": f"at most {MAX_BATCH} states per request"}), 400
        try:
            states = [parse_state(item) for item in raw]
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        features = loaded.model.features
        matrix = np.vstack([state.as_features(features) for state in states])
        probabilities = [float(value) for value in loaded.model.predict(matrix)]
        body: dict[str, object] = {"model": loaded.provenance()}
        if batch:
            body["win_probabilities"] = probabilities
        else:
            body["win_probability"] = probabilities[0]
        return jsonify(body)

    return app
