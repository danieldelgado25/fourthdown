"""The model card: provenance that ships next to a win-probability artifact.

It lives apart from :mod:`fourthdown.models.tracking` so the serving container can read
a card without installing MLflow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fourthdown import __version__
from fourthdown.data import provenance
from fourthdown.models import winprob

MODEL_CARD = "model_card.json"

ParamValue = str | int | float


@dataclass(frozen=True)
class ModelCard:
    """Provenance for one win-probability artifact."""

    model: str
    artifact: str
    artifact_sha256: str
    features: tuple[str, ...]
    data_version: str
    seasons: dict[str, str]
    params: dict[str, ParamValue]
    metrics: dict[str, float]
    trained_at: str
    git_commit: str | None
    package_version: str
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None

    @classmethod
    def build(
        cls,
        *,
        artifact: Path,
        manifest: provenance.Manifest,
        params: Mapping[str, ParamValue],
        metrics: Mapping[str, float],
    ) -> ModelCard:
        model = winprob.WinProbabilityModel.load(artifact)
        return cls(
            model="win_probability",
            artifact=artifact.name,
            artifact_sha256=provenance.sha256_file(artifact),
            features=model.features,
            data_version=manifest.data_version,
            seasons={
                key.removeprefix("split."): str(value)
                for key, value in params.items()
                if key.startswith("split.")
            },
            params=dict(params),
            metrics=dict(metrics),
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=provenance.git_commit(),
            package_version=__version__,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ModelCard:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["features"] = tuple(payload["features"])
        return cls(**payload)
