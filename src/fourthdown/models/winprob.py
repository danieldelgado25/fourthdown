"""Win probability from pre-snap game state, in PyTorch.

The model answers one question: given the score, the clock, the field, and who is
favoured, how often does the team with the ball go on to win? Everything else in phase
04 is built on it -- a fourth-down recommendation is just this model evaluated at the
three states a coach can choose between.

Design notes worth defending:

*A small MLP, not a tree ensemble.* The inputs are continuous and the surface is smooth
and monotone in most of them (more time, more points, higher probability), which is what
a network extrapolates well and a tree approximates with steps. It also gives a
differentiable function of the state, which matters if this is ever used inside a search.

*Trained on every play, evaluated on unseen seasons.* Plays inside a game are heavily
correlated, so a random split would leak the outcome; splits are season-disjoint.

*Calibration is the point.* A win-probability number is only meaningful if the 70% plays
win 70% of the time, so the score card reports expected calibration error alongside log
loss, and `nflfastR`'s own ``vegas_wp`` runs as the comparison column.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fourthdown.models.features import Dataset, feature_matrix, target_vector

LOGGER = logging.getLogger(__name__)

HIDDEN = (64, 64)
DROPOUT = 0.1
BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 17
ARTIFACT = "winprob.pt"


@dataclass(frozen=True)
class GameState:
    """A situation to evaluate, from the point of view of the team with the ball."""

    yardline_100: float
    down: int
    ydstogo: float
    game_seconds_remaining: float
    score_differential: float
    posteam_timeouts_remaining: int = 3
    defteam_timeouts_remaining: int = 3
    posteam_is_home: bool = True
    posteam_spread: float = 0.0
    goal_to_go: bool = False
    half_seconds_remaining: float | None = None

    def half_clock(self) -> float:
        """Second-half plays have no first half left; before then the halves are symmetric."""
        if self.half_seconds_remaining is not None:
            return self.half_seconds_remaining
        return min(self.game_seconds_remaining, 1800.0)

    def possession_flip(self, yardline_100: float, *, seconds_elapsed: float = 6.0) -> GameState:
        """The same game after the other team takes over at ``yardline_100``.

        Field position mirrors (their 30 is our 70), the score differential negates, the
        home flag and the spread negate with it, and the ball is first and ten.
        """
        clock = max(self.game_seconds_remaining - seconds_elapsed, 0.0)
        half = max(self.half_clock() - seconds_elapsed, 0.0)
        return GameState(
            yardline_100=float(np.clip(yardline_100, 1.0, 99.0)),
            down=1,
            ydstogo=min(10.0, float(np.clip(yardline_100, 1.0, 99.0))),
            game_seconds_remaining=clock,
            score_differential=-self.score_differential,
            posteam_timeouts_remaining=self.defteam_timeouts_remaining,
            defteam_timeouts_remaining=self.posteam_timeouts_remaining,
            posteam_is_home=not self.posteam_is_home,
            posteam_spread=-self.posteam_spread,
            goal_to_go=yardline_100 <= 10,
            half_seconds_remaining=half,
        )

    def as_features(self, features: Sequence[str]) -> np.ndarray:
        values = {
            "yardline_100": float(self.yardline_100),
            "down": float(self.down),
            "ydstogo": float(self.ydstogo),
            "goal_to_go": float(self.goal_to_go),
            "game_seconds_remaining": float(self.game_seconds_remaining),
            "half_seconds_remaining": float(self.half_clock()),
            "score_differential": float(self.score_differential),
            "posteam_timeouts_remaining": float(self.posteam_timeouts_remaining),
            "defteam_timeouts_remaining": float(self.defteam_timeouts_remaining),
            "posteam_is_home": float(self.posteam_is_home),
            "posteam_spread": float(self.posteam_spread),
        }
        missing = [name for name in features if name not in values]
        if missing:
            raise ValueError(f"GameState cannot supply {missing}")
        return np.array([[values[name] for name in features]], dtype=np.float32)


class WinProbabilityNet(nn.Module):
    """Standardise, then a two-layer MLP to a single logit."""

    center: torch.Tensor
    scale: torch.Tensor

    def __init__(self, inputs: int, hidden: Sequence[int] = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.register_buffer("center", torch.zeros(inputs))
        self.register_buffer("scale", torch.ones(inputs))
        layers: list[nn.Module] = []
        width = inputs
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.stack = nn.Sequential(*layers)

    def adapt(self, matrix: torch.Tensor) -> None:
        """Freeze the training set's mean and spread into the module itself.

        Keeping them as buffers means the saved checkpoint is self-contained: loading it
        cannot be paired with the wrong scaler.
        """
        self.center.copy_(matrix.mean(dim=0))
        spread = matrix.std(dim=0)
        self.scale.copy_(torch.where(spread > 1e-6, spread, torch.ones_like(spread)))

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return self.stack((matrix - self.center) / self.scale).squeeze(-1)


@dataclass
class TrainingHistory:
    """Per-epoch validation loss, kept so the report can show the stopping point."""

    train_loss: list[float]
    valid_loss: list[float]
    best_epoch: int

    @property
    def best_valid_loss(self) -> float:
        return self.valid_loss[self.best_epoch]


class WinProbabilityModel:
    """A trained network plus the feature order it expects."""

    def __init__(self, features: Sequence[str], net: WinProbabilityNet) -> None:
        self.features = tuple(features)
        self.net = net
        self.net.eval()

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(np.asarray(matrix, dtype=np.float32)))
            return torch.sigmoid(logits).numpy().astype(np.float64)

    def probability(self, state: GameState) -> float:
        """Win probability for the team with the ball in ``state``."""
        return float(self.predict(state.as_features(self.features))[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": list(self.features), "state_dict": self.net.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinProbabilityModel:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        features = tuple(str(name) for name in payload["features"])
        net = WinProbabilityNet(len(features))
        net.load_state_dict(payload["state_dict"])
        return cls(features, net)


def _loader(
    matrix: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    tensors = torch.utils.data.TensorDataset(torch.from_numpy(matrix), torch.from_numpy(labels))
    return torch.utils.data.DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def train(
    dataset: Dataset,
    *,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> tuple[WinProbabilityModel, TrainingHistory]:
    """Fit on ``dataset.train``, early-stop on ``dataset.valid``, never touch ``test``."""
    torch.manual_seed(seed)
    train_x = feature_matrix(dataset.train.frame, dataset.features)
    train_y = target_vector(dataset.train.frame, dataset.target)
    valid_x = feature_matrix(dataset.valid.frame, dataset.features)
    valid_y = target_vector(dataset.valid.frame, dataset.target)

    net = WinProbabilityNet(len(dataset.features))
    net.adapt(torch.from_numpy(train_x))
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    batches = _loader(train_x, train_y, batch_size=batch_size, shuffle=True)
    valid_batches = _loader(valid_x, valid_y, batch_size=batch_size * 4, shuffle=False)

    history = TrainingHistory(train_loss=[], valid_loss=[], best_epoch=0)
    best_state = {key: value.clone() for key, value in net.state_dict().items()}
    best_loss = float("inf")
    stale = 0
    for epoch in range(max_epochs):
        net.train()
        running = 0.0
        seen = 0
        for features, labels in batches:
            optimiser.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(labels)
            seen += len(labels)
        net.eval()
        with torch.no_grad():
            total = 0.0
            count = 0
            for features, labels in valid_batches:
                total += float(criterion(net(features), labels)) * len(labels)
                count += len(labels)
        train_loss = running / max(seen, 1)
        valid_loss = total / max(count, 1)
        history.train_loss.append(train_loss)
        history.valid_loss.append(valid_loss)
        LOGGER.info("epoch %02d train %.4f valid %.4f", epoch + 1, train_loss, valid_loss)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            history.best_epoch = epoch
            best_state = {key: value.clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                LOGGER.info("stopping early at epoch %d", epoch + 1)
                break
    net.load_state_dict(best_state)
    return WinProbabilityModel(dataset.features, net), history


def write_history(history: TrainingHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train_loss": history.train_loss,
                "valid_loss": history.valid_loss,
                "best_epoch": history.best_epoch,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
