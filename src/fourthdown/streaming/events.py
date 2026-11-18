"""The messages on the two topics, as JSON. Every message is keyed by ``game_id``, so all
of a game's events land on one partition and stay in order."""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Literal

PLAYS_TOPIC = "nfl.plays"
WP_TOPIC = "nfl.win_probability"
DEAD_LETTER_TOPIC = "nfl.plays.dlq"
SCHEMA_VERSION = 1


class EventError(ValueError):
    """A message that is not a valid event: wrong JSON, wrong kind, missing fields."""


@dataclass(frozen=True)
class PlayEvent:
    """One scrimmage play's pre-snap situation, from the offense's point of view."""

    game_id: str
    play_id: int
    sequence: int
    home_team: str
    away_team: str
    posteam: str
    qtr: int
    down: int
    ydstogo: float
    yardline_100: float
    goal_to_go: bool
    game_seconds_remaining: float
    half_seconds_remaining: float
    score_differential: float
    posteam_timeouts_remaining: int
    defteam_timeouts_remaining: int
    posteam_is_home: bool
    posteam_spread: float
    home_score: int
    away_score: int
    description: str
    nflfastr_home_wp: float | None = None
    kind: Literal["play"] = "play"


@dataclass(frozen=True)
class GameFinal:
    """The end of a replayed game, with its final score."""

    game_id: str
    sequence: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    kind: Literal["final"] = "final"


@dataclass(frozen=True)
class WinProbability:
    """The home team's win probability after scoring one event."""

    game_id: str
    play_id: int
    sequence: int
    home_team: str
    away_team: str
    qtr: int
    game_seconds_remaining: float
    home_score: int
    away_score: int
    home_wp: float
    home_wp_change: float
    description: str
    """The play that led to this situation (and so caused ``home_wp_change``)."""
    nflfastr_home_wp: float | None
    model_version: str
    final: bool = False


GameEvent = PlayEvent | GameFinal
_KINDS: dict[str, type[PlayEvent] | type[GameFinal]] = {"play": PlayEvent, "final": GameFinal}


def encode(event: PlayEvent | GameFinal | WinProbability) -> bytes:
    return json.dumps({"v": SCHEMA_VERSION, **asdict(event)}, separators=(",", ":")).encode()


def _build(cls: type[Any], payload: dict[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    missing = sorted(
        field.name
        for field in fields(cls)
        if field.name not in payload
        and field.default is MISSING
        and field.default_factory is MISSING
    )
    if missing:
        raise EventError(f"{cls.__name__} is missing {', '.join(missing)}")
    try:
        return cls(**{key: value for key, value in payload.items() if key in names})
    except TypeError as error:
        raise EventError(str(error)) from error


def decode_game_event(value: bytes) -> GameEvent:
    """A message from ``nfl.plays``; raises ``EventError`` for anything else."""
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EventError(f"not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EventError("not a JSON object")
    if payload.get("v") != SCHEMA_VERSION:
        raise EventError(f"schema version {payload.get('v')!r}, expected {SCHEMA_VERSION}")
    cls = _KINDS.get(str(payload.get("kind")))
    if cls is None:
        raise EventError(f"unknown kind {payload.get('kind')!r}")
    event: GameEvent = _build(cls, payload)
    return event


def decode_win_probability(value: bytes) -> WinProbability:
    payload = json.loads(value)
    result: WinProbability = _build(WinProbability, payload)
    return result
