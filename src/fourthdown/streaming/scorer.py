"""Consume ``nfl.plays``, score each situation, publish to ``nfl.win_probability``.

Delivery: a play's offset is committed only after its result has been flushed to the
output topic, so a scorer that dies mid-play scores that play again on restart (at least
once). Results are keyed by game and carry the play's sequence, and the feed keeps the
latest result per sequence, so the duplicate overwrites instead of adding a point
(effectively once end to end). A message that is not a valid event goes to
``nfl.plays.dlq`` with the reason and is committed, so one bad message cannot wedge the
partition behind it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fourthdown.models.winprob import GameState
from fourthdown.streaming.broker import Consumer, Message, Producer
from fourthdown.streaming.events import (
    DEAD_LETTER_TOPIC,
    WP_TOPIC,
    EventError,
    GameFinal,
    PlayEvent,
    WinProbability,
    decode_game_event,
    encode,
)

LOGGER = logging.getLogger(__name__)
GROUP = "fourthdown-scorer"


@dataclass
class Scorer:
    """Turns game events into the home team's win probability.

    ``probability`` is the served model's ``probability(GameState)``: the chance the team
    with the ball wins.
    """

    probability: Callable[[GameState], float]
    model_version: str
    last: dict[str, float] = field(default_factory=dict)
    previous_play: dict[str, str] = field(default_factory=dict)

    def score(self, event: PlayEvent | GameFinal) -> WinProbability:
        # A result is the situation after the previous play, so the change belongs to that
        # play; the first situation of a game has none.
        previous = self.last.get(event.game_id)
        after = self.previous_play.get(event.game_id, "start of the game")
        if isinstance(event, GameFinal):
            margin = event.home_score - event.away_score
            home_wp = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            result = WinProbability(
                game_id=event.game_id,
                play_id=-1,
                sequence=event.sequence,
                home_team=event.home_team,
                away_team=event.away_team,
                qtr=0,
                game_seconds_remaining=0.0,
                home_score=event.home_score,
                away_score=event.away_score,
                home_wp=home_wp,
                home_wp_change=0.0 if previous is None else home_wp - previous,
                description=after,
                nflfastr_home_wp=None,
                model_version=self.model_version,
                final=True,
            )
        else:
            offense = self.probability(
                GameState(
                    yardline_100=event.yardline_100,
                    down=event.down,
                    ydstogo=event.ydstogo,
                    game_seconds_remaining=event.game_seconds_remaining,
                    score_differential=event.score_differential,
                    posteam_timeouts_remaining=event.posteam_timeouts_remaining,
                    defteam_timeouts_remaining=event.defteam_timeouts_remaining,
                    posteam_is_home=event.posteam_is_home,
                    posteam_spread=event.posteam_spread,
                    goal_to_go=event.goal_to_go,
                    half_seconds_remaining=event.half_seconds_remaining,
                )
            )
            home_wp = offense if event.posteam_is_home else 1.0 - offense
            result = WinProbability(
                game_id=event.game_id,
                play_id=event.play_id,
                sequence=event.sequence,
                home_team=event.home_team,
                away_team=event.away_team,
                qtr=event.qtr,
                game_seconds_remaining=event.game_seconds_remaining,
                home_score=event.home_score,
                away_score=event.away_score,
                home_wp=round(home_wp, 4),
                home_wp_change=0.0 if previous is None else round(home_wp - previous, 4),
                description=after,
                nflfastr_home_wp=event.nflfastr_home_wp,
                model_version=self.model_version,
            )
        self.last[event.game_id] = result.home_wp
        if isinstance(event, PlayEvent):
            self.previous_play[event.game_id] = event.description
        return result


@dataclass
class RunStats:
    scored: int = 0
    dead_lettered: int = 0
    games_finished: int = 0


def _dead_letter(producer: Producer, message: Message, reason: str) -> None:
    envelope = {
        "reason": reason,
        "topic": message.topic,
        "partition": message.partition,
        "offset": message.offset,
        "value": message.value.decode("utf-8", errors="replace"),
    }
    producer.send(DEAD_LETTER_TOPIC, message.key, json.dumps(envelope).encode())


def run(
    consumer: Consumer,
    producer: Producer,
    scorer: Scorer,
    *,
    max_messages: int | None = None,
    idle_seconds: float | None = None,
    stop_after_finals: int | None = None,
    poll_timeout: float = 0.5,
) -> RunStats:
    """Score messages until ``max_messages``, ``stop_after_finals`` games end, or nothing
    arrives for ``idle_seconds``; without any of them, until interrupted."""
    stats = RunStats()
    handled = 0
    idle_since = time.monotonic()
    try:
        while max_messages is None or handled < max_messages:
            message = consumer.poll(poll_timeout)
            if message is None:
                if idle_seconds is not None and time.monotonic() - idle_since >= idle_seconds:
                    break
                continue
            idle_since = time.monotonic()
            try:
                event = decode_game_event(message.value)
            except EventError as error:
                LOGGER.warning("dead-lettering offset %s: %s", message.offset, error)
                _dead_letter(producer, message, str(error))
                producer.flush()
                consumer.commit(message)
                stats.dead_lettered += 1
                handled += 1
                continue
            result = scorer.score(event)
            producer.send(WP_TOPIC, event.game_id, encode(result))
            producer.flush()  # delivered before the offset moves: at least once
            consumer.commit(message)
            stats.scored += 1
            handled += 1
            if result.final:
                stats.games_finished += 1
                if stop_after_finals is not None and stats.games_finished >= stop_after_finals:
                    break
    finally:
        consumer.close()
    return stats
