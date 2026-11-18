"""Publish one game from the warehouse to ``nfl.plays``, in order, paced by the game clock."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from fourthdown.streaming.broker import Producer
from fourthdown.streaming.events import PLAYS_TOPIC, GameEvent, GameFinal, PlayEvent, encode

LOGGER = logging.getLogger(__name__)

MAX_PAUSE = 5.0
"""Longest real-time wait between two plays, whatever the speed: halftime is not a demo."""

GAME_SQL = """
SELECT
    p.play_id,
    p.home_team,
    p.away_team,
    p.posteam,
    p.qtr,
    p.down,
    p.ydstogo,
    p.yardline_100,
    coalesce(p.goal_to_go, false) AS goal_to_go,
    p.game_seconds_remaining,
    coalesce(p.half_seconds_remaining, 0) AS half_seconds_remaining,
    p.score_differential,
    coalesce(p.posteam_timeouts_remaining, 3) AS posteam_timeouts_remaining,
    coalesce(p.defteam_timeouts_remaining, 3) AS defteam_timeouts_remaining,
    p.posteam_is_home,
    coalesce(CASE WHEN p.posteam_is_home THEN p.spread_line ELSE -p.spread_line END, 0)
        AS posteam_spread,
    CASE WHEN p.posteam_is_home THEN p.posteam_score ELSE p.defteam_score END AS home_score,
    CASE WHEN p.posteam_is_home THEN p.defteam_score ELSE p.posteam_score END AS away_score,
    coalesce(p.play_desc, '') AS description,
    CASE WHEN p.posteam_is_home THEN p.wp ELSE 1 - p.wp END AS nflfastr_home_wp
FROM plays AS p
WHERE p.game_id = ?
  AND p.posteam IS NOT NULL
  AND p.down IS NOT NULL
  AND p.game_seconds_remaining IS NOT NULL
  AND p.score_differential IS NOT NULL
ORDER BY p.play_id
"""


class GameNotFound(LookupError):
    pass


def game_events(connection: Any, game_id: str) -> list[GameEvent]:
    """The game's scrimmage plays (situations the model can score) and its final score."""
    final = connection.execute(
        "SELECT home_team, away_team, home_score, away_score FROM games WHERE game_id = ?",
        [game_id],
    ).fetchone()
    if final is None:
        raise GameNotFound(f"no game {game_id!r} in the warehouse")
    cursor = connection.execute(GAME_SQL, [game_id])
    columns = [column[0] for column in cursor.description]
    events: list[GameEvent] = []
    for sequence, row in enumerate(cursor.fetchall()):
        record = dict(zip(columns, row, strict=True))
        events.append(
            PlayEvent(
                game_id=game_id,
                sequence=sequence,
                play_id=int(record["play_id"]),
                home_team=record["home_team"],
                away_team=record["away_team"],
                posteam=record["posteam"],
                qtr=int(record["qtr"]),
                down=int(record["down"]),
                ydstogo=float(record["ydstogo"]),
                yardline_100=float(record["yardline_100"]),
                goal_to_go=bool(record["goal_to_go"]),
                game_seconds_remaining=float(record["game_seconds_remaining"]),
                half_seconds_remaining=float(record["half_seconds_remaining"]),
                score_differential=float(record["score_differential"]),
                posteam_timeouts_remaining=int(record["posteam_timeouts_remaining"]),
                defteam_timeouts_remaining=int(record["defteam_timeouts_remaining"]),
                posteam_is_home=bool(record["posteam_is_home"]),
                posteam_spread=float(record["posteam_spread"]),
                home_score=int(record["home_score"] or 0),
                away_score=int(record["away_score"] or 0),
                description=record["description"],
                nflfastr_home_wp=(
                    None
                    if record["nflfastr_home_wp"] is None
                    else float(record["nflfastr_home_wp"])
                ),
            )
        )
    # A game nflverse has not posted a result for yet ends on its last recorded score.
    last = next((event for event in reversed(events) if isinstance(event, PlayEvent)), None)
    home_score = final[2] if final[2] is not None else (last.home_score if last else 0)
    away_score = final[3] if final[3] is not None else (last.away_score if last else 0)
    events.append(
        GameFinal(
            game_id=game_id,
            sequence=len(events),
            home_team=final[0],
            away_team=final[1],
            home_score=int(home_score),
            away_score=int(away_score),
        )
    )
    return events


def pauses(events: Sequence[GameEvent], speed: float) -> list[float]:
    """Seconds to wait before each event: the game clock between plays over ``speed``.

    Overtime restarts the clock, so a clock that goes up counts as no time. ``speed`` 0
    replays as fast as the broker takes it.
    """
    waits: list[float] = []
    previous: float | None = None
    for event in events:
        clock = event.game_seconds_remaining if isinstance(event, PlayEvent) else previous
        if speed <= 0 or previous is None or clock is None:
            waits.append(0.0)
        else:
            waits.append(min(max(previous - clock, 0.0) / speed, MAX_PAUSE))
        previous = clock if clock is not None else previous
    return waits


def replay(
    producer: Producer,
    events: Sequence[GameEvent],
    *,
    speed: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Publish ``events`` to ``nfl.plays`` keyed by game, waiting between them. Returns
    the number published."""
    for event, wait in zip(events, pauses(events, speed), strict=True):
        if wait:
            sleep(wait)
        producer.send(PLAYS_TOPIC, event.game_id, encode(event))
    producer.flush()
    LOGGER.info("replayed %d events of %s", len(events), events[0].game_id if events else "-")
    return len(events)
