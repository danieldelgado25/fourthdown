"""The consumer end of ``nfl.win_probability``: a live text feed and a game summary."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fourthdown.streaming.broker import Consumer
from fourthdown.streaming.events import WinProbability, decode_win_probability

GROUP = "fourthdown-feed"
BAR = 20


@dataclass
class Feed:
    """Results per game, keyed by sequence, so a redelivered result replaces its first copy."""

    games: dict[str, dict[int, WinProbability]] = field(default_factory=dict)

    def add(self, result: WinProbability) -> bool:
        """Keep ``result``; False when it was a duplicate of one already held."""
        plays = self.games.setdefault(result.game_id, {})
        duplicate = result.sequence in plays
        plays[result.sequence] = result
        return not duplicate

    def timeline(self, game_id: str) -> list[WinProbability]:
        return [result for _, result in sorted(self.games.get(game_id, {}).items())]


def clock(result: WinProbability) -> str:
    if result.final:
        return "FINAL  "
    seconds = result.game_seconds_remaining
    if result.qtr >= 5:
        return f"OT {int(seconds // 60):2d}:{int(seconds % 60):02d}"
    in_quarter = seconds - (4 - result.qtr) * 900
    return f"Q{result.qtr} {int(in_quarter // 60):2d}:{int(in_quarter % 60):02d}"


def line(result: WinProbability) -> str:
    """``Q4  2:11  SF 19-16 KC  [##########..........] SF 50.0% (+12.3)``"""
    filled = round(result.home_wp * BAR)
    bar = "#" * filled + "." * (BAR - filled)
    change = f"{result.home_wp_change * 100:+5.1f}"
    reference = (
        f"  nflfastR {result.nflfastr_home_wp * 100:5.1f}%"
        if result.nflfastr_home_wp is not None
        else ""
    )
    text = f"after {result.description[:60]}"
    return (
        f"{clock(result)}  {result.away_team} {result.away_score:>2}-{result.home_score:<2} "
        f"{result.home_team}  [{bar}] {result.home_team} {result.home_wp * 100:5.1f}% ({change})"
        f"{reference}  {text}"
    ).rstrip()


@dataclass(frozen=True)
class Summary:
    game_id: str
    plays: int
    lead_changes: int
    biggest_swing: WinProbability | None
    mean_gap_to_nflfastr: float | None

    def render(self) -> str:
        lines = [
            f"{self.game_id}: {self.plays} plays scored, {self.lead_changes} changes of favourite"
        ]
        if self.biggest_swing is not None:
            swing = self.biggest_swing
            lines.append(
                f"biggest swing: {swing.home_wp_change * 100:+.1f} points for "
                f"{swing.home_team}, from {swing.description[:90]}"
            )
        if self.mean_gap_to_nflfastr is not None:
            lines.append(
                "mean gap to nflfastR's win probability: "
                f"{self.mean_gap_to_nflfastr * 100:.1f} points"
            )
        return "\n".join(lines)


def summarize(feed: Feed, game_id: str) -> Summary:
    timeline = feed.timeline(game_id)
    plays = [result for result in timeline if not result.final]
    favourites = [result.home_wp > 0.5 for result in plays if result.home_wp != 0.5]
    changes = sum(1 for a, b in zip(favourites, favourites[1:], strict=False) if a != b)
    gaps = [
        abs(result.home_wp - result.nflfastr_home_wp)
        for result in plays
        if result.nflfastr_home_wp is not None
    ]
    return Summary(
        game_id=game_id,
        plays=len(plays),
        lead_changes=changes,
        biggest_swing=max(plays[1:], key=lambda r: abs(r.home_wp_change), default=None),
        mean_gap_to_nflfastr=sum(gaps) / len(gaps) if gaps else None,
    )


def watch(
    consumer: Consumer,
    feed: Feed,
    *,
    emit: Callable[[str], None] = print,
    stop_after_finals: int | None = 1,
    idle_seconds: float | None = None,
    poll_timeout: float = 0.5,
) -> Feed:
    """Print each new result as it arrives, until a game ends (or nothing arrives)."""
    finals = 0
    idle_since = time.monotonic()
    try:
        while stop_after_finals is None or finals < stop_after_finals:
            message = consumer.poll(poll_timeout)
            if message is None:
                if idle_seconds is not None and time.monotonic() - idle_since >= idle_seconds:
                    break
                continue
            idle_since = time.monotonic()
            result = decode_win_probability(message.value)
            if feed.add(result):
                emit(line(result))
            consumer.commit(message)
            if result.final:
                finals += 1
    finally:
        consumer.close()
    return feed
