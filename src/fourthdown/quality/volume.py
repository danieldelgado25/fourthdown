"""Row-count checks: is this week the size a week of football is?

Every game from 2009 to 2025 has between 135 and 245 rows of play-by-play (median 175),
and every week averages between 161 and 207 per game. A file cut off mid-write, a game
published half-parsed, or rows duplicated by an upstream join all show up as counts
outside those bands, long before they show up as a strange answer.

The third check is drift. When nflverse republishes a week (it corrects stats for days
after a game), the row count moves by a handful of plays. A week that comes back 5% larger
or smaller than when it was last ingested is not a correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class VolumeBands:
    min_plays_per_game: int = 100
    max_plays_per_game: int = 300
    min_week_mean: float = 145.0
    max_week_mean: float = 225.0
    max_redelivery_change: float = 0.05


def check_week(
    plays: pl.DataFrame,
    key: str,
    *,
    previous_rows: int | None = None,
    bands: VolumeBands | None = None,
) -> list[str]:
    """Problems with one week's row counts; ``plays`` needs a ``game_id`` column."""
    bands = bands or VolumeBands()
    if plays.is_empty():
        return [f"{key}: no plays"]
    per_game = plays.group_by("game_id").len().sort("game_id")
    problems: list[str] = []
    small = per_game.filter(pl.col("len") < bands.min_plays_per_game)
    if not small.is_empty():
        problems.append(
            f"{key}: {small.height} game(s) with fewer than {bands.min_plays_per_game} plays "
            f"({_games(small)}); a game published part-way, or a truncated file"
        )
    large = per_game.filter(pl.col("len") > bands.max_plays_per_game)
    if not large.is_empty():
        problems.append(
            f"{key}: {large.height} game(s) with more than {bands.max_plays_per_game} plays "
            f"({_games(large)}); duplicated rows"
        )
    mean = plays.height / per_game.height
    if not bands.min_week_mean <= mean <= bands.max_week_mean:
        problems.append(
            f"{key}: {mean:.0f} plays per game on average, outside "
            f"{bands.min_week_mean:.0f}-{bands.max_week_mean:.0f}"
        )
    if previous_rows:
        change = plays.height / previous_rows - 1
        if abs(change) > bands.max_redelivery_change:
            problems.append(
                f"{key}: {plays.height:,} plays, {change:+.1%} against the {previous_rows:,} "
                f"ingested before; a correction moves a week by a handful of plays, not "
                f"{bands.max_redelivery_change:.0%}"
            )
    return problems


def _games(frame: pl.DataFrame, limit: int = 3) -> str:
    shown = [f"{game} has {count}" for game, count in frame.head(limit).iter_rows()]
    more = frame.height - limit
    return ", ".join(shown) + (f", and {more} more" if more > 0 else "")
