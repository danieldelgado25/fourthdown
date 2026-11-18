"""Freshness: during the season, the warehouse keeps up with the games.

Freshness is measured against the schedule rather than a clock. "No update in 7 days" is
wrong in February and in a bye week; "a week ended two days ago and is not in the data"
is right all year. Two ways to be stale:

* a week the schedule shows over (every game has a result) that has not been ingested
  within ``GRACE`` of its last game: the pipeline is not running, or keeps failing;
* games whose date is ``GRACE`` past with no result in the schedule: nflverse itself is
  behind, and the weekly run will have nothing to ingest.

Out of season (no game within a week either side of today) the check passes.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl

from fourthdown.config import Paths, week_key
from fourthdown.data import ingest

GRACE = timedelta(days=2)
"""nflverse publishes overnight; Monday's game is in the data by Wednesday morning."""
SEASON_MARGIN = timedelta(days=7)


@dataclass
class FreshnessReport:
    season: int
    in_season: bool
    stale: list[str] = field(default_factory=list)
    latest_ingested: str | None = None

    @property
    def passed(self) -> bool:
        return not self.stale

    def summary(self) -> str:
        if not self.in_season:
            return f"{self.season}: out of season; nothing is due"
        if self.passed:
            return f"{self.season}: current through {self.latest_ingested or 'no week yet'}"
        return f"{self.season}: " + "; ".join(self.stale)


def check(paths: Paths, ingested: Collection[str], today: date) -> FreshnessReport:
    """Compare the schedule at ``paths.schedule`` with the ingested week keys."""
    season = ingest.current_season(today)
    games = (
        pl.read_parquet(paths.schedule, columns=["season", "week", "gameday", "result"])
        .filter(pl.col("season") == season)
        .with_columns(
            pl.col("week").cast(pl.Int64),
            pl.col("gameday").cast(pl.String).str.to_date(strict=False),
        )
    )
    ours = sorted(key for key in ingested if key.startswith(f"{season}-"))
    report = FreshnessReport(
        season=season, in_season=False, latest_ingested=ours[-1] if ours else None
    )
    if games.is_empty():
        return report
    first, last = games["gameday"].min(), games["gameday"].max()
    assert isinstance(first, date) and isinstance(last, date)
    report.in_season = first - SEASON_MARGIN <= today <= last + SEASON_MARGIN
    if not report.in_season:
        return report
    due = today - GRACE
    weeks = games.group_by("week").agg(
        pl.col("gameday").max().alias("last_game"),
        pl.col("result").is_not_null().all().alias("complete"),
        (pl.col("result").is_null() & (pl.col("gameday") <= due)).sum().alias("overdue"),
    )
    for week, last_game, complete, overdue in weeks.sort("week").iter_rows():
        key = week_key(season, int(week))
        if complete and last_game <= due and key not in ingested:
            report.stale.append(
                f"{key} ended {last_game} and is not ingested; is the weekly run failing?"
            )
        elif overdue:
            report.stale.append(
                f"{key}: {overdue} game(s) played by {due} still have no result in the nflverse "
                "schedule; nflverse is behind"
            )
    return report
