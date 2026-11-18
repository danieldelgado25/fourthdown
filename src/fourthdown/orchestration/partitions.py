"""NFL weeks as Dagster partitions.

One partition per week that has been played, keyed ``<season>-<week>`` (``2026-10``),
the same key the provenance manifest and the processed file layout use. The set is
dynamic rather than a static season x week grid because seasons differ in length (21
weeks through 2020 including the playoffs, 22 since) and a week only exists once it has
been played: the weekly schedule and the bootstrap register weeks as the nflverse
schedule shows them complete.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from dagster import DagsterInstance, DynamicPartitionsDefinition

from fourthdown.config import Paths, parse_week_key, week_key
from fourthdown.data import ingest

WEEKS_NAME = "nfl_weeks"
WEEKS = DynamicPartitionsDefinition(name=WEEKS_NAME)


def completed(paths: Paths, seasons: Iterable[int]) -> list[str]:
    """Keys of every week of ``seasons`` whose games all have results, in calendar order.

    Reads the schedule at ``paths.schedule``; download it first.
    """
    return [
        week_key(season, week)
        for season in sorted(seasons)
        for week in ingest.completed_weeks(paths, season)
    ]


def register(instance: DagsterInstance, keys: Sequence[str]) -> list[str]:
    """Add the keys not yet registered, in calendar order; returns the ones added."""
    known = set(instance.get_dynamic_partitions(WEEKS_NAME))
    added = sorted(key for key in set(keys) if key not in known)
    if added:
        instance.add_dynamic_partitions(WEEKS_NAME, added)
    return added


def by_season(keys: Iterable[str]) -> dict[int, list[int]]:
    """``["2026-09", "2026-10"]`` -> ``{2026: [9, 10]}``."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for key in keys:
        season, week = parse_week_key(key)
        grouped[season].append(week)
    return {season: sorted(weeks) for season, weeks in sorted(grouped.items())}
