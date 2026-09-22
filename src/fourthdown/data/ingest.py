"""Download season play-by-play files from the nflverse data releases.

nflverse publishes one Parquet file per season (~20 MB, ~50k plays, 372 columns) as a
GitHub release asset, with no authentication and no rate limit worth worrying about.
Downloads are idempotent: a season already on disk is skipped unless ``force`` is set.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import requests

from fourthdown.config import DEFAULT_FIRST_SEASON, FIRST_SEASON, Paths, data_paths

LOGGER = logging.getLogger(__name__)

RELEASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)
CHUNK_SIZE = 1 << 20
TIMEOUT = 120


def season_url(season: int) -> str:
    return RELEASE_URL.format(season=season)


def parse_seasons(spec: str, *, latest: int | None = None) -> list[int]:
    """Expand a season spec such as ``"2009-2016,2023"`` into a sorted list of seasons.

    ``latest`` bounds open-ended ranges like ``"2009-"``; it defaults to no bound.
    """
    seasons: set[int] = set()
    for part in (piece.strip() for piece in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            start = int(start_text)
            if end_text:
                end = int(end_text)
            elif latest is not None:
                end = latest
            else:
                raise ValueError(f"open-ended range {part!r} needs a `latest` season")
            if end < start:
                raise ValueError(f"range {part!r} ends before it starts")
            seasons.update(range(start, end + 1))
        else:
            seasons.add(int(part))
    if not seasons:
        raise ValueError(f"no seasons parsed from {spec!r}")
    out_of_range = [season for season in seasons if season < FIRST_SEASON]
    if out_of_range:
        raise ValueError(f"nflverse starts at {FIRST_SEASON}; got {sorted(out_of_range)}")
    below_target = [season for season in seasons if season < DEFAULT_FIRST_SEASON]
    if below_target:
        LOGGER.warning(
            "seasons before %s lack air yards and win probability: %s",
            DEFAULT_FIRST_SEASON,
            sorted(below_target),
        )
    return sorted(seasons)


def download_season(season: int, paths: Paths, *, force: bool = False) -> Path:
    """Fetch one season to ``paths.raw``, returning the local path."""
    destination = paths.season_raw(season)
    if destination.exists() and not force:
        LOGGER.info("season %s already present at %s", season, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".parquet.part")
    url = season_url(season)
    LOGGER.info("downloading season %s from %s", season, url)
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(CHUNK_SIZE):
                handle.write(chunk)
    partial.replace(destination)
    return destination


def download_seasons(
    seasons: Iterable[int],
    paths: Paths | None = None,
    *,
    force: bool = False,
) -> Iterator[Path]:
    resolved = paths or data_paths()
    resolved.ensure()
    for season in seasons:
        yield download_season(season, resolved, force=force)
