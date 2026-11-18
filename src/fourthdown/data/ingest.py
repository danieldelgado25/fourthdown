"""Download season play-by-play files from the nflverse data releases.

nflverse publishes one Parquet file per season (~20 MB, ~50k plays, 372 columns) as a
GitHub release asset, with no authentication and no rate limit worth worrying about.
Downloads are idempotent: a season already on disk is skipped unless ``force`` is set.

During a season nflverse republishes the current file in place, nightly. ``refresh``
asks for it conditionally (``If-None-Match`` with the recorded ETag), so checking for a
new week costs one request and no download when nothing changed. The schedule file says
which weeks are over: a week is complete when every one of its games has a result.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

import polars as pl
import requests

from fourthdown.config import DEFAULT_FIRST_SEASON, FIRST_SEASON, Paths, data_paths
from fourthdown.data import provenance

LOGGER = logging.getLogger(__name__)

RELEASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
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


def download_season(
    season: int, paths: Paths, *, force: bool = False, refresh: bool = False
) -> Path:
    """Fetch one season to ``paths.raw``, returning the local path.

    Args:
        force: download even when the file is on disk.
        refresh: ask nflverse whether its copy changed (a conditional request against the
            recorded ETag) and download only if it did. What the weekly pipeline uses.
    """
    destination = paths.season_raw(season)
    url = season_url(season)
    if destination.exists() and not (force or refresh):
        LOGGER.info("season %s already present at %s", season, destination)
        provenance.record_raw(paths, season, url=url, downloaded=False)
        return destination

    previous = provenance.load(paths).raw.get(season)
    conditional = (
        refresh
        and not force
        and destination.exists()
        and previous is not None
        and previous.etag is not None
        and previous.sha256 == provenance.sha256_file(destination)
    )
    etag = previous.etag if conditional and previous is not None else None
    LOGGER.info("%s season %s from %s", "checking" if etag else "downloading", season, url)
    fetched = _fetch(url, destination, etag=etag)
    if fetched is None:
        LOGGER.info("season %s unchanged since %s", season, previous and previous.retrieved_at)
        provenance.record_raw(paths, season, url=url, downloaded=False)
        return destination
    provenance.record_raw(
        paths, season, url=url, downloaded=True, etag=fetched[0], last_modified=fetched[1]
    )
    return destination


def _fetch(
    url: str, destination: Path, *, etag: str | None = None
) -> tuple[str | None, str | None] | None:
    """Stream ``url`` to ``destination`` atomically, returning its ETag and Last-Modified.

    With ``etag``, the request is conditional; None means the server's copy is unchanged
    and nothing was written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".parquet.part")
    headers = {"If-None-Match": etag} if etag else {}
    with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as response:
        if etag and response.status_code == 304:
            return None
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(CHUNK_SIZE):
                handle.write(chunk)
        new_etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
    partial.replace(destination)
    return new_etag, last_modified


def download_players(paths: Paths, *, force: bool = False) -> Path:
    """Fetch nflverse's player table, which maps the play-by-play ids to full names.

    The warehouse records its hash, so a later download shows up as drift.
    """
    destination = paths.players
    if destination.exists() and not force:
        LOGGER.info("players already present at %s", destination)
        return destination
    LOGGER.info("downloading players from %s", PLAYERS_URL)
    _fetch(PLAYERS_URL, destination)
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


def download_schedule(paths: Paths) -> Path:
    """Fetch nflverse's schedule (every game, with scores once final). Always refreshed:
    it is small, and it is how a run learns that a week is over."""
    LOGGER.info("downloading the schedule from %s", SCHEDULE_URL)
    _fetch(SCHEDULE_URL, paths.schedule)
    return paths.schedule


def _schedule(paths: Paths, season: int) -> pl.DataFrame:
    if not paths.schedule.exists():
        raise FileNotFoundError(f"no schedule at {paths.schedule}; run download_schedule first")
    return (
        pl.read_parquet(paths.schedule, columns=["game_id", "season", "week", "result"])
        .filter(pl.col("season") == season)
        .with_columns(pl.col("week").cast(pl.Int64))
    )


def completed_weeks(paths: Paths, season: int) -> list[int]:
    """Weeks of ``season`` whose every scheduled game has a final result."""
    weeks = (
        _schedule(paths, season)
        .group_by("week")
        .agg(pl.col("result").is_not_null().all().alias("complete"))
        .filter(pl.col("complete"))
    )
    return sorted(int(week) for week in weeks["week"])


def scheduled_games(paths: Paths, season: int, week: int) -> set[str]:
    """The game ids the schedule lists for one week."""
    games = _schedule(paths, season).filter(pl.col("week") == week)["game_id"]
    return {str(game) for game in games}


def season_games(paths: Paths, season: int, week: int) -> set[str]:
    """The game ids the raw play-by-play file holds for one week."""
    games = (
        pl.scan_parquet(paths.season_raw(season))
        .filter(pl.col("week") == week)
        .select(pl.col("game_id").unique())
        .collect()["game_id"]
    )
    return {str(game) for game in games}


def season_weeks(paths: Paths, season: int) -> list[int]:
    """The weeks the raw play-by-play file holds for a season; empty without the file."""
    source = paths.season_raw(season)
    if not source.exists():
        return []
    weeks = pl.scan_parquet(source).select(pl.col("week").unique()).collect()["week"]
    return sorted(int(week) for week in weeks)


def current_season(today: date | None = None) -> int:
    """Seasons are named by their September start, so the current year counts from March."""
    now = today or date.today()
    return now.year if now.month >= 3 else now.year - 1
