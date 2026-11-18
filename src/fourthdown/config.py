"""Filesystem layout and dataset-wide constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FIRST_SEASON = 1999
"""Earliest season nflverse publishes. The project targets 2009+ (tracking-era columns)."""

DEFAULT_FIRST_SEASON = 2009
"""First season with air yards, EPA, and win probability populated."""


@dataclass(frozen=True)
class Paths:
    """Resolved locations of every dataset artifact the pipeline reads or writes."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw" / "pbp"

    @property
    def players(self) -> Path:
        return self.root / "raw" / "players.parquet"

    @property
    def processed(self) -> Path:
        return self.root / "processed" / "plays"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse"

    @property
    def database(self) -> Path:
        return self.warehouse / "fourthdown.duckdb"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def feedback(self) -> Path:
        """Questions whose query came back empty (`fourthdown.feedback`); the one place a
        serving container writes."""
        return self.root / "feedback"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def mlflow(self) -> Path:
        return self.root / "mlflow"

    def season_raw(self, season: int) -> Path:
        return self.raw / f"play_by_play_{season}.parquet"

    @property
    def schedule(self) -> Path:
        """nflverse's schedule: every game of every season, with results once played."""
        return self.root / "raw" / "schedules.parquet"

    @property
    def reports(self) -> Path:
        """What a pipeline run writes for people: model eval, scorecard, lineage."""
        return self.root / "reports"

    def season_processed(self, season: int) -> Path:
        """The directory holding a season's weekly partitions."""
        return self.processed / f"season={season}"

    def week_processed(self, season: int, week: int) -> Path:
        return self.season_processed(season) / f"week={week:02d}.parquet"

    def ensure(self) -> None:
        for directory in (self.raw, self.processed, self.warehouse, self.models):
            directory.mkdir(parents=True, exist_ok=True)


def week_key(season: int, week: int) -> str:
    """The partition key of one NFL week, e.g. ``2026-10``; sorts in calendar order."""
    return f"{season}-{week:02d}"


def parse_week_key(key: str) -> tuple[int, int]:
    """``2026-10`` -> ``(2026, 10)``."""
    season, _, week = key.partition("-")
    if not (season.isdigit() and week.isdigit()):
        raise ValueError(f"not a season-week key: {key!r}")
    return int(season), int(week)


def data_paths(root: str | Path | None = None) -> Paths:
    """Resolve the data root from an argument, ``FOURTHDOWN_DATA_DIR``, or the repo default."""
    if root is not None:
        return Paths(Path(root).expanduser().resolve())
    env = os.environ.get("FOURTHDOWN_DATA_DIR")
    if env:
        return Paths(Path(env).expanduser().resolve())
    return Paths(Path(__file__).resolve().parents[2] / "data")
