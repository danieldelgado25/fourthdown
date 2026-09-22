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
    def processed(self) -> Path:
        return self.root / "processed" / "plays"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse"

    @property
    def database(self) -> Path:
        return self.warehouse / "fourthdown.duckdb"

    def season_raw(self, season: int) -> Path:
        return self.raw / f"play_by_play_{season}.parquet"

    def season_processed(self, season: int) -> Path:
        return self.processed / f"season={season}" / "plays.parquet"

    def ensure(self) -> None:
        for directory in (self.raw, self.processed, self.warehouse):
            directory.mkdir(parents=True, exist_ok=True)


def data_paths(root: str | Path | None = None) -> Paths:
    """Resolve the data root from an argument, ``FOURTHDOWN_DATA_DIR``, or the repo default."""
    if root is not None:
        return Paths(Path(root).expanduser().resolve())
    env = os.environ.get("FOURTHDOWN_DATA_DIR")
    if env:
        return Paths(Path(env).expanduser().resolve())
    return Paths(Path(__file__).resolve().parents[2] / "data")
