"""Command line entry points for the data pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from fourthdown.config import data_paths
from fourthdown.data import audit, etl, ingest, warehouse

app = typer.Typer(help="FourthDown data pipeline", no_args_is_help=True)

DEFAULT_SEASONS = "2009-"


def _latest_season() -> int:
    """Seasons are named by their September start, so the current year counts from March."""
    now = datetime.now()
    return now.year if now.month >= 3 else now.year - 1


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


Seasons = Annotated[
    str, typer.Option("--seasons", "-s", help="e.g. '2009-2016,2023' or an open-ended '2009-'")
]
DataDir = Annotated[Path | None, typer.Option("--data-dir", help="Overrides FOURTHDOWN_DATA_DIR.")]
Verbose = Annotated[bool, typer.Option("--verbose", "-v")]
Output = Annotated[Path, typer.Option("--output", "-o")]
Force = Annotated[bool, typer.Option("--force", help="Re-download seasons already on disk.")]


@app.command("ingest")
def ingest_cmd(
    seasons: Seasons = DEFAULT_SEASONS,
    data_dir: DataDir = None,
    force: Force = False,
    verbose: Verbose = False,
) -> None:
    """Download season play-by-play files from nflverse."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    wanted = ingest.parse_seasons(seasons, latest=_latest_season())
    downloaded = list(ingest.download_seasons(wanted, paths, force=force))
    typer.echo(f"{len(downloaded)} season files available in {paths.raw}")


@app.command("etl")
def etl_cmd(
    seasons: Seasons = DEFAULT_SEASONS,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Transform raw seasons into the processed play table."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    wanted = ingest.parse_seasons(seasons, latest=_latest_season())
    written = list(etl.transform_seasons(wanted, paths))
    typer.echo(f"{len(written)} partitions written to {paths.processed}")


@app.command("warehouse")
def warehouse_cmd(
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Create the DuckDB semantic layer over the processed partitions."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    target = warehouse.build(paths)
    typer.echo(f"warehouse ready at {target} ({', '.join(warehouse.VIEW_NAMES)})")


@app.command("audit")
def audit_cmd(
    output: Output = Path("docs/data_audit.md"),
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Write the coverage and sanity-check report, failing if any check fails."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    with warehouse.connect(paths) as connection:
        audit.write_report(connection, output)
        failures = [check for check in audit.run_checks(connection) if not check.passed]
    typer.echo(f"audit written to {output}")
    for failure in failures:
        typer.echo(f"FAIL {failure.name}: {failure.detail}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def build(
    seasons: Seasons = DEFAULT_SEASONS,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Ingest, transform, and warehouse in one pass."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    wanted = ingest.parse_seasons(seasons, latest=_latest_season())
    list(ingest.download_seasons(wanted, paths))
    list(etl.transform_seasons(wanted, paths))
    target = warehouse.build(paths)
    typer.echo(f"built {len(wanted)} seasons into {target}")


if __name__ == "__main__":
    app()
