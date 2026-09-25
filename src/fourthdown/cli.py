"""Command line entry points for the data pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from fourthdown.config import data_paths
from fourthdown.data import audit, etl, ingest, warehouse
from fourthdown.evaluation import harness
from fourthdown.llm import LLMError, OllamaClient, default_client
from fourthdown.rag import schema_card
from fourthdown.rag.text_to_sql import TextToSQL

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
Model = Annotated[str | None, typer.Option("--model", help="Ollama model; overrides the default.")]


def _render_rows(columns: tuple[str, ...], rows: list[tuple[object, ...]], limit: int = 20) -> str:
    header = " | ".join(columns)
    body = [" | ".join("NULL" if value is None else str(value) for value in row) for row in rows]
    shown = body[:limit]
    if len(body) > limit:
        shown.append(f"... {len(body) - limit} more rows")
    return "\n".join([header, "-" * len(header), *shown])


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


@app.command("schema-card")
def schema_card_cmd(
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Print the prompt the SQL model sees."""
    _configure_logging(verbose)
    with warehouse.connect(data_paths(data_dir)) as connection:
        typer.echo(schema_card.build(connection))


@app.command("ask")
def ask_cmd(
    question: Annotated[str, typer.Argument(help="A question about the play-by-play data.")],
    model: Model = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Answer a question by generating, validating, and running DuckDB SQL."""
    _configure_logging(verbose)
    client = default_client() if model is None else OllamaClient(model)
    with warehouse.connect(data_paths(data_dir)) as connection:
        chain = TextToSQL(connection, client)
        try:
            answer = chain.answer(question)
        except LLMError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
    for index, attempt in enumerate(answer.attempts, start=1):
        if attempt.error:
            typer.echo(f"attempt {index} rejected: {attempt.error}", err=True)
    if answer.unanswerable:
        typer.echo("The warehouse does not contain the data needed to answer that.")
        return
    if not answer.ok:
        typer.echo(f"failed after {len(answer.attempts)} attempts: {answer.error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{answer.sql}\n")
    typer.echo(_render_rows(answer.columns, answer.rows))


@app.command("eval")
def eval_cmd(
    output: Output = Path("docs/text_to_sql_eval.md"),
    model: Model = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Run the golden question set and write the execution-accuracy report."""
    _configure_logging(verbose)
    client = default_client() if model is None else OllamaClient(model)
    with warehouse.connect(data_paths(data_dir)) as connection:
        chain = TextToSQL(connection, client)
        report = harness.evaluate(connection, chain)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.render())
    typer.echo(
        f"{report.correct}/{report.total} correct, "
        f"{report.executed}/{report.total} runnable, written to {output}"
    )


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
