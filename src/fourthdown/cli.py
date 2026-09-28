"""Command line entry points for the data pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from fourthdown.config import data_paths
from fourthdown.data import audit, etl, ingest, warehouse
from fourthdown.evaluation import harness, retrieval_harness
from fourthdown.llm import LLMError, OllamaClient, default_client
from fourthdown.models import fourth_down
from fourthdown.models import train as training
from fourthdown.models.train import FOURTH_DOWN_ARTIFACT, WP_ARTIFACT
from fourthdown.models.winprob import GameState, WinProbabilityModel
from fourthdown.narrative.render import GRAINS
from fourthdown.rag import schema_card
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.retrieval import DocumentStore, Retriever, default_embedder, index, recall

app = typer.Typer(help="FourthDown data pipeline", no_args_is_help=True)

DEFAULT_SEASONS = "2009-"
DEFAULT_GRAINS = ",".join(GRAINS)


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
Grains = Annotated[str, typer.Option("--grains", help="Comma separated: game, drive.")]
TopK = Annotated[int, typer.Option("--k", help="Passages to retrieve.")]
Grain = Annotated[str | None, typer.Option("--grain", help="Restrict to 'game' or 'drive'.")]
Season = Annotated[int | None, typer.Option("--season", help="Restrict to one season.")]
Team = Annotated[str | None, typer.Option("--team", help="Restrict to a team abbreviation.")]


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
    for position, attempt in enumerate(answer.attempts, start=1):
        if attempt.error:
            typer.echo(f"attempt {position} rejected: {attempt.error}", err=True)
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


def _parse_grains(grains: str) -> list[str]:
    wanted = [grain.strip() for grain in grains.split(",") if grain.strip()]
    unknown = [grain for grain in wanted if grain not in GRAINS]
    if unknown:
        raise typer.BadParameter(f"unknown grain(s) {unknown}; expected any of {list(GRAINS)}")
    return wanted


@app.command("index")
def index_cmd(
    grains: Grains = DEFAULT_GRAINS,
    seasons: Seasons = DEFAULT_SEASONS,
    playoff_drives: Annotated[
        bool,
        typer.Option("--playoff-drives", help="Index postseason drives only (4% of them)."),
    ] = False,
    reset: Annotated[bool, typer.Option("--reset", help="Drop the index first.")] = False,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Generate game and drive narratives, embed them, and upsert them into pgvector."""
    _configure_logging(verbose)
    wanted = ingest.parse_seasons(seasons, latest=_latest_season())
    embedder = default_embedder()
    with warehouse.connect(data_paths(data_dir)) as connection, DocumentStore.open() as store:
        report = index.build(
            connection,
            store,
            embedder,
            grains=_parse_grains(grains),
            seasons=wanted,
            postseason_drives_only=playoff_drives,
            reset=reset,
        )
    typer.echo(report.render())


@app.command("recall")
def recall_cmd(
    question: Annotated[str, typer.Argument(help="A question about what happened.")],
    k: TopK = 5,
    grain: Grain = None,
    season: Season = None,
    team: Team = None,
    verbose: Verbose = False,
) -> None:
    """Show the passages hybrid retrieval returns, without asking the model anything."""
    _configure_logging(verbose)
    embedder = default_embedder()
    with DocumentStore.open() as store:
        hits = Retriever(store, embedder).search(
            question, k=k, grain=grain, season=season, team=team
        )
    if not hits:
        typer.echo("no passages matched; is the index built? (`fourthdown index`)")
        return
    for position, hit in enumerate(hits, start=1):
        typer.echo(f"[{position}] {hit.title} ({hit.found_by}, score {hit.score:.4f})")
        typer.echo(f"    {hit.body}\n")


@app.command("explain")
def explain_cmd(
    question: Annotated[str, typer.Argument(help="A question about what happened.")],
    k: TopK = 5,
    grain: Grain = None,
    season: Season = None,
    team: Team = None,
    model: Model = None,
    verbose: Verbose = False,
) -> None:
    """Answer a narrative question from retrieved passages, with citations."""
    _configure_logging(verbose)
    client = default_client() if model is None else OllamaClient(model)
    with DocumentStore.open() as store:
        retriever = Retriever(store, default_embedder())
        try:
            grounded = recall.answer(
                retriever, client, question, k=k, grain=grain, season=season, team=team
            )
        except LLMError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
    typer.echo(grounded.render())


@app.command("eval-retrieval")
def eval_retrieval_cmd(
    output: Output = Path("docs/retrieval_eval.md"),
    k: TopK = 5,
    verbose: Verbose = False,
) -> None:
    """Score the retrieval golden set, hybrid against each half on its own."""
    _configure_logging(verbose)
    with DocumentStore.open() as store:
        report = retrieval_harness.evaluate(store, default_embedder(), k=k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.render())
    typer.echo(
        f"hybrid hit@1 {report.hit_at_1():.2f}, recall@{k} {report.recall():.2f}, "
        f"MRR {report.mrr():.3f}, written to {output}"
    )


@app.command("train")
def train_cmd(
    output: Output = Path("docs/model_eval.md"),
    epochs: Annotated[int, typer.Option("--epochs", help="Cap on win-probability epochs.")] = 60,
    audit_sample: Annotated[
        int, typer.Option("--audit-sample", help="Held-out fourth downs to replay.")
    ] = 4000,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Fit win probability, the fourth-down components, and the play-call model."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    paths.models.mkdir(parents=True, exist_ok=True)
    with warehouse.connect(paths) as connection:
        results = training.run(
            connection,
            model_dir=paths.models,
            max_epochs=epochs,
            audit_sample=audit_sample,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(training.report(results))
    typer.echo(
        f"win probability log loss {results.win_probability.log_loss:.4f} "
        f"(ECE {results.win_probability.calibration_error:.4f}), "
        f"play call {results.play_call.accuracy:.1%} vs "
        f"{results.play_call_majority.accuracy:.1%} base rate; written to {output}"
    )


@app.command("advise")
def advise_cmd(
    yardline: Annotated[int, typer.Option("--yardline", help="Yards from the opponent's goal.")],
    togo: Annotated[float, typer.Option("--togo", help="Yards to go.")],
    minutes: Annotated[float, typer.Option("--minutes", help="Minutes left in the game.")] = 15.0,
    score_diff: Annotated[
        int, typer.Option("--score-diff", help="Offense score minus defense score.")
    ] = 0,
    spread: Annotated[
        float, typer.Option("--spread", help="Closing spread, signed for the offense.")
    ] = 0.0,
    away: Annotated[bool, typer.Option("--away", help="Offense is the visiting team.")] = False,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Rank go / field goal / punt for one fourth down."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    weights = paths.models / WP_ARTIFACT
    components = paths.models / FOURTH_DOWN_ARTIFACT
    if not weights.exists() or not components.exists():
        typer.echo(f"no trained models in {paths.models}; run 'fourthdown train' first", err=True)
        raise typer.Exit(code=1)
    advisor = fourth_down.load_components(components, WinProbabilityModel.load(weights))
    state = GameState(
        yardline_100=float(yardline),
        down=4,
        ydstogo=float(togo),
        game_seconds_remaining=minutes * 60.0,
        score_differential=float(score_diff),
        posteam_is_home=not away,
        posteam_spread=spread,
        goal_to_go=yardline <= togo,
    )
    typer.echo(advisor.recommend(state).render())


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
