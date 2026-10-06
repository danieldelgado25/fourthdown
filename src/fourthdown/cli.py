"""Command line entry points for the data pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from fourthdown.agent import build as agent_build
from fourthdown.agent.router import KeywordRouter, LLMRouter
from fourthdown.api import create_app
from fourthdown.config import data_paths
from fourthdown.data import audit, etl, ingest, provenance, warehouse
from fourthdown.evaluation import harness, retrieval_harness, routing_harness, scorecard, suites
from fourthdown.llm import LLMError, OllamaClient, default_client
from fourthdown.models import fourth_down, tracking, winprob
from fourthdown.models import train as training
from fourthdown.models.card import MODEL_CARD, ModelCard
from fourthdown.models.features import DEFAULT_SPLIT, SeasonSplit
from fourthdown.models.train import FOURTH_DOWN_ARTIFACT, WP_ARTIFACT
from fourthdown.models.winprob import GameState, WinProbabilityModel
from fourthdown.narrative.render import GRAINS
from fourthdown.rag import schema_card
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.retrieval import DocumentStore, Retriever, default_embedder, index, recall
from fourthdown.serving import wait as readiness
from fourthdown.serving import winprob as wp_service

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
    holdout: Annotated[
        bool, typer.Option("--holdout", help="Score the held-out set instead of the dev set.")
    ] = False,
    verbose: Verbose = False,
) -> None:
    """Run a golden question set and write the execution-accuracy report."""
    _configure_logging(verbose)
    client = default_client() if model is None else OllamaClient(model)
    with warehouse.connect(data_paths(data_dir)) as connection:
        chain = TextToSQL(connection, client)
        report = harness.evaluate(connection, chain, harness.load_questions(holdout=holdout))
    output.parent.mkdir(parents=True, exist_ok=True)
    title = "Text-to-SQL evaluation (held-out set)" if holdout else "Text-to-SQL evaluation"
    output.write_text(report.render(title))
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


SplitSeasons = Annotated[
    str | None, typer.Option(help="Season spec; give all three splits or none.")
]


def _split(train: str | None, valid: str | None, test: str | None) -> SeasonSplit:
    given = [spec for spec in (train, valid, test) if spec is not None]
    if not given:
        return DEFAULT_SPLIT
    if len(given) != 3:
        raise typer.BadParameter(
            "give --train-seasons, --valid-seasons and --test-seasons together"
        )
    assert train is not None and valid is not None and test is not None
    try:
        return SeasonSplit(
            train=tuple(ingest.parse_seasons(train)),
            valid=tuple(ingest.parse_seasons(valid)),
            test=tuple(ingest.parse_seasons(test)),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _names(spec: str) -> list[str]:
    return [name.strip() for name in spec.split(",") if name.strip()]


@app.command("scorecard")
def scorecard_cmd(
    suite_names: Annotated[
        str, typer.Option("--suites", help=f"Comma separated, from: {', '.join(suites.ALL)}.")
    ] = ",".join(suites.ALL),
    require: Annotated[
        str, typer.Option("--require", help="Suites that must run; a skip fails the scorecard.")
    ] = "",
    train_seasons: SplitSeasons = None,
    valid_seasons: SplitSeasons = None,
    test_seasons: SplitSeasons = None,
    epochs: Annotated[
        int, typer.Option("--epochs", help="Cap on win-probability epochs.")
    ] = winprob.MAX_EPOCHS,
    audit_sample: Annotated[
        int, typer.Option("--audit-sample", help="Held-out fourth downs to replay.")
    ] = 4000,
    k: TopK = retrieval_harness.DEFAULT_K,
    model: Model = None,
    output: Output = Path("docs/scorecard.md"),
    json_output: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Run the evaluation suites and fail if any gated metric is out of bounds."""
    _configure_logging(verbose)
    selected, required = _names(suite_names), _names(require)
    unknown = sorted(set(selected + required) - set(suites.ALL))
    if unknown:
        raise typer.BadParameter(f"unknown suite(s) {', '.join(unknown)}")
    options = suites.RunOptions(
        paths=data_paths(data_dir),
        client=default_client() if model is None else OllamaClient(model),
        embedder=default_embedder(),
        split=_split(train_seasons, valid_seasons, test_seasons),
        epochs=epochs,
        audit_sample=audit_sample,
        k=k,
    )
    card = scorecard.build(suites.run(selected, options), required=required)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(card.render())
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(card.to_json() + "\n")
    for suite in card.suites:
        if suite.skipped:
            typer.echo(f"{suite.name}: skipped ({suite.skipped})", err=True)
    for verdict in card.failures():
        typer.echo(
            f"FAIL {verdict.gate.key} = {verdict.value} (want {verdict.gate.bound()}): "
            f"{verdict.gate.why}",
            err=True,
        )
    for name in card.missing_required():
        typer.echo(f"FAIL required suite {name} did not run", err=True)
    typer.echo(f"scorecard {'passed' if card.passed else 'failed'}; written to {output}")
    if not card.passed:
        raise typer.Exit(code=1)


@app.command("lineage")
def lineage_cmd(
    output: Output = Path("docs/lineage.md"),
    verify: Annotated[
        bool, typer.Option("--verify", help="Rehash every file and fail on any drift.")
    ] = False,
    data_dir: DataDir = None,
) -> None:
    """Write the data lineage report from the provenance manifest."""
    paths = data_paths(data_dir)
    manifest = provenance.load(paths)
    if verify:
        problems = provenance.verify(paths)
        for problem in problems:
            typer.echo(f"DRIFT {problem}", err=True)
        if problems:
            raise typer.Exit(code=1)
        typer.echo(f"{len(manifest.processed)} partitions match the manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(provenance.render(manifest))
    typer.echo(f"data version {manifest.data_version}; lineage written to {output}")


@app.command("train")
def train_cmd(
    output: Output = Path("docs/model_eval.md"),
    epochs: Annotated[int, typer.Option("--epochs", help="Cap on win-probability epochs.")] = 60,
    audit_sample: Annotated[
        int, typer.Option("--audit-sample", help="Held-out fourth downs to replay.")
    ] = 4000,
    track: Annotated[
        bool, typer.Option("--track/--no-track", help="Log the run to MLflow.")
    ] = True,
    train_seasons: SplitSeasons = None,
    valid_seasons: SplitSeasons = None,
    test_seasons: SplitSeasons = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Fit win probability, the fourth-down components, and the play-call model."""
    _configure_logging(verbose)
    split = _split(train_seasons, valid_seasons, test_seasons)
    paths = data_paths(data_dir)
    problems = provenance.verify(paths)
    if problems:
        for problem in problems:
            typer.echo(f"DRIFT {problem}", err=True)
        typer.echo("data does not match data/manifest.json; rerun `fourthdown build`", err=True)
        raise typer.Exit(code=1)
    manifest = provenance.load(paths)
    paths.models.mkdir(parents=True, exist_ok=True)
    with warehouse.connect(paths) as connection:
        results = training.run(
            connection,
            model_dir=paths.models,
            max_epochs=epochs,
            audit_sample=audit_sample,
            split=split,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(training.report(results))
    card = ModelCard.build(
        artifact=paths.models / WP_ARTIFACT,
        manifest=manifest,
        params=tracking.training_params(
            SeasonSplit(
                train=results.wp_dataset.train.seasons,
                valid=results.wp_dataset.valid.seasons,
                test=results.wp_dataset.test.seasons,
            ),
            epochs=epochs,
            audit_sample=audit_sample,
        ),
        metrics=training.summary_metrics(results),
    )
    card_path = paths.models / MODEL_CARD
    if track:
        card = tracking.log_run(
            card,
            paths=paths,
            card_path=card_path,
            manifest=manifest,
            history=results.history,
            artifacts=[
                paths.models / name
                for name in (
                    WP_ARTIFACT,
                    training.HISTORY_ARTIFACT,
                    training.PLAYCALL_ARTIFACT,
                    FOURTH_DOWN_ARTIFACT,
                )
            ]
            + [output],
        )
        typer.echo(f"MLflow run {card.mlflow_run_id} at {card.mlflow_tracking_uri}")
    else:
        card.save(card_path)
    typer.echo(f"data version {card.data_version}; model card at {card_path}")
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


@app.command("chat")
def chat_cmd(
    question: Annotated[str, typer.Argument(help="Any question; the router picks the tool.")],
    tool: Annotated[
        str | None, typer.Option("--tool", help="Force a tool instead of routing.")
    ] = None,
    model: Model = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Route one question to SQL, retrieval, the advisor, or tendencies, and answer it."""
    _configure_logging(verbose)
    with agent_build.services(paths=data_paths(data_dir), model=model) as services:
        if services.readiness.degraded:
            for name, reason in services.readiness.skipped.items():
                typer.echo(f"unavailable: {name} ({reason})", err=True)
        try:
            answer = services.assistant.ask(question, tool=tool)
        except KeyError:
            typer.echo(
                f"unknown tool {tool!r}; loaded: {', '.join(services.assistant.tool_names)}",
                err=True,
            )
            raise typer.Exit(code=1) from None
        typer.echo(f"[{answer.tool}] {answer.route_reason} ({answer.decided_by})\n")
        typer.echo(answer.answer)
        if answer.table is not None:
            typer.echo("")
            typer.echo(
                _render_rows(answer.table.columns, [tuple(row) for row in answer.table.rows])
            )
        for position, passage in enumerate(answer.passages, start=1):
            typer.echo(f"\n[{position}] {passage.title} ({passage.found_by})")
        if "sql" in answer.detail:
            typer.echo(f"\nSQL:\n{answer.detail['sql']}")


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    model: Model = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Run the Flask API in front of the assistant."""
    _configure_logging(verbose)
    with agent_build.services(paths=data_paths(data_dir), model=model) as services:
        for name, reason in services.readiness.skipped.items():
            typer.echo(f"unavailable: {name} ({reason})", err=True)
        typer.echo(f"tools: {', '.join(services.readiness.tools)}")
        create_app(provided=services).run(host=host, port=port, threaded=False)


@app.command("serve-wp")
def serve_wp_cmd(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8080,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Run the standalone win-probability service (what the container runs)."""
    _configure_logging(verbose)
    wp_service.create_app(data_paths(data_dir).models).run(host=host, port=port)


@app.command("wait")
def wait_cmd(
    data: Annotated[
        bool, typer.Option("--data", help="The warehouse is built and the models trained.")
    ] = False,
    ollama_models: Annotated[
        str, typer.Option("--ollama-models", help="Comma separated models Ollama must have.")
    ] = "",
    postgres: Annotated[
        bool, typer.Option("--postgres", help="Postgres accepts connections.")
    ] = False,
    min_documents: Annotated[
        int, typer.Option("--min-documents", help="The narrative index holds at least this many.")
    ] = 0,
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds before giving up.")] = 1800.0,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between polls.")] = 5.0,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Block until dependencies are ready; what Kubernetes init containers run."""
    _configure_logging(verbose)
    checks: dict[str, readiness.Check] = {}
    if data:
        checks["data"] = readiness.data_check(data_paths(data_dir))
    for model in _names(ollama_models):
        checks[f"ollama:{model}"] = readiness.ollama_check(model)
    if postgres or min_documents > 0:
        checks["postgres"] = readiness.postgres_check(min_documents=min_documents)
    if not checks:
        raise typer.BadParameter("nothing to wait for; pass at least one check")
    pending = readiness.wait_for(checks, timeout=timeout, interval=interval)
    for name, reason in pending.items():
        typer.echo(f"not ready: {name} ({reason})", err=True)
    if pending:
        raise typer.Exit(code=1)
    typer.echo(f"ready: {', '.join(checks)}")


@app.command("eval-routing")
def eval_routing_cmd(
    output: Output = Path("docs/routing_eval.md"),
    model: Model = None,
    verbose: Verbose = False,
) -> None:
    """Score keyword and LLM routing on the golden routing set."""
    _configure_logging(verbose)
    tools = routing_harness.catalog()
    client = default_client() if model is None else OllamaClient(model)
    reports = [routing_harness.evaluate(KeywordRouter(tools), name="keyword")]
    if client.available():
        reports.append(
            routing_harness.evaluate(LLMRouter(tools, client), name=f"llm ({client.model})")
        )
    else:
        typer.echo(f"skipping the LLM router: no {client.model} at {client.host}", err=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(routing_harness.render(reports))
    for report in reports:
        typer.echo(f"{report.name}: {report.correct}/{report.total} ({report.accuracy():.0%})")
    typer.echo(f"written to {output}")


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
