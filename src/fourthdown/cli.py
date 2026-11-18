"""Command line entry points for the data pipeline."""

from __future__ import annotations

import importlib
import logging
import os
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from fourthdown import feedback
from fourthdown.agent.assistant import Assistant
from fourthdown.agent.router import HybridRouter, KeywordRouter, LLMRouter
from fourthdown.agent.tools import Tool
from fourthdown.config import Paths, data_paths
from fourthdown.data import audit, etl, ingest, modeling, provenance, warehouse
from fourthdown.evaluation import harness, retrieval_harness, routing_harness, scorecard, suites
from fourthdown.llm import LLMError, default_client
from fourthdown.models import fourth_down, registry, release, tracking, winprob
from fourthdown.models.card import MODEL_CARD
from fourthdown.models.features import DEFAULT_SPLIT, SeasonSplit
from fourthdown.models.train import FOURTH_DOWN_ARTIFACT, WP_ARTIFACT
from fourthdown.models.winprob import GameState, WinProbabilityModel
from fourthdown.narrative.render import GRAINS
from fourthdown.quality import DataQualityError
from fourthdown.rag import schema_card
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.retrieval import DocumentStore, Retriever, default_embedder, index, recall
from fourthdown.services import models as models_service
from fourthdown.services import retrieval as retrieval_service
from fourthdown.services import warehouse as warehouse_service
from fourthdown.serving import wait as readiness
from fourthdown.serving import winprob as wp_service
from fourthdown.streaming.scorer import Scorer

app = typer.Typer(help="FourthDown data pipeline", no_args_is_help=True)

DEFAULT_SEASONS = "2009-"
DEFAULT_GRAINS = ",".join(GRAINS)


def _latest_season() -> int:
    return ingest.current_season()


def _gate(
    metric: str, value: float, minimum: float | None, *, higher_is_better: bool = True
) -> bool:
    """Compare one evaluation metric against its threshold, for CI.

    Args:
        metric: Name printed in the verdict.
        value: The measured value.
        minimum: The threshold; ``None`` disables the gate.
        higher_is_better: False for losses, where the threshold is a maximum.

    Returns:
        True when the gate passes or is disabled. Failures are reported on stderr; callers
        exit non-zero after writing their report, so the report is always produced.
    """
    if minimum is None:
        return True
    passed = value >= minimum if higher_is_better else value <= minimum
    bound = ">=" if higher_is_better else "<="
    if not passed:
        typer.echo(f"GATE FAILED: {metric} {value:.4f}, required {bound} {minimum}", err=True)
    return passed


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
Model = Annotated[
    str | None, typer.Option("--model", help="LLM model id; overrides FOURTHDOWN_LLM_MODEL.")
]
Grains = Annotated[str, typer.Option("--grains", help="Comma separated: game, drive.")]
TopK = Annotated[int, typer.Option("--k", help="Passages to retrieve.")]
Grain = Annotated[str | None, typer.Option("--grain", help="Restrict to 'game' or 'drive'.")]
Season = Annotated[int | None, typer.Option("--season", help="Restrict to one season.")]
Threshold = Annotated[
    float | None, typer.Option(help="Fail (exit 1) below this score; for CI gates.")
]
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
    ingest.download_players(paths, force=force)
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
    written = _transform(wanted, paths)
    typer.echo(f"{len(written)} partitions written to {paths.processed}")


def _transform(seasons: list[int], paths: Paths) -> list[Path]:
    """`etl.transform_seasons`, turning a failed data contract into a clean exit."""
    try:
        return list(etl.transform_seasons(seasons, paths))
    except DataQualityError as error:
        typer.echo(f"{error}\nThat season's partitions were left as they were.", err=True)
        raise typer.Exit(code=1) from error


@app.command("warehouse")
def warehouse_cmd(
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Build the warehouse with dbt (models and data tests) over the processed partitions."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    target = _build_warehouse(paths)
    typer.echo(f"warehouse ready at {target} ({', '.join(warehouse.VIEW_NAMES)})")


def _build_warehouse(paths: Paths) -> Path:
    """`warehouse.build`, turning a failed model or data test into a clean exit."""
    try:
        return warehouse.build(paths)
    except modeling.DbtError as error:
        typer.echo(f"{error}\nThe live warehouse was left as it was.", err=True)
        raise typer.Exit(code=1) from error


@app.command(
    "dbt",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def dbt_cmd(ctx: typer.Context, data_dir: DataDir = None) -> None:
    """Run any dbt command against this data root, e.g. `fourthdown dbt docs generate`.

    dbt writes to the live warehouse file, so stop the warehouse service before `dbt run`;
    `docs generate`, `ls`, and `compile` only read it. Artifacts (the docs site, the
    manifest) land in data/warehouse/dbt/target.
    """
    if not ctx.args:
        typer.echo("usage: fourthdown dbt <dbt command> [options]", err=True)
        raise typer.Exit(code=2)
    # `--target prod` (Snowflake) keeps its artifacts apart from the DuckDB build's manifest.
    args = list(ctx.args)
    target = next(
        (args[i + 1] for i, arg in enumerate(args[:-1]) if arg in {"--target", "-t"}),
        next((arg.split("=", 1)[1] for arg in args if arg.startswith("--target=")), None),
    )
    result = modeling.invoke(data_paths(data_dir), args, target=target or modeling.DEFAULT_TARGET)
    raise typer.Exit(code=0 if result.success else 1)


@app.command("audit")
def audit_cmd(
    output: Output = Path("docs/data_audit.md"),
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Write the coverage and sanity-check report, failing if any check fails."""
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    tests = modeling.load_test_results(paths)
    with warehouse.connect(paths) as connection:
        audit.write_report(connection, tests, output)
    typer.echo(f"audit written to {output}")
    if tests is None:
        typer.echo("no data test results; rebuild with `fourthdown warehouse`", err=True)
        raise typer.Exit(code=1)
    failures = [test for test in tests if not test.passed]
    for failure in failures:
        typer.echo(f"FAIL {failure.name}: {failure.message}", err=True)
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
    client = default_client(model)
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
    min_correct: Annotated[
        int | None, typer.Option(help="Fail (exit 1) with fewer correct answers.")
    ] = None,
    model: Model = None,
    data_dir: DataDir = None,
    holdout: Annotated[
        bool, typer.Option("--holdout", help="Score the held-out set instead of the dev set.")
    ] = False,
    verbose: Verbose = False,
) -> None:
    """Run a golden question set and write the execution-accuracy report."""
    _configure_logging(verbose)
    client = default_client(model)
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
    if not _gate("text-to-SQL correct", report.correct, min_correct):
        raise typer.Exit(code=1)


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
    missing_only: Annotated[
        bool,
        typer.Option(
            "--missing-only",
            help="Skip seasons that already have documents, so widening the range only "
            "embeds the new ones.",
        ),
    ] = False,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Generate game and drive narratives, embed them, and upsert them into pgvector."""
    _configure_logging(verbose)
    wanted = ingest.parse_seasons(seasons, latest=_latest_season())
    embedder = default_embedder()
    with warehouse.connect(data_paths(data_dir)) as connection, DocumentStore.open() as store:
        if missing_only and not reset:
            indexed = store.seasons()
            wanted = [season for season in wanted if season not in indexed]
            if not wanted:
                typer.echo("every requested season is already indexed")
                return
            typer.echo(f"indexing {len(wanted)} season(s) not yet in the index: {wanted}")
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
    client = default_client(model)
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
    min_hit_at_1: Threshold = None,
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
    if not _gate("hybrid hit@1", report.hit_at_1(), min_hit_at_1):
        raise typer.Exit(code=1)


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
        client=default_client(model),
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


@app.command("quality")
def quality_cmd(
    seasons: Seasons = DEFAULT_SEASONS,
    freshness: Annotated[
        bool,
        typer.Option(
            "--freshness/--no-freshness",
            help="Also check the current season against the nflverse schedule (downloads it).",
        ),
    ] = True,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Run every data contract against the data on disk, and fail if any is broken.

    The processed contract on each weekly partition, the row-count bands, the leakage
    contract on the warehouse, and (in season) freshness against the schedule.
    """
    import polars as pl

    from fourthdown.quality import contracts, leakage, volume
    from fourthdown.quality import freshness as fresh

    _configure_logging(verbose)
    paths = data_paths(data_dir)
    wanted = set(ingest.parse_seasons(seasons, latest=_latest_season()))
    manifest = provenance.load(paths)
    problems: list[str] = []
    checked = 0
    for key, record in sorted(manifest.processed.items()):
        if record.season not in wanted:
            continue
        frame = pl.read_parquet(paths.week_processed(record.season, record.week))
        try:
            contracts.validate_processed(frame, season=record.season, week=record.week)
        except DataQualityError as error:
            problems += [f"{key}: {problem}" for problem in error.problems]
        problems += volume.check_week(frame, key)
        checked += 1
    typer.echo(
        f"{checked} weekly partitions checked against the processed contract and volume bands"
    )
    with warehouse.connect(paths) as connection:
        report = leakage.check(connection)
    problems += report.violations
    typer.echo(f"leakage contract: {'passed' if report.passed else 'BROKEN'}")
    if verbose:
        typer.echo(report.render())
    if freshness:
        ingest.download_schedule(paths)
        status = fresh.check(paths, manifest.processed, date.today())
        typer.echo(f"freshness: {status.summary()}")
        problems += status.stale
    for problem in problems:
        typer.echo(f"FAIL {problem}", err=True)
    if problems:
        raise typer.Exit(code=1)
    typer.echo("every data contract holds")


snowflake_app = typer.Typer(
    help="Build the same dbt project in Snowflake (phase 11); DuckDB stays the default.",
    no_args_is_help=True,
)
app.add_typer(snowflake_app, name="snowflake")


@snowflake_app.command("build")
def snowflake_build_cmd(
    full_refresh: Annotated[
        bool, typer.Option("--full-refresh", help="Start from an empty database.")
    ] = False,
    run_tests: Annotated[bool, typer.Option("--tests/--no-tests")] = True,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Load new or changed weeks into Snowflake and dbt build there, in a clone of the live
    database that is swapped in only if every data test passed."""
    from fourthdown.data import snowflake

    _configure_logging(verbose)
    settings = snowflake.Settings.from_env()
    if not settings.configured:
        typer.echo("Snowflake is not configured; set " + ", ".join(settings.missing()), err=True)
        raise typer.Exit(code=2)
    try:
        result = snowflake.build(
            data_paths(data_dir), settings, full_refresh=full_refresh, run_tests=run_tests
        )
    except (snowflake.SnowflakeError, modeling.DbtError) as error:
        typer.echo(f"{error}\nThe live Snowflake database was left as it was.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(result.summary())


@snowflake_app.command("parity")
def snowflake_parity_cmd(
    output: Annotated[
        Path | None, typer.Option("--output", help="Also write the table as Markdown here.")
    ] = None,
    data_dir: DataDir = None,
) -> None:
    """Compare the Snowflake build with the DuckDB one, measure by measure."""
    from fourthdown.data import snowflake

    rows = snowflake.compare(data_paths(data_dir))
    table = snowflake.render_parity(rows)
    typer.echo(table)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n")
    differing = [row for row in rows if not row.matches]
    if differing:
        typer.echo(f"{len(differing)} of {len(rows)} measures differ", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"all {len(rows)} measures match")


@snowflake_app.command("check")
def snowflake_check_cmd(data_dir: DataDir = None) -> None:
    """Compile every model and test for Snowflake without an account, and check the SQL."""
    from fourthdown.data import snowflake

    paths = data_paths(data_dir)
    problems: list[str] = []
    for weeks in (None, ["2026-01"]):
        compiled, found = snowflake.compile_offline(paths, weeks=weeks)
        problems += found
    for problem in problems:
        typer.echo(f"FAIL {problem}", err=True)
    if problems:
        raise typer.Exit(code=1)
    count = len(list(compiled.rglob("*.sql")))
    typer.echo(f"{count} models and tests compile for Snowflake (full and incremental)")


stream_app = typer.Typer(
    help="Replay a game through Redpanda into a live win-probability feed (phase 12).",
    no_args_is_help=True,
)
app.add_typer(stream_app, name="stream")

DEMO_GAME = "2023_22_SF_KC"
"""Super Bowl LVIII: 25-22 Kansas City in overtime, the most volatile recent final."""
Broker = Annotated[
    str,
    typer.Option(
        "--broker",
        envvar="FOURTHDOWN_KAFKA_BOOTSTRAP",
        help="Redpanda/Kafka bootstrap servers.",
    ),
]
Speed = Annotated[
    float,
    typer.Option("--speed", help="Game seconds per real second; 0 replays without pauses."),
]


def _scorer(paths: Paths) -> Scorer:
    from fourthdown.serving import winprob as served

    loaded = served.load(paths.models)
    version = loaded.card.registered_version
    return Scorer(
        probability=loaded.model.probability,
        model_version=f"v{version}" if version else loaded.card.data_version,
    )


@stream_app.command("demo")
def stream_demo_cmd(
    game: Annotated[str, typer.Argument(help="A game_id, e.g. 2023_22_SF_KC.")] = DEMO_GAME,
    speed: Speed = 0.0,
    data_dir: DataDir = None,
) -> None:
    """The whole path in one process, on an in-memory broker: replay, score, watch."""
    from fourthdown.streaming import MemoryBroker, feed, replay, scorer
    from fourthdown.streaming.events import PLAYS_TOPIC, WP_TOPIC

    paths = data_paths(data_dir)
    broker = MemoryBroker()
    with warehouse.connect(paths) as connection:
        events = replay.game_events(connection, game)
    model = _scorer(paths)
    replay.replay(broker.producer(), events, speed=speed)
    stats = scorer.run(
        broker.consumer([PLAYS_TOPIC], scorer.GROUP), broker.producer(), model, idle_seconds=0
    )
    live = feed.watch(broker.consumer([WP_TOPIC], feed.GROUP), feed.Feed(), emit=typer.echo)
    typer.echo("")
    typer.echo(feed.summarize(live, game).render())
    typer.echo(f"{stats.scored} events scored, {stats.dead_lettered} dead-lettered")


@stream_app.command("replay")
def stream_replay_cmd(
    game: Annotated[str, typer.Argument(help="A game_id, e.g. 2023_22_SF_KC.")] = DEMO_GAME,
    speed: Speed = 60.0,
    broker: Broker = "localhost:19092",
    data_dir: DataDir = None,
) -> None:
    """Publish a game's plays to nfl.plays, paced by the game clock."""
    from fourthdown.streaming import KafkaBroker, replay

    with warehouse.connect(data_paths(data_dir)) as connection:
        events = replay.game_events(connection, game)
    count = replay.replay(KafkaBroker(broker).producer(), events, speed=speed)
    typer.echo(f"published {count} events of {game} to nfl.plays")


@stream_app.command("score")
def stream_score_cmd(
    broker: Broker = "localhost:19092",
    idle: Annotated[
        float | None, typer.Option("--idle", help="Stop after this many idle seconds.")
    ] = None,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Consume nfl.plays, score each play, publish to nfl.win_probability (runs until stopped)."""
    from fourthdown.streaming import KafkaBroker, scorer
    from fourthdown.streaming.events import PLAYS_TOPIC

    _configure_logging(verbose)
    kafka = KafkaBroker(broker)
    stats = scorer.run(
        kafka.consumer([PLAYS_TOPIC], scorer.GROUP),
        kafka.producer(),
        _scorer(data_paths(data_dir)),
        idle_seconds=idle,
    )
    typer.echo(f"{stats.scored} events scored, {stats.dead_lettered} dead-lettered")


@stream_app.command("watch")
def stream_watch_cmd(
    game: Annotated[
        str | None, typer.Option("--game", help="Summarize this game when it ends.")
    ] = None,
    broker: Broker = "localhost:19092",
) -> None:
    """Print nfl.win_probability as it arrives, until a game ends."""
    import uuid

    from fourthdown.streaming import KafkaBroker, feed
    from fourthdown.streaming.events import WP_TOPIC

    # A fresh group each time, so every watcher sees the whole game from the start.
    consumer = KafkaBroker(broker).consumer([WP_TOPIC], f"{feed.GROUP}-{uuid.uuid4().hex[:8]}")
    live = feed.watch(consumer, feed.Feed(), emit=typer.echo)
    for game_id in [game] if game else list(live.games):
        typer.echo("")
        typer.echo(feed.summarize(live, game_id).render())


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
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(provenance.render(manifest))
    except OSError as error:
        # The default lands in docs/, which an image's working directory may not allow;
        # a passed --verify is still a pass, and the report is the only thing missing.
        typer.echo(f"could not write the report to {output} ({error}); pass --output", err=True)
        raise typer.Exit(code=0 if verify else 1) from error
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
    max_wp_log_loss: Annotated[
        float | None,
        typer.Option(help="Fail (exit 1) if held-out win-probability log loss exceeds this."),
    ] = None,
    min_play_call_accuracy: Threshold = None,
    promote: Annotated[
        str,
        typer.Option(
            help="Serve the new models: 'gated' (only if the models gates pass), 'never', or "
            "'always' (CI smoke runs, whose image is built from whatever trained)."
        ),
    ] = "gated",
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Fit win probability, the fourth-down components, and the play-call model.

    The models are trained into a candidate directory. With tracking on, the run becomes a
    new registered version (`fourthdown.models.registry`) and is promoted into the served
    directory per ``--promote``; with ``--no-track`` the candidate is copied in directly.
    """
    if promote not in release.PROMOTE_MODES:
        raise typer.BadParameter(f"--promote must be one of {', '.join(release.PROMOTE_MODES)}")
    _configure_logging(verbose)
    paths = data_paths(data_dir)
    try:
        outcome = release.train_and_release(
            paths,
            report=output,
            split=_split(train_seasons, valid_seasons, test_seasons),
            epochs=epochs,
            audit_sample=audit_sample,
            track=track,
            promote=promote,
        )
    except release.DataDriftError as error:
        for problem in error.problems:
            typer.echo(f"DRIFT {problem}", err=True)
        typer.echo("data does not match data/manifest.json; rerun `fourthdown build`", err=True)
        raise typer.Exit(code=1) from error
    except DataQualityError as error:
        typer.echo(f"{error}\nNothing was trained.", err=True)
        raise typer.Exit(code=1) from error
    card, results = outcome.card, outcome.results
    if card.mlflow_run_id:
        typer.echo(f"MLflow run {card.mlflow_run_id} at {card.mlflow_tracking_uri}")
    for failure in outcome.serving.gate_failures:
        typer.echo(f"GATE {failure}", err=True)
    for note in outcome.serving.notes:
        typer.echo(note, err=note.startswith("not promoted"))
    typer.echo(f"data version {card.data_version}; model card at {paths.models / MODEL_CARD}")
    typer.echo(
        f"win probability log loss {results.win_probability.log_loss:.4f} "
        f"(ECE {results.win_probability.calibration_error:.4f}), "
        f"play call {results.play_call.accuracy:.1%} vs "
        f"{results.play_call_majority.accuracy:.1%} base rate; written to {output}"
    )
    gates = [
        _gate(
            "win-probability log loss",
            results.win_probability.log_loss,
            max_wp_log_loss,
            higher_is_better=False,
        ),
        _gate("play-call accuracy", results.play_call.accuracy, min_play_call_accuracy),
    ]
    if not all(gates):
        raise typer.Exit(code=1)


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
    paths = data_paths(data_dir)
    client = default_client(model)
    with ExitStack() as stack:
        tools: list[Tool] = []
        # The same openers the services use, in one process: no services need be running.
        for open_ in (
            warehouse_service.opener(paths, client),
            retrieval_service.opener(client),
            models_service.opener(paths),
        ):
            state = open_(stack)
            tools += state.tools
            for name, reason in state.skipped.items():
                typer.echo(f"unavailable: {name} ({reason})", err=True)
        if not tools:
            typer.echo("nothing is available to answer with", err=True)
            raise typer.Exit(code=1)
        assistant = Assistant(tools, HybridRouter(tools, client))
        try:
            answer = assistant.ask(question, tool=tool)
        except KeyError:
            loaded = ", ".join(assistant.tool_names)
            typer.echo(f"unknown tool {tool!r}; loaded: {loaded}", err=True)
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


SERVICES = ("gateway", "warehouse", "retrieval", "models")


@app.command("serve")
def serve_cmd(
    service: Annotated[str, typer.Argument(help=f"One of: {', '.join(SERVICES)}.")],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int | None, typer.Option("--port", help="Default: the service's.")] = None,
    data_dir: DataDir = None,
) -> None:
    """Run one service (see `fourthdown.services`); run all four for the full app.

    Locally: `fourthdown serve warehouse`, `... retrieval`, `... models`, then
    `... gateway`, each in its own terminal; or `make up` for all of them in Docker.
    """
    if service not in SERVICES:
        typer.echo(f"unknown service {service!r}; choose from {', '.join(SERVICES)}", err=True)
        raise typer.Exit(code=1)
    if data_dir is not None:
        os.environ["FOURTHDOWN_DATA_DIR"] = str(data_dir)
    importlib.import_module(f"fourthdown.services.{service}").main(host=host, port=port)


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
    min_accuracy: Threshold = None,
    model: Model = None,
    verbose: Verbose = False,
) -> None:
    """Score keyword, LLM, and hybrid (the one that serves) routing on the golden set."""
    _configure_logging(verbose)
    tools = routing_harness.catalog()
    client = default_client(model)
    reports = [routing_harness.evaluate(KeywordRouter(tools), name="keyword")]
    if client.available():
        reports += [
            routing_harness.evaluate(LLMRouter(tools, client), name=f"llm ({client.model})"),
            routing_harness.evaluate(HybridRouter(tools, client), name=f"hybrid ({client.model})"),
        ]
    else:
        typer.echo(f"skipping the LLM router: no {client.model} at {client.base_url}", err=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(routing_harness.render(reports))
    for report in reports:
        typer.echo(f"{report.name}: {report.correct}/{report.total} ({report.accuracy():.0%})")
    typer.echo(f"written to {output}")
    gates = [_gate(f"{r.name} routing accuracy", r.accuracy(), min_accuracy) for r in reports]
    if not all(gates):
        raise typer.Exit(code=1)


@app.command("misses")
def misses_cmd(
    data_dir: DataDir = None,
    verdict: Annotated[
        str | None,
        typer.Option(help="Only this verdict: expected_empty, wrong_query, or unreviewed."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Most recent N records.")] = 20,
    show_sql: Annotated[bool, typer.Option("--sql", help="Print each query.")] = False,
) -> None:
    """List questions whose query came back empty, newest last, with the review's verdict."""
    path = data_paths(data_dir).feedback / feedback.FILENAME
    records = [
        record
        for record in feedback.read(path)
        if verdict is None or record.get("verdict", feedback.NOT_REVIEWED) == verdict
    ]
    if not records:
        typer.echo(f"no empty results recorded in {path}")
        return
    for record in records[-limit:]:
        outcome = record.get("verdict", feedback.NOT_REVIEWED)
        if record.get("revised_rows"):
            outcome += f", revised ({record['revised_rows']} rows)"
        typer.echo(f"{record.get('at', '?')}  [{outcome}]  {record.get('question', '')}")
        reason = record.get("reason") or record.get("error") or record.get("note")
        if reason:
            typer.echo(f"    {reason}")
        if show_sql:
            for key in ("sql", "revised_sql"):
                if record.get(key):
                    typer.echo(f"    {key}: " + " ".join(str(record[key]).split()))
    typer.echo(f"{len(records)} record(s) in {path}")


@app.command("queries")
def queries_cmd(
    data_dir: DataDir = None,
    hours: Annotated[
        float | None, typer.Option("--hours", help="Only the last N hours; all when omitted.")
    ] = None,
    slowest: Annotated[int, typer.Option(help="List this many slowest questions.")] = 3,
) -> None:
    """Summarise every question the gateway answered: latency, errors, and route mix."""
    path = data_paths(data_dir).feedback / feedback.QUERIES_FILENAME
    records = feedback.since(feedback.read(path), hours)
    if not records:
        typer.echo(f"no queries recorded in {path}")
        return
    typer.echo(feedback.summarise(records, slowest=slowest).render())


registry_app = typer.Typer(help="Registered win-probability versions and which one serves.")
app.add_typer(registry_app, name="registry")


@registry_app.command("list")
def registry_list_cmd(data_dir: DataDir = None) -> None:
    """Every registered version, newest first, with its gate verdict and headline metrics."""
    paths = data_paths(data_dir)
    found = registry.versions(paths)
    if not found:
        typer.echo(f"no {registry.MODEL_NAME} versions in {tracking.tracking_uri(paths)}")
        return
    for version in found:
        marker = f" [{registry.ALIAS}]" if version.champion else ""
        metrics = ", ".join(
            f"{key} {version.metrics[key]:.4f}"
            for key in ("wp_log_loss", "wp_auc", "advisor_agreement")
            if key in version.metrics
        )
        typer.echo(
            f"v{version.version}{marker}  gates {version.gates}  data {version.data_version}  "
            f"{metrics}"
        )
        if version.failures:
            typer.echo(f"    {version.failures}")


@registry_app.command("promote")
def registry_promote_cmd(
    version: Annotated[int, typer.Argument(help="The registered version to serve.")],
    data_dir: DataDir = None,
) -> None:
    """Serve a registered version that passed its gates (e.g. to roll back)."""
    paths = data_paths(data_dir)
    try:
        card = registry.promote(paths, version)
    except registry.RegistryError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"version {version} is {registry.ALIAS}; installed into {paths.models} "
        f"(data {card.data_version}, run {card.mlflow_run_id})"
    )


@app.command("pipeline")
def pipeline_cmd(
    seasons: Seasons = DEFAULT_SEASONS,
    index: Annotated[
        bool,
        typer.Option(
            "--index/--no-index",
            help="Embed narratives into Postgres (needs Postgres and the embedding model).",
        ),
    ] = True,
    index_seasons: Annotated[
        str | None,
        typer.Option(help="Seasons to embed narratives for; all in the warehouse by default."),
    ] = None,
    train_seasons: SplitSeasons = None,
    valid_seasons: SplitSeasons = None,
    test_seasons: SplitSeasons = None,
    epochs: Annotated[int, typer.Option("--epochs", help="Cap on win-probability epochs.")] = 60,
    audit_sample: Annotated[
        int, typer.Option("--audit-sample", help="Held-out fourth downs to replay.")
    ] = 4000,
    promote: Annotated[
        str, typer.Option(help="'gated' (the default), 'never', or 'always'.")
    ] = "gated",
    track: Annotated[bool, typer.Option("--track/--no-track", help="Log to MLflow.")] = True,
    data_dir: DataDir = None,
    verbose: Verbose = False,
) -> None:
    """Run the Dagster asset graph once: every completed week of SEASONS, then publish.

    Registers each week the nflverse schedule shows complete as a partition, ingests the
    weeks not processed yet in one run, then builds the warehouse with dbt (incrementally
    when only weeks were added), indexes new narratives, retrains unless the data and
    settings are unchanged, and scores the result. What the compose `pipeline` service and
    the Helm pipeline Job run; `make dagster` shows the same runs.
    """
    from dagster import DagsterInstance

    from fourthdown.orchestration.run import run_pipeline

    _configure_logging(verbose)
    if promote not in release.PROMOTE_MODES:
        raise typer.BadParameter(f"--promote must be one of {', '.join(release.PROMOTE_MODES)}")
    split = [train_seasons, valid_seasons, test_seasons]
    if any(split) and not all(split):
        raise typer.BadParameter(
            "give --train-seasons, --valid-seasons and --test-seasons together"
        )
    paths = data_paths(data_dir)
    paths.ensure()
    if not os.environ.get("DAGSTER_HOME"):
        os.environ["DAGSTER_HOME"] = str(paths.root / "dagster")
    Path(os.environ["DAGSTER_HOME"]).mkdir(parents=True, exist_ok=True)
    train = {
        "train_seasons": train_seasons,
        "valid_seasons": valid_seasons,
        "test_seasons": test_seasons,
        "epochs": epochs,
        "audit_sample": audit_sample,
        "promote": promote,
        "track": track,
    }
    wanted = ingest.parse_seasons(seasons, latest=_latest_season())
    with DagsterInstance.get() as instance:
        result = run_pipeline(
            paths,
            wanted,
            instance=instance,
            index=index,
            index_seasons=index_seasons,
            train=train,
        )
    if result.registered:
        typer.echo(f"registered {len(result.registered)} new week partition(s)")
    typer.echo(
        f"ingested {len(result.ingested)} week(s)"
        + (f" ({result.ingested[0]} to {result.ingested[-1]})" if result.ingested else "")
    )
    if not result.success:
        typer.echo(f"the {result.failed_step} run failed; see its logs above", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"published; reports in {paths.reports}")


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
    ingest.download_players(paths)
    _transform(wanted, paths)
    target = _build_warehouse(paths)
    typer.echo(f"built {len(wanted)} seasons into {target}")


if __name__ == "__main__":
    app()
