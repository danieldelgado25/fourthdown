"""The pipeline as Dagster assets.

::

    raw_plays[week] -> processed_plays[week] -> dbt (staging -> core -> marts, semantic)
    raw_players ----------------------------------^          |
                                                             +-> narrative_index
                                                             +-> trained_models -> model_scorecard
                                                             +-> snowflake_warehouse

``raw_plays`` and ``processed_plays`` are partitioned by NFL week; everything from dbt down
is one unpartitioned graph that reads every week. A new week therefore costs one
conditional download, one week's transform, an incremental dbt build (``fact_plays``
rebuilds only that week's games), embeddings for that week's narratives only, and the
retrain, which by its nature fits on every season.

``snowflake_warehouse`` builds the same dbt project in Snowflake from the same partitions,
after the DuckDB build has passed its tests (phase 11; its own job and sensor, which stay
off until ``SNOWFLAKE_*`` is configured).
"""

import shutil
import sys
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetExecutionContext,
    AssetKey,
    BackfillPolicy,
    Config,
    ConfigurableResource,
    Failure,
    MaterializeResult,
    MetadataValue,
    Output,
    asset,
    asset_check,
)
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, dbt_assets

from fourthdown.config import Paths, data_paths, week_key
from fourthdown.data import etl, ingest, modeling, provenance, warehouse
from fourthdown.orchestration.partitions import WEEKS, by_season
from fourthdown.quality import DataQualityError, alerts, contracts, freshness, leakage, volume


class DataRoot(ConfigurableResource):  # type: ignore[type-arg]
    """The data root every asset reads and writes; ``FOURTHDOWN_DATA_DIR`` by default."""

    root: str | None = None

    def paths(self) -> Paths:
        paths = data_paths(self.root)
        paths.ensure()
        return paths


def dbt_resource() -> DbtCliResource:
    """dbt, run from this environment against the project that builds the warehouse."""
    project = str(modeling.project_dir())
    scripts = Path(sys.executable).parent
    executable = shutil.which("dbt", path=str(scripts)) or shutil.which("dbt") or "dbt"
    return DbtCliResource(project_dir=project, profiles_dir=project, dbt_executable=executable)


# --- ingestion: one partition per week ----------------------------------------------------


class IngestConfig(Config):
    """The row-count bands (``fourthdown.quality.volume``); the defaults fit 2009-2025."""

    min_plays_per_game: int = 100
    max_plays_per_game: int = 300
    min_week_mean: float = 145.0
    max_week_mean: float = 225.0
    max_redelivery_change: float = 0.05

    def bands(self) -> volume.VolumeBands:
        return volume.VolumeBands(
            min_plays_per_game=self.min_plays_per_game,
            max_plays_per_game=self.max_plays_per_game,
            min_week_mean=self.min_week_mean,
            max_week_mean=self.max_week_mean,
            max_redelivery_change=self.max_redelivery_change,
        )


RAW_CHECKS = {
    "complete_week": "Every game the nflverse schedule lists for the week is final and present.",
    "raw_contract": "Required columns present, numeric columns numeric, unique play keys, "
    "the right season (Pandera).",
    "row_counts": "Plays per game and per week inside the 2009-2025 bands, and within 5% of "
    "the last ingest of the same week.",
}


def _results(failures: Mapping[str, list[str]]) -> list[AssetCheckResult]:
    return [
        AssetCheckResult(
            check_name=name,
            passed=not problems,
            metadata={
                "problems": MetadataValue.md("\n".join(f"- {p}" for p in problems) or "none")
            },
        )
        for name, problems in failures.items()
    ]


def _failure(subject: str, failures: Mapping[str, list[str]], note: str = "") -> Failure:
    lines = [f"{subject} failed its data quality checks; nothing was written{note}:"]
    for name, problems in failures.items():
        if problems:
            lines.append(f"{name}:")
            lines += [f"  - {problem}" for problem in problems]
    return Failure(description="\n".join(lines))


def _inspect(
    source: Path,
    paths: Paths,
    season: int,
    season_weeks: list[int],
    previous: Mapping[str, provenance.PartitionRecord],
    bands: volume.VolumeBands,
) -> tuple[dict[str, list[str]], int]:
    """Run ``RAW_CHECKS`` on the requested weeks of one season file."""
    failures: dict[str, list[str]] = {name: [] for name in RAW_CHECKS}
    raw = pl.read_parquet(source)
    complete = set(ingest.completed_weeks(paths, season))
    games = 0
    for week in season_weeks:
        key = week_key(season, week)
        rows = raw.filter(pl.col("week") == week) if "week" in raw.columns else raw.clear()
        scheduled = ingest.scheduled_games(paths, season, week)
        present = (
            {str(game) for game in rows["game_id"].unique()} if "game_id" in rows.columns else set()
        )
        if not scheduled:
            failures["complete_week"].append(f"{key}: the nflverse schedule lists no games")
        elif week not in complete:
            failures["complete_week"].append(
                f"{key}: not every game of the week has a final result yet"
            )
        elif missing := sorted(scheduled - present):
            failures["complete_week"].append(
                f"{key}: the play-by-play lacks {len(missing)} of {len(scheduled)} games "
                f"({', '.join(missing[:3])}); nflverse has not published them yet"
            )
        try:
            contracts.validate_raw(rows, season=season, subject=key)
        except DataQualityError as error:
            failures["raw_contract"] += [f"{key}: {problem}" for problem in error.problems]
        if "game_id" in rows.columns:
            earlier = previous.get(key)
            failures["row_counts"] += volume.check_week(
                rows, key, previous_rows=earlier.rows if earlier else None, bands=bands
            )
        games += len(present)
    return failures, games


@asset(
    partitions_def=WEEKS,
    backfill_policy=BackfillPolicy.single_run(),
    group_name="ingest",
    kinds={"parquet"},
    check_specs=[
        AssetCheckSpec(name, asset="raw_plays", description=text)
        for name, text in RAW_CHECKS.items()
    ],
)
def raw_plays(
    context: AssetExecutionContext, data: DataRoot, config: IngestConfig
) -> Iterator[Any]:
    """A week of nflverse play-by-play, checked before anything is built from it.

    nflverse publishes a file per season and rewrites it nightly in season; the download
    is conditional on its ETag, so asking for a week whose season did not change costs a
    request and no transfer. Each week then has to pass three checks (``RAW_CHECKS``): it
    is complete against the schedule, it meets the raw contract, and its row counts are
    the size of real football. A failure fails the run with every problem listed.
    """
    paths = data.paths()
    ingest.download_schedule(paths)
    weeks = by_season(context.partition_keys)
    previous = provenance.load(paths).processed
    failures: dict[str, list[str]] = {name: [] for name in RAW_CHECKS}
    games = 0
    quarantined: list[int] = []
    for season, season_weeks in weeks.items():
        inspected: list[tuple[dict[str, list[str]], int]] = []

        def accept(
            candidate: Path,
            season: int = season,
            season_weeks: list[int] = season_weeks,
            inspected: list[tuple[dict[str, list[str]], int]] = inspected,
        ) -> None:
            found = _inspect(candidate, paths, season, season_weeks, previous, config.bands())
            inspected.append(found)
            # A week not yet over is not a fault of the file; only integrity rejects it.
            problems = found[0]["raw_contract"] + found[0]["row_counts"]
            if problems:
                raise DataQualityError(f"the new nflverse {season} file", problems)

        try:
            ingest.download_season(season, paths, refresh=True, accept=accept)
        except DataQualityError:
            quarantined.append(season)
        found, count = (
            inspected[0]
            if inspected
            else _inspect(
                paths.season_raw(season), paths, season, season_weeks, previous, config.bands()
            )
        )
        for name, problems in found.items():
            failures[name] += problems
        games += count
    results = _results(failures)
    if any(failures.values()):
        yield from results
        held = (
            f"; the rejected download is in {ingest.quarantine_dir(paths)} and the previous "
            "season file is still in place"
            if quarantined
            else ""
        )
        raise _failure(
            f"nflverse week{'s' if len(context.partition_keys) > 1 else ''} "
            + ", ".join(context.partition_keys),
            failures,
            note=held,
        )
    manifest = provenance.load(paths)
    yield MaterializeResult(
        metadata={
            "weeks": len(context.partition_keys),
            "games": games,
            "seasons": ", ".join(str(season) for season in weeks),
            "raw_sha256": MetadataValue.json(
                {str(season): manifest.raw[season].sha256[:12] for season in weeks}
            ),
        },
        check_results=results,
    )


@asset(
    partitions_def=WEEKS,
    backfill_policy=BackfillPolicy.single_run(),
    deps=[raw_plays],
    group_name="ingest",
    kinds={"polars", "parquet"},
    check_specs=[
        AssetCheckSpec(
            "processed_contract",
            asset="processed_plays",
            description="Exact columns and types, value ranges, allowed labels, and the "
            "partition's own season and week (Pandera).",
        )
    ],
)
def processed_plays(context: AssetExecutionContext, data: DataRoot) -> Iterator[Any]:
    """The week's plays through the Polars transform: ``season=S/week=WW.parquet``.

    A run that covers every week of a season transforms the season file once; otherwise
    each week is transformed alone and the other weeks' files keep their bytes. Every
    partition is checked against the processed contract before it is written.
    """
    paths = data.paths()
    written: list[str] = []
    try:
        for season, season_weeks in by_season(context.partition_keys).items():
            on_file = set(ingest.season_weeks(paths, season))
            if on_file and on_file <= set(season_weeks):
                etl.transform_season(season, paths)
                written += [week_key(season, week) for week in sorted(on_file)]
            else:
                for week in season_weeks:
                    etl.transform_week(season, week, paths)
                    written.append(week_key(season, week))
    except DataQualityError as error:
        failures = {"processed_contract": [f"{error.subject}: {p}" for p in error.problems]}
        yield from _results(failures)
        raise _failure("the processed partitions", failures) from error
    manifest = provenance.load(paths)
    records = [manifest.processed[key] for key in written if key in manifest.processed]
    metadata: dict[str, Any] = {
        "weeks": len(records),
        "rows": sum(record.rows for record in records),
        "data_version": manifest.data_version,
    }
    if len(records) == 1:
        metadata["path"] = records[0].path
        metadata["sha256"] = records[0].sha256[:12]
    yield MaterializeResult(metadata=metadata, check_results=_results({"processed_contract": []}))


@asset_check(
    asset=processed_plays,
    description="In season, every week the schedule shows over (two days' grace) is ingested.",
)
def fresh_in_season(data: DataRoot) -> AssetCheckResult:
    """Freshness against the nflverse schedule (``fourthdown.quality.freshness``); alerts."""
    paths = data.paths()
    ingest.download_schedule(paths)
    report = freshness.check(paths, provenance.load(paths).processed, date.today())
    if not report.passed:
        alerts.send("the data is behind the season", "\n".join(report.stale))
    return AssetCheckResult(
        passed=report.passed,
        metadata={"status": report.summary(), "in_season": report.in_season},
    )


@asset(group_name="ingest", kinds={"parquet"})
def raw_players(data: DataRoot) -> MaterializeResult:
    """nflverse's player table (ids to full names), refreshed so new players get names."""
    paths = data.paths()
    ingest.download_players(paths, force=True)
    return MaterializeResult(metadata={"sha256": provenance.sha256_file(paths.players)[:12]})


# --- the warehouse: the dbt project, one asset per model --------------------------------

SOURCE_ASSETS = {"processed_plays": processed_plays.key, "players": raw_players.key}


class WarehouseTranslator(DagsterDbtTranslator):
    """dbt sources are the ingest assets above; models are keyed and grouped by layer."""

    def get_asset_key(self, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        if dbt_resource_props["resource_type"] == "source":
            return SOURCE_ASSETS[dbt_resource_props["name"]]
        return AssetKey(dbt_resource_props["name"])

    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> str | None:
        fqn = dbt_resource_props.get("fqn", [])
        if dbt_resource_props["resource_type"] == "seed" or len(fqn) < 3:
            return "warehouse_core"
        return f"warehouse_{fqn[1]}"


class WarehouseConfig(Config):
    run_tests: bool = True
    """False only for synthetic data too small to pass league-wide tests (the unit tests)."""


@dbt_assets(
    manifest=modeling.parse_manifest(),
    dagster_dbt_translator=WarehouseTranslator(),
    name="warehouse",
)
def warehouse_models(
    context: AssetExecutionContext, dbt: DbtCliResource, data: DataRoot, config: WarehouseConfig
) -> Iterator[Any]:
    """``dbt build`` on a staging copy, swapped in for the live warehouse only if it passed.

    ``warehouse.plan`` decides between an incremental build (only the weeks whose
    partitions are new or changed since the live warehouse was built) and a full one.
    dbt's data tests report as asset checks. Their results are held until the build has
    been swapped in, so a materialization in Dagster always means the live warehouse.
    """
    paths = data.paths()
    if set(context.selected_asset_keys) != set(warehouse_models.keys):
        raise Failure(
            description="the warehouse is built and swapped in whole; "
            "select every dbt asset (the `publish` job does)"
        )
    build_plan = warehouse.plan(paths)
    mode = "incremental" if build_plan.incremental else "full"
    context.log.info(f"{mode} build: {build_plan.reason}")
    staging = warehouse.stage(paths, build_plan)
    target = modeling.artifacts_dir(paths) / "target"
    (target / "run_results.json").unlink(missing_ok=True)
    weeks = build_plan.weeks if build_plan.incremental else None
    with modeling.environment(paths, staging):
        invocation = dbt.cli(
            ["build", *modeling.build_args(run_tests=config.run_tests, weeks=weeks)],
            context=context,
            target_path=target,
            raise_on_error=False,
        )
        events = list(invocation.stream())
        succeeded = invocation.is_successful()
    if not (target / "run_results.json").exists():
        raise Failure(description="dbt did not run; see the dbt log in the step's output")
    outcome = modeling.outcome_from_artifacts(target)
    if not succeeded:
        for event in events:
            if isinstance(event, AssetCheckResult):
                yield event
        failed = [test for test in outcome.tests if not test.passed]
        raise Failure(
            description="dbt build failed; the live warehouse was left as it was:\n"
            + "\n".join(f"{test.name}: {test.message}" for test in failed),
            metadata={"failed_tests": len(failed)},
        )
    warehouse.publish(paths, staging, outcome, run_tests=config.run_tests)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "lineage.md").write_text(provenance.render(provenance.load(paths)))
    build_metadata = {
        "build": mode,
        "reason": build_plan.reason,
        "weeks_rebuilt": ", ".join(build_plan.weeks) if build_plan.incremental else "all",
    }
    for event in events:
        if isinstance(event, Output):
            yield event.with_metadata({**event.metadata, **build_metadata})
        else:
            yield event


# --- what reads the warehouse -------------------------------------------------------------


@asset_check(
    asset=AssetKey("plays"),
    blocking=True,
    description="Every model feature is a declared pre-snap column, none is an nflfastR "
    "model output, and none predicts its target with AUC above 0.95. Blocks training.",
)
def leakage_contract(data: DataRoot) -> AssetCheckResult:
    """The leakage contract (``fourthdown.quality.leakage``) on the warehouse."""
    with warehouse.connect(data.paths()) as connection:
        report = leakage.check(connection)
    return AssetCheckResult(
        passed=report.passed,
        metadata={
            "violations": MetadataValue.json(report.violations),
            "single_feature_auc": MetadataValue.md(report.render()),
        },
    )


class IndexConfig(Config):
    seasons: str | None = None
    """A season spec (``2025``, ``2009-``) to embed; every season in the warehouse when
    unset. A CPU-only cluster may index one season to keep the first run short."""
    postseason_drives_only: bool = True
    """Game narratives for every game; drive narratives for the postseason only (4% of
    drives), as `fourthdown index --playoff-drives` has always indexed them."""


@asset(
    deps=[AssetKey("games"), AssetKey("drives")],
    group_name="retrieval",
    kinds={"postgres"},
)
def narrative_index(data: DataRoot, config: IndexConfig) -> MaterializeResult:
    """Game and drive narratives, embedded into pgvector; only documents not yet indexed."""
    from fourthdown.retrieval import DocumentStore, default_embedder, index

    paths = data.paths()
    with warehouse.connect(paths) as connection, DocumentStore.open() as store:
        report = index.build(
            connection,
            store,
            default_embedder(),
            seasons=(
                ingest.parse_seasons(config.seasons, latest=ingest.current_season())
                if config.seasons
                else None
            ),
            postseason_drives_only=config.postseason_drives_only,
            missing_only=True,
        )
    coverage = "\n".join(
        f"| {grain} | {count:,} | {first}-{last} |" for grain, count, first, last in report.coverage
    )
    return MaterializeResult(
        metadata={
            "documents_added": report.documents,
            "seconds": round(report.elapsed_seconds, 1),
            "coverage": MetadataValue.md(
                "| grain | documents | seasons |\n| --- | --- | --- |\n" + coverage
            ),
        }
    )


class TrainConfig(Config):
    """The same settings as `fourthdown train`; seasons default to the documented split."""

    train_seasons: str | None = None
    valid_seasons: str | None = None
    test_seasons: str | None = None
    epochs: int = 60
    audit_sample: int = 4000
    promote: str = "gated"
    track: bool = True
    force: bool = False
    """Retrain even when the last run used this data version and these settings."""


@asset(
    deps=[AssetKey("plays")],
    group_name="models",
    kinds={"pytorch", "mlflow"},
    output_required=False,
)
def trained_models(
    context: AssetExecutionContext, data: DataRoot, config: TrainConfig
) -> Iterator[Output[None]]:
    """Win probability, fourth-down, and play-call models: trained, registered, and served
    only through the registry's gates. Skipped (not materialized) when the last run was
    already on this data version with these settings."""
    from fourthdown.models import release
    from fourthdown.models.features import DEFAULT_SPLIT, SeasonSplit

    paths = data.paths()
    given = [config.train_seasons, config.valid_seasons, config.test_seasons]
    if any(given) and not all(given):
        raise Failure(description="give train_seasons, valid_seasons, and test_seasons together")
    split = (
        SeasonSplit(
            train=tuple(ingest.parse_seasons(config.train_seasons or "")),
            valid=tuple(ingest.parse_seasons(config.valid_seasons or "")),
            test=tuple(ingest.parse_seasons(config.test_seasons or "")),
        )
        if all(given)
        else DEFAULT_SPLIT
    )
    if not config.force and release.is_current(
        paths, split, epochs=config.epochs, audit_sample=config.audit_sample
    ):
        context.log.info("the last training run used this data version and these settings")
        return
    outcome = release.train_and_release(
        paths,
        report=paths.reports / "model_eval.md",
        split=split,
        epochs=config.epochs,
        audit_sample=config.audit_sample,
        track=config.track,
        promote=config.promote,
    )
    card, serving = outcome.card, outcome.serving
    yield Output(
        None,
        metadata={
            "data_version": card.data_version,
            "registered_version": serving.version or 0,
            "served": serving.served,
            "gates": "pass" if not serving.gate_failures else "; ".join(serving.gate_failures),
            "seasons": MetadataValue.json(card.seasons),
            **{name: float(value) for name, value in sorted(card.metrics.items())},
            "report": MetadataValue.path(str(outcome.report)),
        },
    )


class ScorecardConfig(Config):
    suites: list[str] = ["guard", "routing_keyword", "references", "grounding"]
    """Suites to run besides the models' own metrics. The LLM suites (text_to_sql,
    routing_llm, retrieval) are opt-in: each costs an LLM call per golden question."""


@asset(
    deps=[trained_models],
    group_name="models",
    kinds={"python"},
    check_specs=[AssetCheckSpec("gates_pass", asset="model_scorecard")],
)
def model_scorecard(data: DataRoot, config: ScorecardConfig) -> Iterator[Any]:
    """The evaluation scorecard (docs/scorecard.md's format) for the run just trained.

    The models suite is scored from the training run's own metrics rather than by
    training again; the other suites run here. The ``gates_pass`` check fails when any
    gated metric is out of bounds.
    """
    from fourthdown.evaluation import scorecard, suites
    from fourthdown.evaluation.scorecard import SuiteResult
    from fourthdown.llm import default_client
    from fourthdown.models import release
    from fourthdown.models.card import ModelCard
    from fourthdown.retrieval import default_embedder

    paths = data.paths()
    selected = [name for name in config.suites if name != suites.MODELS]
    options = suites.RunOptions(
        paths=paths, client=default_client(None), embedder=default_embedder()
    )
    results = suites.run(selected, options)
    card = ModelCard.load(paths.reports / release.LAST_TRAINED)
    results.append(
        SuiteResult(
            suites.MODELS,
            metrics=dict(card.metrics),
            findings=(f"from the training run on data version {card.data_version}",),
        )
    )
    card_ = scorecard.build(results)
    output = paths.reports / "scorecard.md"
    output.write_text(card_.render())
    (paths.reports / "scorecard.json").write_text(card_.to_json() + "\n")
    failures = [
        f"{verdict.gate.key} = {verdict.value} (want {verdict.gate.bound()})"
        for verdict in card_.failures()
    ]
    yield MaterializeResult(
        metadata={
            "passed": card_.passed,
            "suites": ", ".join(
                f"{suite.name} (skipped)" if suite.skipped else suite.name for suite in card_.suites
            ),
            "scorecard": MetadataValue.path(str(output)),
        }
    )
    yield AssetCheckResult(
        check_name="gates_pass",
        passed=card_.passed,
        metadata={"failures": MetadataValue.json(failures)},
    )


# --- Snowflake: the same dbt project on a second engine -------------------------------------


@asset(deps=[AssetKey("plays")], group_name="snowflake", kinds={"snowflake", "dbt"})
def snowflake_warehouse(data: DataRoot) -> MaterializeResult:
    """The dbt project built in Snowflake: new or changed weeks loaded, an incremental
    build in a zero-copy clone, swapped in only if every data test passed
    (``fourthdown.data.snowflake``). Runs after the DuckDB build, so Snowflake only ever
    receives weeks that already passed every contract and test."""
    from fourthdown.data import snowflake

    settings = snowflake.Settings.from_env()
    if not settings.configured:
        raise Failure(
            description="Snowflake is not configured; set " + ", ".join(settings.missing())
        )
    try:
        result = snowflake.build(data.paths(), settings)
    except (snowflake.SnowflakeError, modeling.DbtError) as error:
        raise Failure(
            description=f"{error}\nThe live Snowflake database was left as it was."
        ) from error
    return MaterializeResult(
        metadata={
            "database": result.database,
            "dbt_build": result.dbt,
            "reason": result.reason,
            "weeks_loaded": ", ".join(result.weeks_loaded) or "none",
            "rows_loaded": result.rows_loaded,
            "tests_passed": result.tests_passed,
            "seconds": result.seconds,
        }
    )


CHECKS = [leakage_contract, fresh_in_season]

ASSETS = [
    raw_plays,
    processed_plays,
    raw_players,
    warehouse_models,
    narrative_index,
    trained_models,
    model_scorecard,
    snowflake_warehouse,
]
