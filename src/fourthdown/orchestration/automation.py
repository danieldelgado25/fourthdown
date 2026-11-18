"""Jobs, the weekly schedule, and the sensor that retrains when new data lands.

* ``ingest_weeks`` (partitioned by week): ``raw_plays`` and ``processed_plays``.
* ``publish``: everything downstream, from ``raw_players`` and the dbt build to the
  narrative index, the retrain, and the scorecard.
* ``weekly_ingest`` (Tuesdays at 10:00 New York time, September through February): asks
  the nflverse schedule which weeks of the current season are over, registers the new
  ones as partitions, and runs ``ingest_weeks`` for them plus the latest week again,
  since nflverse corrects a week's stats in the days after it.
* ``new_data_sensor``: when a processed week is newer than the last warehouse build, runs
  ``publish``. Several weeks landing together (a backfill) make one publish run.
* ``daily_freshness`` (noon New York time, September through February): the
  ``fresh_in_season`` check, which alerts when a finished week is not in the data.
* ``snowflake_sensor``: when the DuckDB warehouse is newer than the Snowflake build, runs
  ``publish_snowflake`` (phase 11). Off unless ``SNOWFLAKE_*`` is configured.
* ``alert_on_failure``: any failed run (a week that fails its data contract, a dbt test, the
  leakage contract) goes to ``fourthdown.quality.alerts`` with the failing steps' errors.

Runs queue one at a time (``docker/dagster.yaml``): every one of them writes the same
data volume.
"""

from __future__ import annotations

from dagster import (
    AssetKey,
    AssetSelection,
    DagsterInstance,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    RunFailureSensorContext,
    RunRequest,
    ScheduleEvaluationContext,
    SensorEvaluationContext,
    SkipReason,
    define_asset_job,
    run_failure_sensor,
    schedule,
    sensor,
)

from fourthdown.config import data_paths
from fourthdown.data import ingest, snowflake
from fourthdown.orchestration import partitions
from fourthdown.orchestration.assets import (
    fresh_in_season,
    model_scorecard,
    narrative_index,
    processed_plays,
    raw_players,
    raw_plays,
    snowflake_warehouse,
    trained_models,
    warehouse_models,
)
from fourthdown.orchestration.partitions import WEEKS
from fourthdown.quality import alerts

# Freshness runs on its own daily schedule, not inside every ingest.
INGEST = AssetSelection.assets(raw_plays, processed_plays) - AssetSelection.checks(fresh_in_season)
WAREHOUSE = AssetSelection.assets(raw_players, warehouse_models)
MODELS = AssetSelection.assets(trained_models, model_scorecard)

ingest_job = define_asset_job(
    "ingest_weeks",
    selection=INGEST,
    partitions_def=WEEKS,
    description="Download and transform NFL weeks (one partition each).",
)
publish_job = define_asset_job(
    "publish",
    selection=WAREHOUSE | AssetSelection.assets(narrative_index) | MODELS,
    description="Build the warehouse with dbt, index new narratives, retrain, and score.",
)
publish_without_index_job = define_asset_job(
    "publish_without_index",
    selection=WAREHOUSE | MODELS,
    description="`publish` for deployments with no Postgres or embedding model.",
)

snowflake_job = define_asset_job(
    "publish_snowflake",
    selection=AssetSelection.assets(snowflake_warehouse),
    description="Load new weeks into Snowflake and build the same dbt project there.",
)

freshness_job = define_asset_job(
    "check_freshness",
    selection=AssetSelection.checks(fresh_in_season),
    description="Is every finished week of the season in the data?",
)

WEEKLY_CRON = "0 10 * 1,2,9-12 2"
"""Tuesdays at 10:00, September through February: Monday night's game is in by then."""

LANDED = processed_plays.key
BUILT = AssetKey("fact_plays")


@schedule(
    job=ingest_job,
    cron_schedule=WEEKLY_CRON,
    execution_timezone="America/New_York",
    default_status=DefaultScheduleStatus.RUNNING,
)
def weekly_ingest(context: ScheduleEvaluationContext) -> list[RunRequest] | SkipReason:
    paths = data_paths()
    paths.ensure()
    ingest.download_schedule(paths)
    season = ingest.current_season(context.scheduled_execution_time.date())
    complete = partitions.completed(paths, [season])
    if not complete:
        return SkipReason(f"no week of {season} is complete yet")
    added = partitions.register(context.instance, complete)
    wanted = sorted({*added, complete[-1]})
    day = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return [RunRequest(partition_key=key, run_key=f"{key}@{day}") for key in wanted]


@sensor(
    job=publish_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.RUNNING,
)
def new_data_sensor(context: SensorEvaluationContext) -> RunRequest | SkipReason:
    landed = _latest(context.instance, LANDED)
    if landed is None:
        return SkipReason("no week has been processed yet")
    built = _latest(context.instance, BUILT)
    if built is not None and built > landed:
        return SkipReason("the warehouse was built after the latest processed week")
    # One request per processed materialization: a failed publish is not retried for the
    # same data, and the next week to land asks again.
    return RunRequest(run_key=f"processed-{landed}")


def _latest(instance: DagsterInstance, key: AssetKey) -> int | None:
    """The event-log position of the asset's latest materialization (any partition)."""
    records = instance.fetch_materializations(key, limit=1).records
    return records[0].storage_id if records else None


@sensor(
    job=snowflake_job,
    minimum_interval_seconds=300,
    default_status=(
        DefaultSensorStatus.RUNNING
        if snowflake.Settings.from_env().configured
        else DefaultSensorStatus.STOPPED
    ),
)
def snowflake_sensor(context: SensorEvaluationContext) -> RunRequest | SkipReason:
    built = _latest(context.instance, BUILT)
    if built is None:
        return SkipReason("the DuckDB warehouse has not been built yet")
    mirrored = _latest(context.instance, snowflake_warehouse.key)
    if mirrored is not None and mirrored > built:
        return SkipReason("Snowflake was built after the latest DuckDB build")
    return RunRequest(run_key=f"snowflake-{built}")


@schedule(
    job=freshness_job,
    cron_schedule="0 12 * 1,2,9-12 *",
    execution_timezone="America/New_York",
    default_status=DefaultScheduleStatus.RUNNING,
)
def daily_freshness(context: ScheduleEvaluationContext) -> RunRequest:
    day = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return RunRequest(run_key=f"freshness@{day}")


@run_failure_sensor(default_status=DefaultSensorStatus.RUNNING)
def alert_on_failure(context: RunFailureSensorContext) -> None:
    """Send every failed run, with what each failing step said, to the alert channel."""
    steps = [
        f"{event.step_key}: {event.step_failure_data.error.message.strip()}"
        for event in context.get_step_failure_events()
        if event.step_failure_data.error is not None
    ]
    alerts.send(
        f"run {context.dagster_run.job_name} failed ({context.dagster_run.run_id[:8]})",
        "\n".join(steps) or context.failure_event.message or "see the run's logs",
    )
