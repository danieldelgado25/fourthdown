"""The Dagster asset graph, driven week by week against a fake nflverse.

The fake serves a synthetic season file and schedule with ETags, so the tests exercise the
real conditional download, the completeness check, the weekly transform, the incremental
dbt build, the schedule, and the sensor, without the network. dbt's league-wide data tests
are off (two synthetic teams cannot pass them); training needs real seasons, so it is
covered by its own skip logic here and by `fourthdown pipeline` on real data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from dagster import (
    AssetKey,
    DagsterInstance,
    RunRequest,
    SkipReason,
    build_schedule_context,
    build_sensor_context,
    instance_for_test,
)
from dagster._core.storage.tags import (
    ASSET_PARTITION_RANGE_END_TAG,
    ASSET_PARTITION_RANGE_START_TAG,
)

from fourthdown.config import data_paths
from fourthdown.data import ingest, provenance
from fourthdown.orchestration import defs, partitions
from fourthdown.orchestration.automation import new_data_sensor, weekly_ingest
from fourthdown.quality import alerts, leakage

WAREHOUSE_CONFIG = {"ops": {"warehouse": {"config": {"run_tests": False}}}}


class FakeNflverse:
    """What ``ingest._fetch`` would download, with ETags and conditional requests."""

    def __init__(self, raw_frame: pl.DataFrame) -> None:
        self.week_one = raw_frame
        self.week_two = raw_frame.with_columns(
            week=pl.lit(2, dtype=raw_frame["week"].dtype), game_id=pl.lit("2023_02_BBB_AAA")
        )
        self.season = raw_frame
        self.results: dict[str, float | None] = {"2023_01_AAA_BBB": 7.0, "2023_02_BBB_AAA": None}
        self.season_downloads = 0

    def play_week_two(self) -> None:
        self.season = pl.concat([self.week_one, self.week_two])
        self.results["2023_02_BBB_AAA"] = -3.0

    def schedule(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "game_id": list(self.results),
                "season": [2023, 2023],
                "week": [1, 2],
                "gameday": ["2023-09-10", "2023-09-17"],
                "result": list(self.results.values()),
            }
        )

    def fetch(self, url: str, destination: Path, *, etag: str | None = None):
        if url == ingest.SCHEDULE_URL:
            frame = self.schedule()
        elif url == ingest.PLAYERS_URL:
            frame = pl.DataFrame({"gsis_id": ["X"], "display_name": ["A B"], "position": ["QB"]})
        else:
            frame = self.season
        current = hashlib.sha256(frame.write_ipc(None).getvalue()).hexdigest()[:16]
        if etag == current:
            return None
        if url == ingest.season_url(2023):
            self.season_downloads += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(destination)
        return current, "Tue, 19 Sep 2023 00:00:00 GMT"


@pytest.fixture(autouse=True)
def tiny_data_cannot_be_measured(monkeypatch):
    """On a four-play game any feature separates the outcome perfectly; the measured half
    of the leakage contract needs real seasons (tests/test_quality.py covers it)."""
    monkeypatch.setattr(leakage, "measured", lambda frame, report: None)


@pytest.fixture()
def nflverse(raw_frame, monkeypatch) -> FakeNflverse:
    fake = FakeNflverse(raw_frame)
    monkeypatch.setattr(ingest, "_fetch", fake.fetch)
    return fake


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURTHDOWN_DATA_DIR", str(tmp_path))
    resolved = data_paths(tmp_path)
    resolved.ensure()
    return resolved


@pytest.fixture()
def instance():
    with instance_for_test() as active:
        yield active


# A synthetic game has a handful of plays, not 175: open the row-count bands for it.
SMALL_GAMES = {"ops": {"raw_plays": {"config": {"min_plays_per_game": 1, "min_week_mean": 1}}}}


def _ingest(instance: DagsterInstance, first: str, last: str | None = None):
    return defs.resolve_job_def("ingest_weeks").execute_in_process(
        instance=instance,
        run_config=SMALL_GAMES,
        tags={ASSET_PARTITION_RANGE_START_TAG: first, ASSET_PARTITION_RANGE_END_TAG: last or first},
        raise_on_error=False,
    )


def _build_warehouse(instance: DagsterInstance):
    selection = [AssetKey("raw_players"), *defs.resolve_assets_def("fact_plays").keys]
    return defs.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance,
        asset_selection=selection,
        run_config=WAREHOUSE_CONFIG,
        raise_on_error=False,
    )


def _metadata(instance: DagsterInstance, key: str) -> dict:
    event = instance.get_latest_materialization_event(AssetKey(key))
    assert event is not None and event.asset_materialization is not None
    return {name: value.value for name, value in event.asset_materialization.metadata.items()}


def test_the_graph_runs_from_ingest_through_dbt_to_the_models():
    graph = defs.resolve_asset_graph()
    assert graph.get(AssetKey("stg_plays")).parent_keys == {AssetKey("processed_plays")}
    assert graph.get(AssetKey("stg_players")).parent_keys == {AssetKey("raw_players")}
    assert AssetKey("plays") in graph.get(AssetKey("trained_models")).parent_keys
    assert graph.get(AssetKey("model_scorecard")).parent_keys == {AssetKey("trained_models")}
    assert graph.get(AssetKey("processed_plays")).partitions_def == partitions.WEEKS
    # dbt's data tests are asset checks on the models they test.
    checks = {key.asset_key.to_user_string() for key in graph.asset_check_keys}
    assert {"fact_plays", "dim_game", "mart_team_season_tendencies", "model_scorecard"} <= checks


def test_an_incomplete_week_fails_with_the_reason(nflverse, paths, instance):
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-02"])
    result = _ingest(instance, "2023-02")
    assert not result.success
    failure = next(event for event in result.all_events if event.is_step_failure)
    assert "2023-02: not every game of the week has a final result yet" in str(
        failure.step_failure_data.error
    )


def test_a_new_week_flows_end_to_end_without_a_full_rebuild(nflverse, paths, instance):
    # Week 1 is played; the schedule registers it, ingest runs, the sensor publishes.
    first_tuesday = datetime(2023, 9, 12, 10)
    requests = weekly_ingest(
        build_schedule_context(instance, scheduled_execution_time=first_tuesday)
    )
    assert [request.partition_key for request in requests] == ["2023-01"]
    assert _ingest(instance, "2023-01").success
    assert isinstance(new_data_sensor(build_sensor_context(instance)), RunRequest)
    assert _build_warehouse(instance).success
    assert _metadata(instance, "fact_plays")["build"] == "full"
    assert isinstance(new_data_sensor(build_sensor_context(instance)), SkipReason)

    week_one = paths.week_processed(2023, 1)
    week_one_before = (week_one.stat().st_mtime_ns, provenance.load(paths).processed["2023-01"])

    # Week 2 is played and nflverse republishes the season file.
    nflverse.play_week_two()
    next_tuesday = datetime(2023, 9, 19, 10)
    requests = weekly_ingest(
        build_schedule_context(instance, scheduled_execution_time=next_tuesday)
    )
    assert [request.partition_key for request in requests] == ["2023-02"]
    assert instance.get_dynamic_partitions(partitions.WEEKS_NAME) == ["2023-01", "2023-02"]
    assert _ingest(instance, "2023-02").success

    # Only week 2's partition was written.
    assert (week_one.stat().st_mtime_ns, provenance.load(paths).processed["2023-01"]) == (
        week_one_before
    )
    assert paths.week_processed(2023, 2).exists()

    # The sensor sees data newer than the warehouse; dbt rebuilds just that week.
    assert isinstance(new_data_sensor(build_sensor_context(instance)), RunRequest)
    assert _build_warehouse(instance).success
    built = _metadata(instance, "fact_plays")
    assert (built["build"], built["weeks_rebuilt"]) == ("incremental", "2023-02")
    from fourthdown.data import warehouse

    with warehouse.connect(paths) as connection:
        games = connection.execute("SELECT game_id FROM games ORDER BY game_id").fetchall()
    assert games == [("2023_01_AAA_BBB",), ("2023_02_BBB_AAA",)]
    assert (paths.reports / "lineage.md").exists()


def test_rerunning_a_week_whose_season_did_not_change_downloads_nothing(nflverse, paths, instance):
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    assert _ingest(instance, "2023-01").success
    assert _ingest(instance, "2023-01").success
    assert nflverse.season_downloads == 1


def test_a_backfill_ingests_every_week_in_one_run(nflverse, paths, instance):
    nflverse.play_week_two()
    ingest.download_schedule(paths)
    partitions.register(instance, partitions.completed(paths, [2023]))
    result = _ingest(instance, "2023-01", "2023-02")
    assert result.success
    assert instance.get_materialized_partitions(AssetKey("processed_plays")) == {
        "2023-01",
        "2023-02",
    }


def test_a_failing_data_test_leaves_the_live_warehouse_alone(nflverse, paths, instance):
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    assert _ingest(instance, "2023-01").success
    assert _build_warehouse(instance).success
    before = paths.database.stat().st_mtime_ns
    # With the data tests on, two synthetic teams fail "every season has 32 teams".
    result = defs.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance,
        asset_selection=list(defs.resolve_assets_def("fact_plays").keys),
        raise_on_error=False,
    )
    assert not result.success
    assert paths.database.stat().st_mtime_ns == before
    failed = [
        evaluation for evaluation in result.get_asset_check_evaluations() if not evaluation.passed
    ]
    assert failed, "the failing dbt tests should still report as failed asset checks"


def test_training_is_skipped_when_the_data_has_not_changed(paths, instance, monkeypatch):
    from fourthdown.models import release

    monkeypatch.setattr(release, "is_current", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        release,
        "train_and_release",
        lambda *args, **kwargs: pytest.fail("trained although the data did not change"),
    )
    result = defs.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance, asset_selection=[AssetKey("trained_models")]
    )
    assert result.success
    assert instance.get_latest_materialization_event(AssetKey("trained_models")) is None


# --- phase 10: data quality in the graph --------------------------------------------------


def _failure_text(result) -> str:
    return "\n".join(
        str(event.step_failure_data.error.message)
        for event in result.all_events
        if event.is_step_failure and event.step_failure_data.error is not None
    )


def _failed_checks(result) -> set[str]:
    return {e.check_name for e in result.get_asset_check_evaluations() if not e.passed}


def test_a_corrupted_file_fails_the_run_with_a_clear_message(nflverse, paths, instance):
    # nflverse publishes ydstogo as text: a cast would have loaded it as nulls.
    nflverse.season = nflverse.week_one.with_columns(pl.col("ydstogo").cast(pl.String))
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    result = _ingest(instance, "2023-01")
    assert not result.success
    assert _failed_checks(result) == {"raw_contract"}
    message = _failure_text(result)
    assert "nflverse week 2023-01 failed its data quality checks; nothing was written" in message
    assert "2023-01: ydstogo: expected numbers, got String; it would load as nulls" in message
    assert not paths.week_processed(2023, 1).exists()


def test_impossible_values_fail_the_processed_contract(nflverse, paths, instance):
    nflverse.season = nflverse.week_one.with_columns(pl.lit(9.0).alias("down"))
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    result = _ingest(instance, "2023-01")
    assert not result.success
    assert _failed_checks(result) == {"processed_contract"}
    assert "down: 4 value(s) not between 1 and 4 (e.g. 9)" in _failure_text(result)
    assert not paths.week_processed(2023, 1).exists()


def test_a_truncated_redelivery_fails_the_row_counts(nflverse, paths, instance):
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    assert _ingest(instance, "2023-01").success
    nflverse.season = nflverse.week_one.head(2)  # half the week comes back
    result = _ingest(instance, "2023-01")
    assert not result.success
    assert _failed_checks(result) == {"row_counts"}
    assert "2023-01: 2 plays, -50.0% against the 4 ingested before" in _failure_text(result)
    assert pl.read_parquet(paths.week_processed(2023, 1)).height == 4


def test_a_broken_leakage_contract_blocks_training(nflverse, paths, instance, monkeypatch):
    from fourthdown.models import features, release

    monkeypatch.setattr(features, "WP_FEATURES", (*features.WP_FEATURES, "wp"))
    monkeypatch.setattr(
        release, "train_and_release", lambda *a, **k: pytest.fail("trained past the contract")
    )
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    assert _ingest(instance, "2023-01").success
    selection = [
        AssetKey("raw_players"),
        *defs.resolve_assets_def("fact_plays").keys,
        AssetKey("trained_models"),
    ]
    result = defs.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance,
        asset_selection=selection,
        run_config=WAREHOUSE_CONFIG,
        raise_on_error=False,
    )
    assert not result.success
    assert "leakage_contract" in _failed_checks(result)
    assert instance.get_latest_materialization_event(AssetKey("trained_models")) is None
    evaluation = next(
        e for e in result.get_asset_check_evaluations() if e.check_name == "leakage_contract"
    )
    assert evaluation.metadata["violations"].value == [
        "win probability: wp are nflfastR model outputs, never features"
    ]


def test_a_failed_run_is_sent_to_the_alert_channel(nflverse, paths, instance, monkeypatch):
    from dagster import build_run_status_sensor_context

    from fourthdown.orchestration.automation import alert_on_failure

    sent = []
    monkeypatch.setattr(alerts, "send", lambda title, body: sent.append((title, body)) or True)
    nflverse.season = nflverse.week_one.with_columns(pl.lit(9.0).alias("down"))
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    result = _ingest(instance, "2023-01")
    context = build_run_status_sensor_context(
        sensor_name="alert_on_failure",
        dagster_instance=instance,
        dagster_run=result.dagster_run,
        dagster_event=result.get_run_failure_event(),
    ).for_run_failure()
    alert_on_failure(context)
    ((title, body),) = sent
    assert title.startswith("run ingest_weeks failed")
    assert "down: 4 value(s) not between 1 and 4" in body


def test_a_rejected_download_is_quarantined_and_the_good_file_kept(nflverse, paths, instance):
    ingest.download_schedule(paths)
    partitions.register(instance, ["2023-01"])
    assert _ingest(instance, "2023-01").success
    good = paths.season_raw(2023).read_bytes()
    nflverse.season = nflverse.week_one.with_columns(pl.col("ydstogo").cast(pl.String))
    result = _ingest(instance, "2023-01")
    assert not result.success
    assert "the rejected download is in" in _failure_text(result)
    assert paths.season_raw(2023).read_bytes() == good
    (held,) = ingest.quarantine_dir(paths).glob("play_by_play_2023.*.parquet")
    assert pl.read_parquet(held).schema["ydstogo"] == pl.String
    # The kept file is the one the manifest records; no raw or processed drift.
    assert [p for p in provenance.verify(paths) if p.startswith(("raw", "processed"))] == []
