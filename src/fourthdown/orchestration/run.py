"""Run the asset graph once, in-process: what `fourthdown pipeline` does.

The compose ``pipeline`` service and the Helm pipeline Job run this instead of a daemon:

1. download the nflverse schedule and register every completed week of the requested
   seasons as a partition;
2. run ``ingest_weeks`` once over the weeks not processed yet (one run for all of them,
   since the ingest assets take a single-run backfill policy);
3. run ``publish``: dbt (incremental when only weeks were added), the narratives not yet
   indexed, the retrain (skipped when the last run used this data and these settings),
   and the scorecard.

Re-running with nothing new costs the schedule download, a dbt build, and no training,
which is what the old ``docker/pipeline.sh`` bought with a stamp file. Runs are recorded in
the Dagster instance at ``$DAGSTER_HOME``, the same one ``make dagster`` shows.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dagster import DagsterInstance
from dagster._core.storage.tags import (
    ASSET_PARTITION_RANGE_END_TAG,
    ASSET_PARTITION_RANGE_START_TAG,
)

from fourthdown.config import Paths
from fourthdown.data import ingest
from fourthdown.orchestration import defs, partitions
from fourthdown.orchestration.assets import processed_plays

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    registered: list[str] = field(default_factory=list)
    ingested: list[str] = field(default_factory=list)
    success: bool = True
    failed_step: str | None = None


def _range(instance: DagsterInstance, keys: Sequence[str]) -> tuple[str, str]:
    """The first and last of ``keys`` in partition order (the order they were registered)."""
    registered = instance.get_dynamic_partitions(partitions.WEEKS_NAME)
    order = {key: position for position, key in enumerate(registered)}
    ordered = sorted(keys, key=order.__getitem__)
    return ordered[0], ordered[-1]


def run_pipeline(
    paths: Paths,
    seasons: Sequence[int],
    *,
    instance: DagsterInstance,
    index: bool = True,
    index_seasons: str | None = None,
    train: dict[str, Any] | None = None,
) -> PipelineResult:
    """Ingest what is missing, then publish. Never raises for a failed step: check
    ``success`` (the step's own error is in the run's logs)."""
    paths.ensure()
    # Every asset resolves the data root from the environment (``DataRoot``).
    os.environ["FOURTHDOWN_DATA_DIR"] = str(paths.root)
    # The miss log, the one directory a serving container writes; made by the data's owner.
    paths.feedback.mkdir(parents=True, exist_ok=True)
    result = PipelineResult()
    ingest.download_schedule(paths)
    keys = partitions.completed(paths, seasons)
    if not keys:
        raise ValueError(f"no completed week in seasons {list(seasons)}")
    result.registered = partitions.register(instance, keys)
    processed = instance.get_materialized_partitions(processed_plays.key)
    missing = [key for key in keys if key not in processed]
    if missing:
        first, last = _range(instance, missing)
        LOGGER.info("ingesting %d week(s), %s to %s", len(missing), first, last)
        ingested = defs.resolve_job_def("ingest_weeks").execute_in_process(
            instance=instance,
            tags={ASSET_PARTITION_RANGE_START_TAG: first, ASSET_PARTITION_RANGE_END_TAG: last},
            raise_on_error=False,
        )
        if not ingested.success:
            result.success, result.failed_step = False, "ingest_weeks"
            return result
        result.ingested = missing
    job = "publish" if index else "publish_without_index"
    ops: dict[str, Any] = {"trained_models": {"config": dict(train or {})}}
    if index:
        ops["narrative_index"] = {"config": {"seasons": index_seasons}}
    published = defs.resolve_job_def(job).execute_in_process(
        instance=instance,
        run_config={"ops": ops},
        raise_on_error=False,
    )
    if not published.success:
        result.success, result.failed_step = False, job
    return result
