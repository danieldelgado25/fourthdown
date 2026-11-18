"""The data pipeline as a Dagster asset graph (phase 09).

``dagster dev -m fourthdown.orchestration`` (``make dagster``) serves the graph, the
weekly schedule, and the new-data sensor locally; ``fourthdown pipeline`` runs it once,
in-process, which is what the compose ``pipeline`` service and the Helm Job do.
"""

from __future__ import annotations

from dagster import Definitions

from fourthdown.orchestration.assets import ASSETS, CHECKS, DataRoot, dbt_resource
from fourthdown.orchestration.automation import (
    alert_on_failure,
    daily_freshness,
    freshness_job,
    ingest_job,
    new_data_sensor,
    publish_job,
    publish_without_index_job,
    weekly_ingest,
)

defs = Definitions(
    assets=ASSETS,
    asset_checks=CHECKS,
    jobs=[ingest_job, publish_job, publish_without_index_job, freshness_job],
    schedules=[weekly_ingest, daily_freshness],
    sensors=[new_data_sensor, alert_on_failure],
    resources={"data": DataRoot(), "dbt": dbt_resource()},
)

__all__ = ["defs"]
