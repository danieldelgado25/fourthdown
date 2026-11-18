"""The data pipeline as a Dagster asset graph (phase 09).

``dagster dev -m fourthdown.orchestration`` (``make dagster``) serves the graph, the
weekly schedule, and the new-data sensor locally; ``fourthdown pipeline`` runs it once,
in-process, which is what the compose ``pipeline`` service and the Helm Job do.
"""

from __future__ import annotations

from dagster import Definitions

from fourthdown.orchestration.assets import ASSETS, DataRoot, dbt_resource
from fourthdown.orchestration.automation import (
    ingest_job,
    new_data_sensor,
    publish_job,
    publish_without_index_job,
    weekly_ingest,
)

defs = Definitions(
    assets=ASSETS,
    jobs=[ingest_job, publish_job, publish_without_index_job],
    schedules=[weekly_ingest],
    sensors=[new_data_sensor],
    resources={"data": DataRoot(), "dbt": dbt_resource()},
)

__all__ = ["defs"]
