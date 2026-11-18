"""The DuckDB warehouse: built by dbt, read by every query path.

dbt (``dbt_project/``, run by `fourthdown.data.modeling`) models the processed Parquet as
a star schema (``core.fact_plays`` and its dimensions) and rebuilds from it the six
semantic tables in ``main`` that are the only surface the query layer and the text-to-SQL
prompt see. Keeping that vocabulary small and pre-joined is what makes generated SQL
correct often enough to be useful: a model asked to reason about ``team_game`` rarely
invents a join, whereas a model handed 372 raw columns, or a star schema, invents one
constantly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb

from fourthdown.config import Paths, data_paths
from fourthdown.data import provenance

LOGGER = logging.getLogger(__name__)

VIEW_NAMES: tuple[str, ...] = ("plays", "games", "drives", "team_game", "player_game", "players")
"""The semantic layer (dbt_project/models/semantic): what generated SQL may read."""


def _anchor(connection: duckdb.DuckDBPyConnection, paths: Paths) -> None:
    """Resolve the staging views' relative Parquet paths against this machine's data root.

    GLOBAL, so cursors taken from the connection for concurrent requests inherit it.
    """
    root = str(paths.root).replace("'", "''")
    connection.execute(f"SET GLOBAL file_search_path = '{root}'")


def build(
    paths: Paths | None = None, *, database: Path | None = None, run_tests: bool = True
) -> Path:
    """Build the warehouse with dbt (``dbt_project/``), and swap it in only if it passed.

    ``run_tests=False`` skips the data tests, for synthetic datasets too small to pass the
    league-wide ones; it also forgets the last build's results, which no longer apply.
    """
    from fourthdown.data import modeling  # dbt is a pipeline dependency, not a service one

    resolved = paths or data_paths()
    resolved.ensure()
    partitions = sorted(resolved.processed.glob("season=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(
            f"no processed partitions under {resolved.processed}. Run `fourthdown etl` first."
        )
    target = database or resolved.database
    # Build beside the live file and swap it in: a running warehouse service holds a
    # read-only lock on the live one, which a writer cannot take but a rename never needs.
    # The staging file keeps the live file's name (in its own directory) because DuckDB
    # names the catalog after the file, and dbt writes the catalog into every view.
    staging_dir = target.parent / ".building"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    staging = staging_dir / target.name
    outcome = modeling.build(resolved, staging, run_tests=run_tests)
    with duckdb.connect(str(staging), read_only=True) as connection:
        _anchor(connection, resolved)
        rows = connection.execute("SELECT count(*) FROM plays").fetchone()
    LOGGER.info("warehouse at %s covers %s plays in %d seasons", target, rows, len(partitions))
    staging.replace(target)
    shutil.rmtree(staging_dir, ignore_errors=True)
    if run_tests:
        modeling.save_test_results(resolved, outcome.tests)
    else:
        (resolved.warehouse / modeling.TEST_RESULTS_FILE).unlink(missing_ok=True)
    manifest = provenance.record_models(
        resolved, outcome.models, project_sha256=modeling.project_sha256()
    )
    LOGGER.info("data version %s", manifest.data_version)
    return target


def connect(paths: Paths | None = None, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, defaulting to read-only because every query path is a reader."""
    resolved = paths or data_paths()
    if not resolved.database.exists():
        raise FileNotFoundError(
            f"warehouse missing at {resolved.database}. Run `fourthdown warehouse` first."
        )
    connection = duckdb.connect(str(resolved.database), read_only=read_only)
    _anchor(connection, resolved)
    return connection
