"""The DuckDB warehouse: built by dbt, read by every query path.

dbt (``dbt_project/``, run by `fourthdown.data.modeling`) models the processed Parquet as
a star schema (``core.fact_plays`` and its dimensions) and rebuilds from it the six
semantic tables in ``main`` that are the only surface the query layer and the text-to-SQL
prompt see. Keeping that vocabulary small and pre-joined is what makes generated SQL
correct often enough to be useful: a model asked to reason about ``team_game`` rarely
invents a join, whereas a model handed 372 raw columns, or a star schema, invents one
constantly.

A build is full (a fresh database) or incremental: ``plan`` compares the partition hashes
the live warehouse was built from with the ones on disk, and when only weeks were added or
changed, dbt runs on a copy of the live database with just those weeks (``fact_plays`` is
an incremental model; everything downstream of it is small enough to rebuild). Either way
the result is built beside the live file and swapped in only if every data test passed.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from fourthdown.config import Paths, data_paths
from fourthdown.data import provenance

if TYPE_CHECKING:
    from fourthdown.data.modeling import BuildOutcome

LOGGER = logging.getLogger(__name__)

VIEW_NAMES: tuple[str, ...] = ("plays", "games", "drives", "team_game", "player_game", "players")
"""The semantic layer (dbt_project/models/semantic): what generated SQL may read."""


def _anchor(connection: duckdb.DuckDBPyConnection, paths: Paths) -> None:
    """Resolve the staging views' relative Parquet paths against this machine's data root.

    GLOBAL, so cursors taken from the connection for concurrent requests inherit it.
    """
    root = str(paths.root).replace("'", "''")
    connection.execute(f"SET GLOBAL file_search_path = '{root}'")


@dataclass(frozen=True)
class BuildPlan:
    """What the next build has to do: everything, or only some weeks."""

    incremental: bool
    weeks: tuple[str, ...]
    reason: str

    @classmethod
    def full(cls, reason: str) -> BuildPlan:
        return cls(incremental=False, weeks=(), reason=reason)


def plan(paths: Paths, *, database: Path | None = None) -> BuildPlan:
    """Decide between a full and an incremental build of ``paths``' warehouse.

    Incremental only when the live warehouse exists, was built by the same dbt code, and
    every partition it was built from is still on disk; then the plan lists the weeks that
    are new or whose bytes changed (nflverse corrects earlier weeks too).
    """
    from fourthdown.data import modeling

    target = database or paths.database
    manifest = provenance.load(paths)
    if not target.exists():
        return BuildPlan.full("no live warehouse")
    if not manifest.built_from:
        return BuildPlan.full("no record of what the live warehouse was built from")
    project = modeling.project_dir()
    for name, model in manifest.models.items():
        source = project / model.path
        if not source.exists() or modeling.sha256_file(source) != model.sha256:
            return BuildPlan.full(f"dbt model {name} changed")
    if manifest.project_sha256 != modeling.project_sha256():
        return BuildPlan.full("dbt macros, seeds, or config changed")
    current = {key: record.sha256 for key, record in manifest.processed.items()}
    removed = sorted(set(manifest.built_from) - set(current))
    if removed:
        return BuildPlan.full(f"partitions removed since the last build: {', '.join(removed)}")
    changed = tuple(
        sorted(key for key, digest in current.items() if manifest.built_from.get(key) != digest)
    )
    return BuildPlan(
        incremental=True,
        weeks=changed,
        reason=f"{len(changed)} new or changed week(s)" if changed else "no partition changed",
    )


def _require_partitions(paths: Paths) -> list[Path]:
    partitions = sorted(paths.processed.glob("season=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(
            f"no processed partitions under {paths.processed}. Run `fourthdown etl` first."
        )
    return partitions


def stage(paths: Paths, build_plan: BuildPlan, *, database: Path | None = None) -> Path:
    """Prepare the database dbt builds into, beside the live one.

    Build beside the live file and swap it in: a running warehouse service holds a
    read-only lock on the live one, which a writer cannot take but a rename never needs.
    The staging file keeps the live file's name (in its own directory) because DuckDB
    names the catalog after the file, and dbt writes the catalog into every view. An
    incremental build starts from a copy of the live file.
    """
    target = database or paths.database
    staging_dir = target.parent / ".building"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    staging = staging_dir / target.name
    if build_plan.incremental:
        shutil.copy2(target, staging)
    return staging


def publish(
    paths: Paths,
    staging: Path,
    outcome: BuildOutcome,
    *,
    run_tests: bool,
    database: Path | None = None,
) -> Path:
    """Swap a passed build in for the live warehouse and record what it was built from."""
    from fourthdown.data import modeling

    target = database or paths.database
    with duckdb.connect(str(staging), read_only=True) as connection:
        _anchor(connection, paths)
        rows = connection.execute("SELECT count(*) FROM plays").fetchone()
    staging.replace(target)
    shutil.rmtree(staging.parent, ignore_errors=True)
    if run_tests:
        modeling.save_test_results(paths, outcome.tests)
    else:
        (paths.warehouse / modeling.TEST_RESULTS_FILE).unlink(missing_ok=True)
    manifest = provenance.record_models(
        paths, outcome.models, project_sha256=modeling.project_sha256()
    )
    LOGGER.info(
        "warehouse at %s covers %s plays in %d seasons; data version %s",
        target,
        rows[0] if rows else 0,
        len(manifest.seasons),
        manifest.data_version,
    )
    return target


def build(
    paths: Paths | None = None,
    *,
    database: Path | None = None,
    run_tests: bool = True,
    incremental: bool = False,
) -> Path:
    """Build the warehouse with dbt (``dbt_project/``), and swap it in only if it passed.

    ``incremental=True`` builds only the weeks ``plan`` finds new or changed, when it can.
    ``run_tests=False`` skips the data tests, for synthetic datasets too small to pass the
    league-wide ones; it also forgets the last build's results, which no longer apply.
    """
    from fourthdown.data import modeling  # dbt is a pipeline dependency, not a service one

    resolved = paths or data_paths()
    resolved.ensure()
    _require_partitions(resolved)
    build_plan = (
        plan(resolved, database=database) if incremental else BuildPlan.full("full build asked for")
    )
    LOGGER.info(
        "%s build: %s", "incremental" if build_plan.incremental else "full", build_plan.reason
    )
    staging = stage(resolved, build_plan, database=database)
    outcome = modeling.build(
        resolved,
        staging,
        run_tests=run_tests,
        weeks=build_plan.weeks if build_plan.incremental else None,
    )
    return publish(resolved, staging, outcome, run_tests=run_tests, database=database)


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
