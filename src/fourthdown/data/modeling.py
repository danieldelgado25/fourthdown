"""The dbt project (``dbt_project/``) that builds the warehouse, run in-process.

The Polars ETL writes processed Parquet; dbt turns it into a star schema (``core``), an
aggregate mart (``marts``), and the six semantic tables text-to-SQL is prompted with
(``main``), testing each as it goes. This module points dbt at a data root and a database
file, runs it, and reports what it built and what its tests found.

Everything dbt needs is passed through the environment rather than flags, so the same
settings serve ``warehouse.build`` and ``fourthdown dbt <any dbt command>``:

* ``FOURTHDOWN_DATA_DIR``: the data root, which the staging views read Parquet relative to;
* ``FOURTHDOWN_DBT_TARGET``: ``dev`` (DuckDB, the default) or ``prod`` (Snowflake,
  `fourthdown.data.snowflake`); the same models build on both;
* ``FOURTHDOWN_DBT_DATABASE``: the DuckDB file to build;
* ``FOURTHDOWN_SNOWFLAKE_DATABASE``: the Snowflake database to build, when not the live one
  (``SNOWFLAKE_DATABASE``): the clone a Snowflake build happens in;
* ``FOURTHDOWN_PLAYERS_AVAILABLE``: whether ``raw/players.parquet`` exists yet;
* ``DBT_PROJECT_DIR``/``DBT_PROFILES_DIR``: the project, which carries its own profile;
* ``DBT_TARGET_PATH``/``DBT_LOG_PATH``: dbt's artifacts, kept under the data root because
  the installed project directory is read-only in the container.

dbt itself is imported only when something runs it: the warehouse service image reads the
built tables and does not install dbt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fourthdown.config import Paths

LOGGER = logging.getLogger(__name__)

PROJECT_ENV = "FOURTHDOWN_DBT_PROJECT"
TEST_RESULTS_FILE = "data_tests.json"
DEFAULT_TARGET = "dev"
SNOWFLAKE_TARGET = "prod"
"""The Snowflake output in ``profiles.yml`` (`fourthdown.data.snowflake`)."""
SNOWFLAKE_DATABASE_ENV = "FOURTHDOWN_SNOWFLAKE_DATABASE"
# Files that change what a build writes without being a model: macros, the seed, and the
# project config (materializations, schemas). Tests and docs only check or describe.
PROJECT_INPUTS = ("dbt_project.yml", "macros", "seeds")


class DbtError(RuntimeError):
    """A dbt command failed: a model did not build, or a data test failed."""


def project_dir() -> Path:
    """The dbt project: ``$FOURTHDOWN_DBT_PROJECT``, else ``dbt_project/`` in the checkout."""
    configured = os.environ.get(PROJECT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "dbt_project"


def artifacts_dir(paths: Paths, target: str = DEFAULT_TARGET) -> Path:
    """dbt's target and log directories for this data root and dbt target.

    One per target: a Snowflake build must not overwrite the manifest that records what
    the DuckDB warehouse was built from.
    """
    base = paths.warehouse / "dbt"
    return base if target == DEFAULT_TARGET else base / target


@contextmanager
def environment(
    paths: Paths, database: Path | str | None = None, *, target: str = DEFAULT_TARGET
) -> Iterator[dict[str, str]]:
    """Set the variables the project and its profile read, restoring the old values after."""
    project = project_dir()
    if not (project / "dbt_project.yml").exists():
        raise FileNotFoundError(f"no dbt project at {project}; set {PROJECT_ENV}")
    artifacts = artifacts_dir(paths, target)
    values = {
        "FOURTHDOWN_DATA_DIR": str(paths.root),
        "FOURTHDOWN_DBT_TARGET": target,
        "FOURTHDOWN_DBT_DATABASE": str(
            (database or paths.database) if target == DEFAULT_TARGET else paths.database
        ),
        "FOURTHDOWN_PLAYERS_AVAILABLE": "true" if paths.players.exists() else "false",
        "DBT_PROJECT_DIR": str(project),
        "DBT_PROFILES_DIR": str(project),
        "DBT_TARGET_PATH": str(artifacts / "target"),
        "DBT_LOG_PATH": str(artifacts / "logs"),
    }
    if target != DEFAULT_TARGET and database is not None:
        values[SNOWFLAKE_DATABASE_ENV] = str(database)
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def invoke(
    paths: Paths,
    args: Sequence[str],
    *,
    database: Path | str | None = None,
    target: str = DEFAULT_TARGET,
) -> Any:
    """Run one dbt command against ``paths`` and return dbt's own result object."""
    from dbt.cli.main import dbtRunner

    with environment(paths, database, target=target):
        try:
            return dbtRunner().invoke(list(args))
        finally:
            _release_duckdb()
            _release_log_files()


def _release_log_files() -> None:
    """Close the dbt.log handle dbt keeps after a run, so the data root can be moved."""
    try:
        from dbt_common.events import event_manager_client
    except ImportError:  # pragma: no cover - dbt-core always ships dbt_common
        return
    for logger in list(event_manager_client.get_event_manager().loggers):
        python_logger = getattr(logger, "_python_logger", None)
        for handler in list(getattr(python_logger, "handlers", [])):
            handler.close()
    event_manager_client.cleanup_event_logger()


def _release_duckdb() -> None:
    """Close the DuckDB database dbt-duckdb keeps open for the life of the process.

    Otherwise the file stays locked after the run: it cannot be reopened read-only to
    check it, nor (on Windows) renamed over the live warehouse. dbt-duckdb's own
    ``close_all_connections`` only drops the reference and waits for garbage collection.
    """
    try:
        from dbt.adapters.duckdb.connections import DuckDBConnectionManager
    except ImportError:  # another adapter (a Snowflake target) holds no local file
        return
    with DuckDBConnectionManager._LOCK:
        held = DuckDBConnectionManager._ENV
        DuckDBConnectionManager._ENV = None
    if held is not None:
        held.close()


@dataclass(frozen=True)
class TestResult:
    """One data test from a build: what it checked and whether the data passed."""

    name: str
    status: str
    failures: int
    message: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class ModelRecord:
    """A model or seed the build wrote, for the provenance manifest."""

    name: str
    path: str
    schema: str
    materialized: str
    sha256: str
    reads: tuple[str, ...]


@dataclass(frozen=True)
class BuildOutcome:
    models: tuple[ModelRecord, ...]
    tests: tuple[TestResult, ...]


def _field(item: Any, name: str) -> Any:
    """An attribute of a dbt node object, or a key of the same node read from JSON."""
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _test_name(node: Any) -> str:
    """A readable label: ``accepted_values fact_plays.down``, or a singular test's name."""
    metadata = _field(node, "test_metadata")
    if metadata is None:
        return str(_field(node, "name")).replace("_", " ")
    model = str(_field(node, "attached_node") or "").rsplit(".", 1)[-1]
    column = _field(node, "column_name")
    target = f"{model}.{column}" if column else model
    kwargs = _field(metadata, "kwargs") or {}
    detail = kwargs.get("expression")
    suffix = f" ({detail})" if detail else ""
    return f"{_field(metadata, 'name')} {target}{suffix}".strip()


def _dependency_name(unique_id: str) -> str:
    """``model.fourthdown.fact_plays`` -> ``fact_plays``; sources keep their source name."""
    parts = unique_id.split(".")
    if parts[0] == "source":
        return ".".join(parts[2:])
    return parts[-1]


def sha256_file(path: Path) -> str:
    # Line endings normalised, so a Windows checkout hashes like the Linux image.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def project_sha256(project: Path | None = None) -> str:
    """One hash over the project inputs that are not models (see ``PROJECT_INPUTS``)."""
    root = project or project_dir()
    digest = hashlib.sha256()
    for name in PROJECT_INPUTS:
        entry = root / name
        files = sorted(entry.rglob("*")) if entry.is_dir() else [entry]
        for path in files:
            if path.is_file():
                digest.update(f"{path.relative_to(root).as_posix()}\n".encode())
                digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _models(manifest_path: Path) -> tuple[ModelRecord, ...]:
    """Every model and seed in the manifest dbt wrote, with its file hash and inputs."""
    project = project_dir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for node in manifest["nodes"].values():
        if node["resource_type"] not in {"model", "seed"} or node["package_name"] != "fourthdown":
            continue
        records.append(
            ModelRecord(
                name=node["name"],
                path=node["original_file_path"].replace("\\", "/"),
                schema=node["schema"],
                materialized=node["config"]["materialized"],
                sha256=sha256_file(project / node["original_file_path"]),
                reads=tuple(
                    sorted(_dependency_name(dep) for dep in node["depends_on"].get("nodes", []))
                ),
            )
        )
    return tuple(sorted(records, key=lambda record: (record.schema, record.name)))


def vars_args(weeks: Sequence[str] | None) -> list[str]:
    """``--vars`` handing ``fact_plays`` the weeks to rebuild; none means all of them."""
    return [] if weeks is None else ["--vars", json.dumps({"weeks": list(weeks)})]


def build_args(
    *, run_tests: bool = True, weeks: Sequence[str] | None = None, full_refresh: bool = False
) -> list[str]:
    """The ``dbt build`` arguments after ``build`` itself."""
    args = ["--full-refresh"] if full_refresh else []
    if not run_tests:
        # Tiny synthetic datasets (the unit tests') cannot pass league-wide checks.
        args += ["--exclude-resource-type", "test", "--exclude-resource-type", "unit_test"]
    return args + vars_args(weeks)


def build(
    paths: Paths,
    database: Path | str,
    *,
    run_tests: bool = True,
    weeks: Sequence[str] | None = None,
    target: str = DEFAULT_TARGET,
    full_refresh: bool = False,
) -> BuildOutcome:
    """``dbt build`` into ``database``: seeds, models, and (unless told not to) data tests.

    ``weeks`` makes ``fact_plays`` rebuild only those weeks' games (an incremental build
    on a copy of the live database; see ``warehouse.plan``). ``target`` picks the profile
    output: a DuckDB file (``dev``) or a Snowflake database (``snowflake``).

    Raises:
        DbtError: a model failed to build or a data test failed. dbt skips everything
            downstream of a failure, so a failed build never yields a partial warehouse
            that looks complete.
    """
    args = [
        "build",
        "--quiet",
        *build_args(run_tests=run_tests, weeks=weeks, full_refresh=full_refresh),
    ]
    result = invoke(paths, args, database=database, target=target)
    if result.exception is not None:
        raise DbtError(f"dbt build did not run: {result.exception}") from result.exception
    tests: list[TestResult] = []
    errors: list[str] = []
    for node_result in result.result.results:
        node = node_result.node
        status = str(node_result.status)
        if node.resource_type == "test":
            tests.append(
                TestResult(
                    name=_test_name(node),
                    status=status,
                    failures=int(node_result.failures or 0),
                    message=str(node_result.message or ""),
                )
            )
        if status in {"error", "fail"}:
            errors.append(f"{node.resource_type} {_test_name(node)}: {node_result.message}")
    if errors or not result.success:
        raise DbtError("dbt build failed:\n  " + "\n  ".join(errors or ["see the dbt log"]))
    manifest_path = artifacts_dir(paths, target) / "target" / "manifest.json"
    return BuildOutcome(models=_models(manifest_path), tests=tuple(tests))


def outcome_from_artifacts(target: Path) -> BuildOutcome:
    """What a finished ``dbt build`` wrote, read from its ``run_results.json`` and manifest.

    For builds dbt ran in another process (the orchestrator runs dbt through dagster-dbt).
    """
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((target / "run_results.json").read_text(encoding="utf-8"))
    tests = [
        TestResult(
            name=_test_name(manifest["nodes"][result["unique_id"]]),
            status=str(result["status"]),
            failures=int(result.get("failures") or 0),
            message=str(result.get("message") or ""),
        )
        for result in results["results"]
        if result["unique_id"].startswith("test.")
    ]
    return BuildOutcome(models=_models(target / "manifest.json"), tests=tuple(tests))


def parse_manifest() -> Path:
    """Parse the project into ``<project>/target/manifest.json`` and return its path.

    The orchestrator loads its dbt assets from this manifest. Parsing reads no data, so
    any data root serves; the image does it once at build time.
    """
    project = project_dir()
    target = project / "target"
    manifest = target / "manifest.json"
    sources = [project / "dbt_project.yml", project / "profiles.yml"]
    for folder in ("models", "macros", "seeds", "tests"):
        sources += [path for path in (project / folder).rglob("*") if path.is_file()]
    if manifest.exists() and manifest.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return manifest
    from fourthdown.config import data_paths

    paths = data_paths()
    with environment(paths):
        # The manifest is the graph Dagster builds from, so it must not depend on what this
        # machine has downloaded: parse as if the players file exists, so stg_players
        # always reads the players source (a fresh checkout or CI has no data).
        os.environ["FOURTHDOWN_PLAYERS_AVAILABLE"] = "true"
        os.environ["DBT_TARGET_PATH"] = str(target)
        os.environ["DBT_LOG_PATH"] = str(target / "logs")
        from dbt.cli.main import dbtRunner

        try:
            result = dbtRunner().invoke(["parse", "--quiet"])
        finally:
            _release_log_files()
    if not result.success:
        raise DbtError(f"dbt parse failed: {result.exception}")
    return manifest


def save_test_results(paths: Paths, tests: Sequence[TestResult]) -> Path:
    """Keep a build's test results beside the warehouse, for ``fourthdown audit``."""
    target = paths.warehouse / TEST_RESULTS_FILE
    target.write_text(json.dumps([asdict(test) for test in tests], indent=2) + "\n")
    return target


def load_test_results(paths: Paths) -> list[TestResult] | None:
    """The data tests of the last tested build, or None when no build ran them."""
    source = paths.warehouse / TEST_RESULTS_FILE
    if not source.exists():
        return None
    return [TestResult(**entry) for entry in json.loads(source.read_text(encoding="utf-8"))]
