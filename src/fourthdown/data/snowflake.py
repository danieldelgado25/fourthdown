"""Snowflake: the same dbt project, built in Snowflake (phase 11).

DuckDB stays the warehouse the services read. Snowflake is a second target for the same
models, to show the modelling is not tied to one engine:

1. **Load.** The processed weekly Parquet partitions are PUT to an internal stage and
   COPY'd into ``NFLVERSE.PROCESSED_PLAYS``, a table with the processed contract's columns
   and types. Only weeks whose partition hash differs from what Snowflake last loaded go
   up (``NFLVERSE.LOAD_STATE`` records it), and each file's loaded row count must match the
   row count the ETL recorded, or the build stops.
2. **Build.** ``dbt build --target prod``: the same models, macros, seed, and data
   tests as the DuckDB build, with the same incremental ``fact_plays`` (only the loaded
   weeks' games are rebuilt).
3. **Swap.** All of it happens in a zero-copy clone of the live database
   (``<DATABASE>_BUILDING``), swapped with the live one (``ALTER DATABASE ... SWAP WITH``,
   atomic) only when every data test passed. A failed build drops the clone; the live
   database never sees a partial load or a failed model. That is the DuckDB build's
   staging file, in Snowflake's terms.

``parity`` then compares the two builds measure by measure. ``compile_offline`` renders
the whole project for Snowflake without an account, so CI checks the Snowflake SQL on
every push.

Credentials come from the environment only (``SNOWFLAKE_*``, see ``.env.example``) and are
never logged or written to disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fourthdown.config import Paths, parse_week_key
from fourthdown.data import modeling, provenance, schema

LOGGER = logging.getLogger(__name__)

SOURCE_SCHEMA = "NFLVERSE"
"""Where the loaded sources live: dbt's source ``nflverse`` (its default schema name)."""
PLAYS_TABLE = f"{SOURCE_SCHEMA}.PROCESSED_PLAYS"
PLAYERS_TABLE = f"{SOURCE_SCHEMA}.PLAYERS"
STATE_TABLE = f"{SOURCE_SCHEMA}.LOAD_STATE"
STAGE = f"{SOURCE_SCHEMA}.PARTITIONS"
FILE_FORMAT = f"{SOURCE_SCHEMA}.PARQUET"
PLAYERS_KEY = "players"
DBT_KEY = "dbt"
BUILD_REPORT = "snowflake_build.json"
PLAYER_COLUMNS = (("gsis_id", "VARCHAR"), ("display_name", "VARCHAR"), ("position", "VARCHAR"))


class SnowflakeError(RuntimeError):
    """The Snowflake build could not finish; the live database was left as it was."""


# --- settings ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Who to connect as and where to build, from ``SNOWFLAKE_*`` environment variables."""

    account: str
    user: str
    role: str = "FOURTHDOWN"
    warehouse: str = "FOURTHDOWN_WH"
    database: str = "FOURTHDOWN"
    private_key_path: str = ""
    private_key: str = field(default="", repr=False)
    private_key_passphrase: str = field(default="", repr=False)
    password: str = field(default="", repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        return cls(
            account=env.get("SNOWFLAKE_ACCOUNT", ""),
            user=env.get("SNOWFLAKE_USER", ""),
            role=env.get("SNOWFLAKE_ROLE") or cls.role,
            warehouse=env.get("SNOWFLAKE_WAREHOUSE") or cls.warehouse,
            database=(env.get("SNOWFLAKE_DATABASE") or cls.database).upper(),
            private_key_path=env.get("SNOWFLAKE_PRIVATE_KEY_PATH", ""),
            private_key=env.get("SNOWFLAKE_PRIVATE_KEY", ""),
            private_key_passphrase=env.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", ""),
            password=env.get("SNOWFLAKE_PASSWORD", ""),
        )

    def missing(self) -> list[str]:
        """The environment variables still needed to connect."""
        needed = [
            name
            for name, value in (("SNOWFLAKE_ACCOUNT", self.account), ("SNOWFLAKE_USER", self.user))
            if not value
        ]
        if not (self.private_key_path or self.private_key or self.password):
            needed.append(
                "SNOWFLAKE_PRIVATE_KEY_PATH (or SNOWFLAKE_PRIVATE_KEY, or SNOWFLAKE_PASSWORD)"
            )
        return needed

    @property
    def configured(self) -> bool:
        return not self.missing()

    @property
    def building(self) -> str:
        """The clone each build happens in, swapped with ``database`` when it passed."""
        return f"{self.database}_BUILDING"


def _identifier(name: str) -> str:
    """A database name from the environment, refused unless it is a plain identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name):
        raise SnowflakeError(f"{name!r} is not a plain Snowflake identifier")
    return name.upper()


# --- the session ---------------------------------------------------------------------------


class Session(Protocol):
    """The few things the build needs from a Snowflake connection (a fake in the tests)."""

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]: ...

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None: ...

    def close(self) -> None: ...


class ConnectorSession:
    """``Session`` over snowflake-connector-python."""

    def __init__(self, settings: Settings) -> None:
        import snowflake.connector

        if not settings.configured:
            raise SnowflakeError(
                "Snowflake is not configured; set " + ", ".join(settings.missing())
            )
        auth: dict[str, Any] = {}
        if settings.private_key_path or settings.private_key:
            auth["authenticator"] = "SNOWFLAKE_JWT"
            if settings.private_key_path:
                auth["private_key_file"] = settings.private_key_path
            else:
                auth["private_key"] = settings.private_key
            if settings.private_key_passphrase:
                auth["private_key_password"] = settings.private_key_passphrase
        else:
            auth["password"] = settings.password
        self._connection = snowflake.connector.connect(
            account=settings.account,
            user=settings.user,
            role=settings.role,
            warehouse=settings.warehouse,
            session_parameters={"QUERY_TAG": "fourthdown-load"},
            **auth,
        )

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        LOGGER.debug("snowflake: %s", sql.split("\n", 1)[0][:200])
        with self._connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall()) if cursor.description else []

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if rows:
            with self._connection.cursor() as cursor:
                cursor.executemany(sql, list(rows))

    def close(self) -> None:
        self._connection.close()


@contextmanager
def session(settings: Settings) -> Iterator[Session]:
    opened = ConnectorSession(settings)
    try:
        yield opened
    finally:
        opened.close()


# --- the processed contract as a Snowflake table ------------------------------------------


def column_type(column: str) -> str:
    """The Snowflake type of a processed column, from its contract type."""
    if column in schema.INT_COLUMNS:
        return "INTEGER"
    if column in schema.FLOAT_COLUMNS:
        return "FLOAT"
    if column in (*schema.BOOL_COLUMNS, *schema.DERIVED_BOOL_COLUMNS):
        return "BOOLEAN"
    if column in schema.DATE_COLUMNS:
        return "DATE"
    if column in schema.STRING_COLUMNS:
        return "VARCHAR"
    raise KeyError(f"{column} has no contract type")


def plays_columns() -> list[tuple[str, str]]:
    return [(column, column_type(column)) for column in schema.PROCESSED_COLUMNS]


def _create_table(name: str, columns: Sequence[tuple[str, str]], *, replace: bool) -> str:
    verb = "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE IF NOT EXISTS"
    body = ",\n    ".join(f"{column} {kind}" for column, kind in columns)
    return f"{verb} {name} (\n    {body}\n)"


# --- what to load -------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadPlan:
    """The weeks to (re)load into Snowflake and the weeks to delete from it."""

    load: tuple[str, ...]
    drop: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not (self.load or self.drop)


def plan_load(local: Mapping[str, str], loaded: Mapping[str, str]) -> LoadPlan:
    """Weeks whose partition hash on disk differs from the one Snowflake last loaded, and
    weeks Snowflake holds that are no longer on disk."""
    return LoadPlan(
        load=tuple(sorted(key for key, digest in local.items() if loaded.get(key) != digest)),
        drop=tuple(sorted(set(loaded) - set(local))),
    )


def dbt_fingerprint() -> str:
    """One hash over every model and project input: a change means a full dbt rebuild."""
    project = modeling.project_dir()
    digest = hashlib.sha256(modeling.project_sha256().encode())
    for path in sorted((project / "models").rglob("*.sql")):
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(modeling.sha256_file(path).encode())
    return digest.hexdigest()


def _week_filter(keys: Sequence[str]) -> str:
    """``season * 100 + week IN (...)``: keys are validated ``YYYY-WW`` strings, so the
    numbers are safe to inline."""
    numbers = sorted(season * 100 + week for season, week in map(parse_week_key, keys))
    return f"season * 100 + week IN ({', '.join(str(n) for n in numbers)})"


def _file_url(path: Path) -> str:
    """A PUT source: ``file://`` and forward slashes, on Windows too."""
    url = "file://" + path.resolve().as_posix()
    if "'" in url:
        raise SnowflakeError(f"cannot PUT {path}: quote in the path")
    return url


def _staged_name(key: str) -> str:
    season, week = parse_week_key(key)
    return f"season={season}/week={week:02d}.parquet"


# --- the build -----------------------------------------------------------------------------


@dataclass
class SnowflakeBuild:
    """What one Snowflake build did, kept beside the warehouse as ``snowflake_build.json``."""

    database: str
    dbt: str
    reason: str
    weeks_loaded: list[str]
    weeks_dropped: list[str]
    rows_loaded: int
    players_loaded: bool
    tests_passed: int
    seconds: float
    built_at: str = ""

    def summary(self) -> str:
        loaded = (
            f"{len(self.weeks_loaded)} week(s) loaded ({self.rows_loaded:,} rows)"
            if self.weeks_loaded
            else "no week loaded"
        )
        dropped = f", {len(self.weeks_dropped)} dropped" if self.weeks_dropped else ""
        return (
            f"{self.database}: {loaded}{dropped}; {self.dbt} dbt build ({self.reason}), "
            f"{self.tests_passed} data tests passed, {self.seconds:.0f}s"
        )


def _database_exists(connection: Session, name: str) -> bool:
    return bool(connection.execute(f"SHOW DATABASES LIKE '{name}'"))


def _prepare(connection: Session) -> bool:
    """Create the source schema, stage, and tables in the build database.

    Returns True when ``PROCESSED_PLAYS`` was (re)created, i.e. holds nothing loaded: a new
    database, or a contract change (columns added, removed, or retyped).
    """
    connection.execute(f"CREATE SCHEMA IF NOT EXISTS {SOURCE_SCHEMA}")
    connection.execute(
        f"CREATE FILE FORMAT IF NOT EXISTS {FILE_FORMAT} TYPE = PARQUET USE_LOGICAL_TYPE = TRUE"
    )
    connection.execute(f"CREATE STAGE IF NOT EXISTS {STAGE} FILE_FORMAT = {FILE_FORMAT}")
    connection.execute(
        _create_table(
            STATE_TABLE,
            [
                ("key", "VARCHAR"),
                ("sha256", "VARCHAR"),
                ("row_count", "INTEGER"),
                ("loaded_at", "TIMESTAMP_LTZ"),
            ],
            replace=False,
        )
    )
    expected = [(name.upper(), kind) for name, kind in plays_columns()]
    found = connection.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema = '{SOURCE_SCHEMA}' AND table_name = 'PROCESSED_PLAYS' "
        "ORDER BY ordinal_position"
    )
    # information_schema reports INTEGER as NUMBER and VARCHAR as TEXT.
    aliases = {"INTEGER": "NUMBER", "VARCHAR": "TEXT"}
    if [(name, kind) for name, kind in found] == [
        (name, aliases.get(kind, kind)) for name, kind in expected
    ]:
        return False
    if found:
        LOGGER.warning("the processed contract changed; reloading every week")
    connection.execute(_create_table(PLAYS_TABLE, plays_columns(), replace=True))
    connection.execute(f"DELETE FROM {STATE_TABLE} WHERE key <> '{PLAYERS_KEY}'")
    return True


def _put(connection: Session, files: Sequence[Path], prefix: str) -> None:
    for path in files:
        connection.execute(
            f"PUT '{_file_url(path)}' '@{STAGE}/{prefix}' AUTO_COMPRESS = FALSE OVERWRITE = TRUE"
        )


def _copy(connection: Session, table: str, staged: Sequence[str]) -> dict[str, int]:
    """COPY the staged files into ``table``; returns rows loaded per staged file."""
    files = ", ".join(f"'{name}'" for name in staged)
    results = connection.execute(
        f"COPY INTO {table} FROM @{STAGE} FILES = ({files}) "
        "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE FORCE = TRUE PURGE = TRUE"
    )
    # COPY returns (file, status, rows_parsed, rows_loaded, ..., first_error, ...) per file;
    # the file is named relative to the stage, possibly behind the stage's own name.
    loaded: dict[str, int] = {}
    for row in results:
        if str(row[1]).upper() != "LOADED":
            raise SnowflakeError(f"COPY of {row[0]} failed: {row[1]} ({row[6]})")
        name = next((name for name in staged if str(row[0]).endswith(name)), str(row[0]))
        loaded[name] = int(row[3])
    return loaded


def _load_weeks(
    connection: Session, paths: Paths, plan: LoadPlan, records: Mapping[str, Any]
) -> int:
    """Delete the plan's weeks from ``PROCESSED_PLAYS`` and load the new ones' files."""
    touched = sorted({*plan.load, *plan.drop})
    if touched:
        connection.execute(f"DELETE FROM {PLAYS_TABLE} WHERE {_week_filter(touched)}")
        connection.execute(
            f"DELETE FROM {STATE_TABLE} WHERE key IN ({', '.join('%s' for _ in touched)})",
            touched,
        )
    if not plan.load:
        return 0
    staged = {}
    for key in plan.load:
        season, week = parse_week_key(key)
        path = paths.week_processed(season, week)
        _put(connection, [path], f"season={season}/")
        staged[_staged_name(key)] = key
    loaded = _copy(connection, PLAYS_TABLE, list(staged))
    mismatched = [
        f"{key}: {loaded.get(name, 0):,} rows loaded, {records[key].rows:,} written by the ETL"
        for name, key in staged.items()
        if loaded.get(name, 0) != records[key].rows
    ]
    if mismatched:
        raise SnowflakeError("rows lost in the load:\n  " + "\n  ".join(mismatched))
    connection.executemany(
        f"INSERT INTO {STATE_TABLE} (key, sha256, row_count, loaded_at) "
        "VALUES (%s, %s, %s, CURRENT_TIMESTAMP())",
        [(key, records[key].sha256, records[key].rows) for key in plan.load],
    )
    return sum(loaded.values())


def _load_players(connection: Session, paths: Paths, loaded: Mapping[str, str]) -> bool:
    """Reload nflverse's player table when its file changed. Returns whether it did."""
    if not paths.players.exists():
        return False
    digest = provenance.sha256_file(paths.players)
    if loaded.get(PLAYERS_KEY) == digest:
        return False
    connection.execute(_create_table(PLAYERS_TABLE, PLAYER_COLUMNS, replace=True))
    _put(connection, [paths.players], "players/")
    _copy(connection, PLAYERS_TABLE, ["players/players.parquet"])
    connection.execute(f"DELETE FROM {STATE_TABLE} WHERE key = %s", [PLAYERS_KEY])
    connection.executemany(
        f"INSERT INTO {STATE_TABLE} (key, sha256, row_count, loaded_at) "
        "VALUES (%s, %s, 0, CURRENT_TIMESTAMP())",
        [(PLAYERS_KEY, digest)],
    )
    return True


def build(
    paths: Paths,
    settings: Settings | None = None,
    *,
    full_refresh: bool = False,
    run_tests: bool = True,
    connect: Callable[[Settings], Any] = session,
) -> SnowflakeBuild:
    """Load what changed and ``dbt build`` in a clone of the live database; swap it in
    only if every data test passed.

    Raises:
        SnowflakeError: a load lost rows or a COPY failed.
        modeling.DbtError: a model failed to build or a data test failed.
        Either way the clone is dropped and the live database is as it was.
    """
    started = time.monotonic()
    resolved = settings or Settings.from_env()
    live, building = _identifier(resolved.database), _identifier(resolved.building)
    manifest = provenance.load(paths)
    if not manifest.processed:
        raise FileNotFoundError(
            f"no processed partitions recorded under {paths.processed}. Run `fourthdown etl` first."
        )
    with connect(resolved) as connection:
        existed = _database_exists(connection, live)
        connection.execute(f"DROP DATABASE IF EXISTS {building}")
        # Transient: everything here is rebuilt from the Parquet partitions, so the seven
        # days of Fail-safe storage a permanent database pays for would buy nothing. One day
        # of Time Travel stays, so `UNDROP DATABASE` can bring back the previous build.
        if existed and not full_refresh:
            connection.execute(f"CREATE TRANSIENT DATABASE {building} CLONE {live}")
        else:
            connection.execute(f"CREATE TRANSIENT DATABASE {building}")
        try:
            connection.execute(f"USE DATABASE {building}")
            recreated = _prepare(connection)
            state = dict(
                (str(key), str(digest))
                for key, digest in connection.execute(f"SELECT key, sha256 FROM {STATE_TABLE}")
            )
            weeks_loaded = {k: v for k, v in state.items() if k not in {PLAYERS_KEY, DBT_KEY}}
            load_plan = plan_load(
                {key: record.sha256 for key, record in manifest.processed.items()}, weeks_loaded
            )
            LOGGER.info(
                "snowflake: %d week(s) to load, %d to drop",
                len(load_plan.load),
                len(load_plan.drop),
            )
            rows = _load_weeks(connection, paths, load_plan, manifest.processed)
            players = _load_players(connection, paths, state)

            fingerprint = dbt_fingerprint()
            if not existed or full_refresh or recreated:
                incremental, reason = False, "full refresh" if full_refresh else "new database"
            elif state.get(DBT_KEY) != fingerprint:
                incremental, reason = False, "dbt project changed"
            elif load_plan.drop:
                incremental, reason = False, "weeks removed"
            else:
                incremental = True
                reason = f"{len(load_plan.load)} new or changed week(s)"
            outcome = modeling.build(
                paths,
                building,
                target=modeling.SNOWFLAKE_TARGET,
                run_tests=run_tests,
                weeks=load_plan.load if incremental else None,
                full_refresh=not incremental,
            )
            connection.execute(f"DELETE FROM {STATE_TABLE} WHERE key = %s", [DBT_KEY])
            connection.executemany(
                f"INSERT INTO {STATE_TABLE} (key, sha256, row_count, loaded_at) "
                "VALUES (%s, %s, 0, CURRENT_TIMESTAMP())",
                [(DBT_KEY, fingerprint)],
            )
            if existed:
                connection.execute(f"ALTER DATABASE {live} SWAP WITH {building}")
                connection.execute(f"DROP DATABASE {building}")
            else:
                connection.execute(f"ALTER DATABASE {building} RENAME TO {live}")
        except BaseException:
            LOGGER.error("snowflake build failed; dropping %s, %s is unchanged", building, live)
            connection.execute(f"DROP DATABASE IF EXISTS {building}")
            raise
    result = SnowflakeBuild(
        database=live,
        dbt="incremental" if incremental else "full",
        reason=reason,
        weeks_loaded=list(load_plan.load),
        weeks_dropped=list(load_plan.drop),
        rows_loaded=rows,
        players_loaded=players,
        tests_passed=sum(test.passed for test in outcome.tests),
        seconds=round(time.monotonic() - started, 1),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (paths.warehouse / BUILD_REPORT).write_text(json.dumps(asdict(result), indent=2) + "\n")
    LOGGER.info("snowflake: %s", result.summary())
    return result


# --- parity with the DuckDB build ---------------------------------------------------------

PARITY: tuple[tuple[str, str], ...] = (
    ("main.plays", "count(*)"),
    ("main.plays", "count(distinct game_id)"),
    ("main.plays", "sum(epa)"),
    ("main.plays", "sum(yards_gained)"),
    ("main.games", "count(*)"),
    ("main.games", "sum(home_score + away_score)"),
    ("main.drives", "count(*)"),
    ("main.drives", "sum(yards_gained)"),
    ("main.team_game", "count(*)"),
    ("main.team_game", "sum(designed_plays)"),
    ("main.team_game", "avg(pass_rate)"),
    ("main.player_game", "count(*)"),
    ("main.player_game", "sum(yards)"),
    ("main.players", "count(*)"),
    ("core.dim_player", "count(*)"),
    ("core.dim_player", "count(distinct short_name)"),
    ("core.dim_player", "sum(credited_plays)"),
    ("core.dim_season", "sum(plays)"),
    ("marts.mart_team_season_tendencies", "count(*)"),
    ("marts.mart_team_season_tendencies", "avg(neutral_pass_rate)"),
)
"""Measures both warehouses must agree on: row counts exactly, sums and means of floating
point values to a relative 1e-6 (Snowflake averages integers as fixed-point decimals, and
summation order differs between engines)."""

TOLERANCE = 1e-6


@dataclass(frozen=True)
class ParityRow:
    table: str
    measure: str
    duckdb: float | None
    snowflake: float | None

    @property
    def matches(self) -> bool:
        if self.duckdb is None or self.snowflake is None:
            return self.duckdb is None and self.snowflake is None
        return math.isclose(self.duckdb, self.snowflake, rel_tol=TOLERANCE, abs_tol=1e-9)


def parity(
    duckdb_scalar: Callable[[str], Any],
    snowflake_scalar: Callable[[str], Any],
    measures: Sequence[tuple[str, str]] = PARITY,
) -> list[ParityRow]:
    """Each measure computed by both warehouses; ``*_scalar`` run one SQL and return one value."""

    def number(value: Any) -> float | None:
        return None if value is None else float(value)

    rows = []
    for table, measure in measures:
        sql = f"SELECT {measure} FROM {table}"
        rows.append(
            ParityRow(table, measure, number(duckdb_scalar(sql)), number(snowflake_scalar(sql)))
        )
    return rows


def compare(
    paths: Paths,
    settings: Settings | None = None,
    *,
    connect: Callable[[Settings], Any] = session,
) -> list[ParityRow]:
    """``parity`` between this data root's DuckDB warehouse and the live Snowflake one."""
    from fourthdown.data import warehouse

    resolved = settings or Settings.from_env()
    live = _identifier(resolved.database)
    with warehouse.connect(paths) as duck, connect(resolved) as snow:
        snow.execute(f"USE DATABASE {live}")

        def duck_scalar(sql: str) -> Any:
            row = duck.execute(sql).fetchone()
            return row[0] if row else None

        def snow_scalar(sql: str) -> Any:
            rows = snow.execute(sql)
            return rows[0][0] if rows else None

        return parity(duck_scalar, snow_scalar)


def render_parity(rows: Sequence[ParityRow]) -> str:
    lines = ["| table | measure | DuckDB | Snowflake | |", "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        lines.append(
            f"| {row.table} | `{row.measure}` | {_fmt(row.duckdb)} | {_fmt(row.snowflake)} | "
            f"{'match' if row.matches else '**differs**'} |"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.6f}"


# --- compiling for Snowflake without an account ---------------------------------------

DUCKDB_ONLY = {
    r"\barg_(?:min|max)\s*\(": "arg_min/arg_max (use min_by/max_by)",
    r"\bfilter\s*\(\s*where\b": "aggregate FILTER (use count_if or a CASE inside the aggregate)",
    r"\bread_parquet\s*\(": "read_parquet (sources are tables on Snowflake)",
    r"\bstring_agg\s*\(": "string_agg (use listagg)",
    r"\blist\s*\(": "list() (use array_agg)",
}
OFFLINE_DATABASE = "FOURTHDOWN_OFFLINE_CHECK"


@contextmanager
def _offline_adapter(incremental: bool) -> Iterator[None]:
    """Stand in for the catalog queries dbt makes while compiling.

    dbt asks Snowflake which relations exist only to decide ``is_incremental()``. Answer
    "none" (the full build) or "``fact_plays`` exists" (the incremental build), so both of
    its SQL paths render without a connection.
    """
    from dbt.adapters.snowflake.impl import SnowflakeAdapter

    original = SnowflakeAdapter.list_relations_without_caching

    def listed(self: Any, schema_relation: Any) -> list[Any]:
        if not incremental or str(schema_relation.schema).lower() != "core":
            return []
        return [
            self.Relation.create(
                # As Snowflake reports unquoted names: upper case.
                database=str(schema_relation.database).upper(),
                schema="CORE",
                identifier="FACT_PLAYS",
                type="table",
            )
        ]

    SnowflakeAdapter.list_relations_without_caching = listed  # type: ignore[method-assign]
    try:
        yield
    finally:
        SnowflakeAdapter.list_relations_without_caching = original  # type: ignore[method-assign]


def compile_offline(paths: Paths, *, weeks: Sequence[str] | None = None) -> tuple[Path, list[str]]:
    """Render every model and test for Snowflake, without connecting, and check the SQL.

    ``weeks`` renders ``fact_plays``' incremental path for those weeks. Returns the
    compiled directory and the problems found: SQL Snowflake's parser (sqlglot's) rejects,
    DuckDB-only functions, or a database name left in a reference (which would break the
    clone-and-swap build).
    """
    variables = {
        "SNOWFLAKE_ACCOUNT": os.environ.get("SNOWFLAKE_ACCOUNT") or "offline-check",
        "SNOWFLAKE_USER": os.environ.get("SNOWFLAKE_USER") or "offline-check",
    }
    previous = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    target = "snowflake-offline"
    try:
        with _offline_adapter(incremental=weeks is not None):
            result = modeling.invoke(
                paths,
                [
                    "compile",
                    "--no-introspect",
                    "--no-populate-cache",
                    "--quiet",
                    "--target",
                    modeling.SNOWFLAKE_TARGET,
                    *modeling.vars_args(weeks),
                ],
                database=OFFLINE_DATABASE,
                target=target,
            )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not result.success:
        raise modeling.DbtError(f"dbt compile for Snowflake failed: {result.exception}")
    compiled = modeling.artifacts_dir(paths, target) / "target" / "compiled" / "fourthdown"
    return compiled, check_compiled(compiled)


def check_compiled(compiled: Path) -> list[str]:
    """Problems in compiled Snowflake SQL; see ``compile_offline``."""
    import sqlglot
    from sqlglot.errors import ParseError

    problems: list[str] = []
    for path in sorted(compiled.rglob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        name = path.relative_to(compiled).as_posix()
        try:
            sqlglot.parse(sql, read="snowflake")
        except ParseError as error:
            problems.append(f"{name}: Snowflake would not parse it ({str(error).splitlines()[0]})")
        lowered = sql.lower()
        for pattern, label in DUCKDB_ONLY.items():
            if re.search(pattern, lowered):
                problems.append(f"{name}: DuckDB-only {label}")
        if OFFLINE_DATABASE.lower() in lowered:
            problems.append(f"{name}: a reference names its database (breaks the swap)")
    return problems
