"""Snowflake as a second dbt target: loading, the clone-and-swap build, parity, and the
offline compile. Everything here runs without a Snowflake account."""

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
import yaml

from fourthdown.data import etl, modeling, provenance, schema, snowflake
from fourthdown.data.modeling import BuildOutcome, DbtError, TestResult
from fourthdown.quality import contracts

LIVE, BUILDING = "FOURTHDOWN", "FOURTHDOWN_BUILDING"
SETTINGS = snowflake.Settings(account="acct", user="me", password="not-a-real-secret")


# --- a fake Snowflake: databases as dicts, statements recorded ---------------------------


class FakeSnowflake:
    """Just enough of Snowflake for the build: databases, the load state, PUT and COPY."""

    def __init__(self) -> None:
        self.databases: dict[str, dict] = {}
        self.current: str | None = None
        self.statements: list[str] = []
        self.staged: dict[str, int] = {}
        self.lose_rows = 0

    @contextmanager
    def connect(self, settings):
        yield self

    def _db(self) -> dict:
        assert self.current is not None
        return self.databases[self.current]

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if match := re.fullmatch(r"SHOW DATABASES LIKE '(\w+)'", sql):
            return [(match[1],)] if match[1] in self.databases else []
        if match := re.fullmatch(r"CREATE TRANSIENT DATABASE (\w+) CLONE (\w+)", sql):
            self.databases[match[1]] = copy.deepcopy(self.databases[match[2]])
        elif match := re.fullmatch(r"CREATE TRANSIENT DATABASE (\w+)", sql):
            self.databases[match[1]] = {"columns": [], "state": {}, "weeks": set()}
        elif match := re.fullmatch(r"DROP DATABASE (?:IF EXISTS )?(\w+)", sql):
            self.databases.pop(match[1], None)
        elif match := re.fullmatch(r"USE DATABASE (\w+)", sql):
            self.current = match[1]
        elif match := re.fullmatch(r"ALTER DATABASE (\w+) SWAP WITH (\w+)", sql):
            a, b = match[1], match[2]
            self.databases[a], self.databases[b] = self.databases[b], self.databases[a]
        elif match := re.fullmatch(r"ALTER DATABASE (\w+) RENAME TO (\w+)", sql):
            self.databases[match[2]] = self.databases.pop(match[1])
        elif sql.startswith("SELECT column_name"):
            return list(self._db()["columns"])
        elif sql.startswith(f"CREATE OR REPLACE TABLE {snowflake.PLAYS_TABLE}"):
            aliases = {"INTEGER": "NUMBER", "VARCHAR": "TEXT"}
            self._db()["columns"] = [
                (name.upper(), aliases.get(kind, kind)) for name, kind in snowflake.plays_columns()
            ]
            self._db()["weeks"] = set()
        elif sql.startswith(f"DELETE FROM {snowflake.STATE_TABLE} WHERE key <> "):
            state = self._db()["state"]
            for key in [k for k in state if k != snowflake.PLAYERS_KEY]:
                del state[key]
        elif sql.startswith(f"DELETE FROM {snowflake.STATE_TABLE} WHERE key"):
            for key in params:
                self._db()["state"].pop(key, None)
        elif sql.startswith(f"DELETE FROM {snowflake.PLAYS_TABLE} WHERE"):
            numbers = {int(n) for n in re.findall(r"\d{6}", sql)}
            self._db()["weeks"] = {
                key for key in self._db()["weeks"] if int(key.replace("-", "")) not in numbers
            }
        elif sql.startswith("SELECT key, sha256"):
            return list(self._db()["state"].items())
        elif match := re.match(r"PUT 'file://(.+?)' '@[\w.]+/(.*)'", sql):
            path = Path(match[1])
            self.staged[match[2] + path.name] = pl.read_parquet(path).height
        elif sql.startswith("COPY INTO"):
            files = re.findall(r"'([^']+)'", sql.split("FILES = ", 1)[1])
            rows = []
            for name in files:
                count = self.staged.pop(name) - self.lose_rows
                rows.append((f"partitions/{name}", "LOADED", count, count, 1, 0, None))
                if name.startswith("season="):
                    season, week = re.findall(r"\d+", name)
                    self._db()["weeks"].add(f"{season}-{week}")
            return rows
        return []

    def executemany(self, sql, rows):
        self.statements.append(sql)
        for key, digest, *_ in rows:
            self._db()["state"][key] = digest

    def close(self):
        pass


@pytest.fixture()
def fake():
    return FakeSnowflake()


class DbtRuns(list):
    """The arguments of each dbt build; ``fail = True`` makes the next one fail."""

    fail = False


@pytest.fixture()
def dbt_runs(monkeypatch):
    """Replaces dbt, which needs a real Snowflake, with a recorder."""
    runs = DbtRuns()

    def build(paths, database, **kwargs):
        if runs.fail:
            raise DbtError("dbt build failed:\n  test not_null fact_plays.down: 3 failures")
        runs.append({"database": database, **kwargs})
        return BuildOutcome(models=(), tests=(TestResult("not_null", "pass", 0, ""),))

    monkeypatch.setattr(modeling, "build", build)
    return runs


def _two_weeks(raw_frame: pl.DataFrame) -> pl.DataFrame:
    second = raw_frame.with_columns(
        pl.lit(2, dtype=raw_frame["week"].dtype).alias("week"),
        pl.col("game_id").str.replace("_01_", "_02_"),
    )
    return pl.concat([raw_frame, second])


@pytest.fixture()
def partitions(tmp_paths, raw_frame):
    _two_weeks(raw_frame).write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    return tmp_paths


def _build(paths, fake, **kwargs):
    return snowflake.build(paths, SETTINGS, connect=fake.connect, **kwargs)


# --- the build ------------------------------------------------------------------------------


def test_the_first_build_loads_every_week_and_becomes_the_live_database(partitions, fake, dbt_runs):
    result = _build(partitions, fake)
    assert set(fake.databases) == {LIVE}
    assert fake.databases[LIVE]["weeks"] == {"2023-01", "2023-02"}
    assert result.weeks_loaded == ["2023-01", "2023-02"]
    assert result.rows_loaded == 8
    assert (result.dbt, result.reason) == ("full", "new database")
    assert dbt_runs == [
        {
            "database": BUILDING,
            "target": "prod",
            "run_tests": True,
            "weeks": None,
            "full_refresh": True,
        }
    ]
    assert f"ALTER DATABASE {BUILDING} RENAME TO {LIVE}" in fake.statements
    assert (partitions.warehouse / snowflake.BUILD_REPORT).exists()


def test_nothing_new_loads_nothing_and_rebuilds_no_week(partitions, fake, dbt_runs):
    _build(partitions, fake)
    fake.statements.clear()
    result = _build(partitions, fake)
    assert result.weeks_loaded == [] and result.rows_loaded == 0
    assert dbt_runs[-1]["weeks"] == () and dbt_runs[-1]["full_refresh"] is False
    assert not any(sql.startswith(("PUT", "COPY")) for sql in fake.statements)
    assert f"CREATE TRANSIENT DATABASE {BUILDING} CLONE {LIVE}" in fake.statements


def test_a_changed_week_alone_is_reloaded_and_rebuilt(partitions, fake, dbt_runs, raw_frame):
    _build(partitions, fake)
    changed = _two_weeks(raw_frame).filter(~((pl.col("week") == 2) & (pl.col("play_id") == 4)))
    changed.write_parquet(partitions.season_raw(2023))
    etl.transform_week(2023, 2, partitions)
    fake.statements.clear()

    result = _build(partitions, fake)
    assert result.weeks_loaded == ["2023-02"] and result.rows_loaded == 3
    assert (result.dbt, dbt_runs[-1]["weeks"]) == ("incremental", ("2023-02",))
    assert (
        f"DELETE FROM {snowflake.PLAYS_TABLE} WHERE season * 100 + week IN (202302)"
        in fake.statements
    )
    assert [sql for sql in fake.statements if sql.startswith("COPY")][0].count(".parquet") == 1
    assert fake.statements[-2:] == [
        f"ALTER DATABASE {LIVE} SWAP WITH {BUILDING}",
        f"DROP DATABASE {BUILDING}",
    ]
    assert (
        fake.databases[LIVE]["state"]["2023-02"]
        == provenance.load(partitions).processed["2023-02"].sha256
    )


def test_a_failed_dbt_build_leaves_the_live_database_as_it_was(
    partitions, fake, dbt_runs, raw_frame
):
    _build(partitions, fake)
    before = copy.deepcopy(fake.databases[LIVE])
    raw_frame.with_columns(
        pl.lit(9, dtype=raw_frame["ydstogo"].dtype).alias("ydstogo")
    ).write_parquet(partitions.season_raw(2023))
    etl.transform_week(2023, 1, partitions)
    dbt_runs.fail = True
    with pytest.raises(DbtError, match="not_null"):
        _build(partitions, fake)
    assert set(fake.databases) == {LIVE}
    assert fake.databases[LIVE] == before
    assert not any("SWAP" in sql for sql in fake.statements)


def test_rows_lost_in_the_load_stop_the_build(partitions, fake, dbt_runs):
    fake.lose_rows = 1
    with pytest.raises(snowflake.SnowflakeError, match="2023-01: 3 rows loaded, 4 written"):
        _build(partitions, fake)
    assert fake.databases == {} and dbt_runs == []


def test_a_changed_dbt_project_rebuilds_every_model(partitions, fake, dbt_runs, monkeypatch):
    _build(partitions, fake)
    monkeypatch.setattr(snowflake, "dbt_fingerprint", lambda: "a different project")
    result = _build(partitions, fake)
    assert (result.dbt, result.reason) == ("full", "dbt project changed")
    assert dbt_runs[-1]["full_refresh"] is True and result.weeks_loaded == []


def test_a_contract_change_reloads_every_week(partitions, fake, dbt_runs):
    _build(partitions, fake)
    fake.databases[LIVE]["columns"] = fake.databases[LIVE]["columns"][:-1]
    result = _build(partitions, fake)
    assert result.weeks_loaded == ["2023-01", "2023-02"] and result.dbt == "full"


# --- pieces ---------------------------------------------------------------------------------


def test_plan_load_finds_new_changed_and_removed_weeks():
    plan = snowflake.plan_load(
        {"2026-09": "a", "2026-10": "b2", "2026-11": "c"},
        {"2026-09": "a", "2026-10": "b", "2026-08": "x"},
    )
    assert plan.load == ("2026-10", "2026-11")
    assert plan.drop == ("2026-08",)


def test_the_snowflake_table_types_match_the_processed_contract():
    expected = {
        pl.Int32: "INTEGER",
        pl.Float64: "FLOAT",
        pl.Boolean: "BOOLEAN",
        pl.Date: "DATE",
        pl.String: "VARCHAR",
    }
    columns = snowflake.plays_columns()
    assert [name for name, _ in columns] == list(schema.PROCESSED_COLUMNS)
    for name, kind in columns:
        assert kind == expected[contracts.DTYPES[name]], name


def test_settings_say_what_is_missing_and_never_show_secrets():
    empty = snowflake.Settings.from_env({})
    assert not empty.configured
    assert empty.missing() == [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PRIVATE_KEY_PATH (or SNOWFLAKE_PRIVATE_KEY, or SNOWFLAKE_PASSWORD)",
    ]
    configured = snowflake.Settings.from_env(
        {
            "SNOWFLAKE_ACCOUNT": "acct",
            "SNOWFLAKE_USER": "me",
            "SNOWFLAKE_PRIVATE_KEY_PATH": "/keys/rsa.p8",
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": "hunter2",
            "SNOWFLAKE_DATABASE": "nfl",
        }
    )
    assert configured.configured and configured.building == "NFL_BUILDING"
    assert "hunter2" not in repr(configured) and "not-a-real-secret" not in repr(SETTINGS)


def test_a_database_name_must_be_a_plain_identifier(partitions, fake, dbt_runs):
    hostile = snowflake.Settings(account="a", user="u", password="p", database="x; DROP DATABASE y")
    with pytest.raises(snowflake.SnowflakeError, match="not a plain Snowflake identifier"):
        snowflake.build(partitions, hostile, connect=fake.connect)


def test_the_snowflake_profile_takes_every_credential_from_the_environment():
    profile = yaml.safe_load((modeling.project_dir() / "profiles.yml").read_text())
    output = profile["fourthdown"]["outputs"]["prod"]
    secrets = (
        "account",
        "user",
        "password",
        "private_key_path",
        "private_key",
        "private_key_passphrase",
    )
    for key in secrets:
        assert re.fullmatch(r"\{\{ env_var\('SNOWFLAKE_[A-Z_]+', ''\) \}\}", output[key]), key
    assert output["type"] == "snowflake"
    assert "env_var('SNOWFLAKE_DATABASE'" in output["database"]


# --- parity ---------------------------------------------------------------------------------


def test_identical_warehouses_match_on_every_measure(synthetic_paths):
    from fourthdown.data import warehouse

    with warehouse.connect(synthetic_paths) as a, warehouse.connect(synthetic_paths) as b:
        rows = snowflake.parity(
            lambda sql: a.execute(sql).fetchone()[0], lambda sql: b.execute(sql).fetchone()[0]
        )
    assert len(rows) == len(snowflake.PARITY) and all(row.matches for row in rows)


def test_parity_tolerates_fixed_point_means_but_not_a_lost_row():
    values = {"SELECT count(*) FROM main.plays": (816_856, 816_855)}
    values["SELECT avg(pass_rate) FROM main.team_game"] = (0.5812345678901, Decimal("0.581235"))
    rows = snowflake.parity(
        lambda sql: values[sql][0],
        lambda sql: values[sql][1],
        measures=[("main.plays", "count(*)"), ("main.team_game", "avg(pass_rate)")],
    )
    assert [row.matches for row in rows] == [False, True]
    assert "**differs**" in snowflake.render_parity(rows)


# --- compiling for Snowflake without an account ----------------------------------------


def test_every_model_and_test_compiles_for_snowflake(tmp_paths):
    pytest.importorskip("dbt.adapters.snowflake")
    for weeks in (None, ["2026-10"]):
        compiled, problems = snowflake.compile_offline(tmp_paths, weeks=weeks)
        assert problems == []
        fact_plays = (compiled / "models" / "core" / "fact_plays.sql").read_text()
        assert "from staging.stg_plays" in fact_plays
        assert ("in ('2026-10')" in fact_plays) is (weeks is not None)
    stg = (compiled / "models" / "staging" / "stg_plays.sql").read_text()
    assert "from nflverse.processed_plays" in stg


def test_the_compiled_sql_check_catches_duckdb_only_sql(tmp_path):
    (tmp_path / "a.sql").write_text("select arg_min(x, y), count(*) filter (where z) from t")
    (tmp_path / "b.sql").write_text(f"select * from {snowflake.OFFLINE_DATABASE}.core.fact_plays")
    (tmp_path / "c.sql").write_text("select (1 from")
    problems = snowflake.check_compiled(tmp_path)
    assert any(p.startswith("a.sql: DuckDB-only arg_min") for p in problems)
    assert any(p.startswith("a.sql: DuckDB-only aggregate FILTER") for p in problems)
    assert any(p.startswith("b.sql: a reference names its database") for p in problems)
    assert any(p.startswith("c.sql: Snowflake would not parse it") for p in problems)


# Snowflake's reserved keywords (docs.snowflake.com, "Reserved & limited keywords").
RESERVED_WORDS = """
    ACCOUNT ALL ALTER AND ANY AS BETWEEN BY CASE CAST CHECK COLUMN CONNECT CONNECTION
    CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP CURRENT_USER
    DATABASE DELETE DISTINCT DROP ELSE EXISTS FALSE FOLLOWING FOR FROM FULL GRANT GROUP
    GSCLUSTER HAVING ILIKE IN INCREMENT INNER INSERT INTERSECT INTO IS ISSUE JOIN LATERAL LEFT
    LIKE LOCALTIME LOCALTIMESTAMP MINUS NATURAL NOT NULL OF ON OR ORDER ORGANIZATION QUALIFY
    REGEXP REVOKE RIGHT RLIKE ROW ROWS SAMPLE SCHEMA SELECT SET SOME START TABLE TABLESAMPLE
    THEN TO TRIGGER TRUE TRY_CAST UNION UNIQUE UPDATE USING VALUES VIEW WHEN WHENEVER WHERE
    WINDOW WITH"""
RESERVED = frozenset(RESERVED_WORDS.split())


def test_no_column_the_loader_creates_is_a_reserved_word(partitions, fake, dbt_runs):
    _build(partitions, fake)
    created = [sql for sql in fake.statements if sql.startswith("CREATE") and "TABLE" in sql]
    columns = {line.split()[0].upper() for sql in created for line in sql.splitlines()[1:-1]}
    assert columns and not columns & RESERVED


def test_every_statement_the_loader_sends_parses_as_snowflake_sql(partitions, fake, dbt_runs):
    import sqlglot

    _build(partitions, fake)
    for sql in fake.statements:
        if sql.startswith(("PUT ", "COPY ", "SHOW ", "CREATE FILE FORMAT", "CREATE STAGE")):
            continue  # Snowflake-only commands sqlglot doesn't model
        sqlglot.parse_one(sql.replace("%s", "'x'"), read="snowflake")
