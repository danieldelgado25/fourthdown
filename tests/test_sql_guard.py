"""Guardrails: everything a model might write that must never reach DuckDB."""

from __future__ import annotations

import pytest

from fourthdown.sql import guard

SAFE = "SELECT posteam, avg(epa) FROM plays WHERE season = 2023 GROUP BY posteam"


def test_accepts_a_plain_select() -> None:
    validated = guard.validate(SAFE)
    assert validated.tables == frozenset({"plays"})
    assert "LIMIT" in validated.sql


def test_accepts_a_join_across_views() -> None:
    validated = guard.validate(
        "SELECT t.team, g.temp FROM team_game t JOIN games g ON g.game_id = t.game_id"
    )
    assert validated.tables == frozenset({"team_game", "games"})


def test_accepts_a_cte_without_treating_it_as_a_table() -> None:
    validated = guard.validate(
        "WITH rates AS (SELECT posteam, avg(is_pass_call::INT) r FROM plays GROUP BY posteam) "
        "SELECT * FROM rates ORDER BY r DESC"
    )
    assert validated.tables == frozenset({"plays"})


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE plays",
        "DELETE FROM plays WHERE season = 2023",
        "INSERT INTO plays VALUES (1)",
        "UPDATE plays SET epa = 0",
        "CREATE TABLE scratch AS SELECT * FROM plays",
        "COPY plays TO '/tmp/out.csv'",
        "ATTACH '/tmp/other.duckdb'",
        "PRAGMA database_list",
    ],
)
def test_rejects_anything_that_is_not_a_read(sql: str) -> None:
    with pytest.raises(guard.UnsafeSQLError):
        guard.validate(sql)


def test_rejects_a_second_statement_hidden_behind_a_semicolon() -> None:
    with pytest.raises(guard.UnsafeSQLError, match="one statement"):
        guard.validate(f"{SAFE}; DROP TABLE plays")


def test_rejects_unknown_relations() -> None:
    with pytest.raises(guard.UnsafeSQLError, match="unknown table"):
        guard.validate("SELECT * FROM player_season")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_parquet('/etc/passwd')",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM plays WHERE 1 IN (SELECT glob('/*'))",
    ],
)
def test_rejects_filesystem_access(sql: str) -> None:
    with pytest.raises(guard.UnsafeSQLError):
        guard.validate(sql)


def test_rejects_a_query_that_reads_no_view() -> None:
    with pytest.raises(guard.UnsafeSQLError):
        guard.validate("SELECT 1")


def test_rejects_unparseable_text() -> None:
    with pytest.raises(guard.UnsafeSQLError):
        guard.validate("this is not sql at all )(")


def test_applies_the_default_limit() -> None:
    assert guard.validate(SAFE).limit == guard.DEFAULT_ROW_LIMIT


def test_keeps_a_tighter_limit_the_model_asked_for() -> None:
    validated = guard.validate(f"{SAFE} ORDER BY 2 DESC LIMIT 5")
    assert validated.limit == 5
    assert validated.sql.rstrip().endswith("LIMIT 5")


def test_tightens_a_limit_that_is_too_large() -> None:
    validated = guard.validate(f"{SAFE} LIMIT 100000", max_rows=50)
    assert validated.limit == 50
