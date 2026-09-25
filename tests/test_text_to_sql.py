"""The chain, driven by a scripted client so the tests do not need a model."""

from __future__ import annotations

import pytest
from conftest import ScriptedClient

from fourthdown.rag import text_to_sql
from fourthdown.rag.text_to_sql import TextToSQL


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SELECT 1 FROM plays", "SELECT 1 FROM plays"),
        ("```sql\nSELECT 1 FROM plays\n```", "SELECT 1 FROM plays"),
        ("Here is the query:\nSELECT 1 FROM plays;", "SELECT 1 FROM plays"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("UNANSWERABLE", "UNANSWERABLE"),
    ],
)
def test_extracts_sql_from_whatever_the_model_wraps_it_in(reply: str, expected: str) -> None:
    assert text_to_sql.extract_sql(reply) == expected


def test_answers_a_question_in_one_attempt(views_connection) -> None:
    client = ScriptedClient("SELECT count(*) FROM plays")
    answer = TextToSQL(views_connection, client).answer("How many plays are there?")
    assert answer.ok
    assert answer.rows == [(4,)]
    assert len(answer.attempts) == 1


def test_prompt_contains_the_schema_card(views_connection) -> None:
    client = ScriptedClient("SELECT count(*) FROM plays")
    chain = TextToSQL(views_connection, client)
    chain.answer("How many plays are there?")
    assert "is_designed_play" in client.prompts[0]


def test_repairs_sql_the_guard_rejects(views_connection) -> None:
    client = ScriptedClient(
        "SELECT count(*) FROM play_by_play",
        "SELECT count(*) FROM plays",
    )
    answer = TextToSQL(views_connection, client).answer("How many plays?")
    assert answer.ok
    assert len(answer.attempts) == 2
    assert "unknown table" in client.prompts[1]


def test_repairs_sql_duckdb_rejects(views_connection) -> None:
    client = ScriptedClient(
        "SELECT count(*) FROM plays WHERE nonexistent_column = 1",
        "SELECT count(*) FROM plays",
    )
    answer = TextToSQL(views_connection, client).answer("How many plays?")
    assert answer.ok
    assert answer.attempts[0].error is not None


def test_gives_up_after_the_attempt_budget(views_connection) -> None:
    client = ScriptedClient(*["SELECT * FROM nope"] * 3)
    answer = TextToSQL(views_connection, client, max_attempts=3).answer("How many plays?")
    assert not answer.ok
    assert len(answer.attempts) == 3
    assert answer.error is not None


def test_never_executes_a_write(views_connection) -> None:
    client = ScriptedClient(*["DROP TABLE plays"] * 2)
    answer = TextToSQL(views_connection, client, max_attempts=2).answer("Delete everything")
    assert not answer.ok
    assert views_connection.execute("SELECT count(*) FROM plays").fetchone() == (4,)


def test_passes_through_an_unanswerable_question(views_connection) -> None:
    client = ScriptedClient("UNANSWERABLE")
    answer = TextToSQL(views_connection, client).answer("Which team had the highest payroll?")
    assert answer.unanswerable
    assert not answer.ok


def test_repair_prompt_lists_the_columns_that_exist(views_connection) -> None:
    client = ScriptedClient(
        "SELECT avg(is_neutral_script) FROM team_game",
        "SELECT count(*) FROM plays",
    )
    TextToSQL(views_connection, client).answer("How neutral was it?")
    hint = client.prompts[1].split("Columns that exist:")[1]
    assert "team_game: " in hint
    assert "neutral_pass_rate" in hint
    assert "plays: " not in hint


def test_repairs_warm_up_so_the_model_does_not_repeat_itself(views_connection) -> None:
    client = ScriptedClient(*["SELECT * FROM nope"] * 3)
    TextToSQL(views_connection, client, max_attempts=3).answer("How many plays?")
    assert client.temperatures[0] == 0.0
    assert client.temperatures == sorted(client.temperatures)
    assert client.temperatures[-1] > 0.0


def test_caps_returned_rows(views_connection) -> None:
    client = ScriptedClient("SELECT * FROM plays")
    answer = TextToSQL(views_connection, client, max_rows=2).answer("Show me the plays")
    assert len(answer.rows) == 2
