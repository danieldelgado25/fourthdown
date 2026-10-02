"""The chain, driven by a scripted client so the tests do not need a model."""

from __future__ import annotations

import pytest
from conftest import ScriptedClient

from fourthdown.evaluation import harness
from fourthdown.rag import text_to_sql
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.sql import guard


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


def test_prompt_carries_worked_examples(views_connection) -> None:
    client = ScriptedClient("SELECT count(*) FROM plays")
    TextToSQL(views_connection, client).answer("How many plays are there?")
    assert "Examples:" in client.prompts[0]
    assert text_to_sql.EXEMPLARS[0][1] in client.prompts[0]


def test_every_exemplar_passes_the_guard_and_runs(views_connection) -> None:
    for _, sql in text_to_sql.EXEMPLARS:
        if sql == text_to_sql.UNANSWERABLE:
            continue
        views_connection.execute(guard.validate(sql).sql).fetchall()


def test_exemplars_are_not_golden_questions() -> None:
    golden = harness.load_questions() + harness.load_questions(holdout=True)
    references = {" ".join((q.reference_sql or "").lower().split()) for q in golden}
    for question, sql in text_to_sql.EXEMPLARS:
        assert question not in {q.question for q in golden}
        assert " ".join(sql.lower().split()) not in references


def test_repair_prompt_carries_every_failed_attempt(views_connection) -> None:
    client = ScriptedClient(
        "SELECT count(*) FROM nope",
        "SELECT count(*) FROM nada",
        "SELECT count(*) FROM plays",
    )
    TextToSQL(views_connection, client).answer("How many plays?")
    assert "FROM nope" in client.prompts[2]
    assert "FROM nada" in client.prompts[2]
    assert "Examples:" not in client.prompts[1]


def test_repair_names_the_view_a_borrowed_column_belongs_to(views_connection) -> None:
    client = ScriptedClient(
        "SELECT count(*) FROM drives WHERE field_zone = 'red_zone'",
        "SELECT count(*) FROM drives",
    )
    TextToSQL(views_connection, client).answer("How many red zone drives?")
    assert "field_zone is only a column of plays" in client.prompts[1]


def test_repair_says_when_a_column_exists_nowhere(views_connection) -> None:
    client = ScriptedClient(
        "SELECT count(*) FROM plays WHERE made_up_column = 1",
        "SELECT count(*) FROM plays",
    )
    TextToSQL(views_connection, client).answer("How many plays?")
    assert "made_up_column is not a column of any view" in client.prompts[1]
