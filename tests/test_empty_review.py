"""The second look an empty result gets: filter probes, the review, and the miss log."""

from __future__ import annotations

import json

import pytest
from conftest import ScriptedClient
from typer.testing import CliRunner

from fourthdown import feedback
from fourthdown.cli import app
from fourthdown.evaluation import harness
from fourthdown.feedback import MissLog
from fourthdown.llm import LLMError
from fourthdown.rag.text_to_sql import (
    CORRECTION_FOUND_NOTHING,
    EXPECTED_EMPTY,
    UNREVIEWED,
    WRONG_QUERY,
    SQLAnswer,
    TextToSQL,
    parse_review,
)
from fourthdown.services.warehouse import StatsTool
from fourthdown.sql import filters

EMPTY_SQL = "SELECT play_id FROM plays WHERE season = 2023 AND season_type = 'POST'"
FIXED_SQL = "SELECT play_id FROM plays WHERE season = 2023 AND season_type = 'REG'"
STILL_EMPTY_SQL = "SELECT sum(yards) FROM player_game WHERE season = 2023 AND week = 99"


def review(verdict: str, reason: str, sql: str | None = None) -> str:
    body = {"verdict": verdict, "reason": reason}
    if sql is not None:
        body["sql"] = sql
    return json.dumps(body)


@pytest.fixture()
def total(views_connection) -> int:
    return int(views_connection.execute("SELECT count(*) FROM plays").fetchone()[0])


@pytest.fixture()
def miss_log(tmp_path) -> MissLog:
    return MissLog(tmp_path)


def logged(log: MissLog) -> list[dict]:
    return list(feedback.read(log.path))


# --- filter probes -------------------------------------------------------------------


def test_conditions_split_on_and_and_keep_an_or_group_whole():
    found = filters.conditions(
        "SELECT * FROM plays WHERE season = 2023 AND (posteam = 'KC' OR defteam = 'KC') "
        "AND down = 4"
    )
    assert [condition.sql() for condition in found] == [
        "season = 2023",
        "(posteam = 'KC' OR defteam = 'KC')",
        "down = 4",
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM plays",
        "SELECT 1 FROM plays WHERE season = 1 UNION ALL SELECT 2 FROM plays WHERE season = 2",
        "not sql at all (",
    ],
)
def test_nothing_to_probe(sql):
    assert filters.conditions(sql) == []


def test_probe_names_the_filter_that_emptied_the_query(views_connection, total):
    report = filters.probe(views_connection, EMPTY_SQL)
    assert report is not None
    assert report.rows_with_all == 0
    by_condition = {probe.condition: probe.rows_without for probe in report.probes}
    assert by_condition == {"season = 2023": 0, "season_type = 'POST'": total}
    assert f"without season_type = 'POST': {total} rows" in report.render()


def test_probe_counts_source_rows_under_grouping_and_limits(views_connection, total):
    report = filters.probe(
        views_connection,
        "SELECT posteam, count(*) FROM plays WHERE week = 99 AND season = 2023 "
        "GROUP BY posteam HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 3",
    )
    assert report is not None
    assert report.rows_with_all == 0
    assert report.probes[0].rows_without == total


def test_a_probe_that_fails_is_reported_not_raised(views_connection):
    report = filters.probe(views_connection, "SELECT 1 FROM plays WHERE no_such_column = 1")
    assert report is not None
    assert report.rows_with_all is None
    assert "query failed" in report.render()


# --- parsing the verdict ---------------------------------------------------------------


def test_parse_review_reads_both_verdicts_and_tolerates_chatter():
    assert parse_review(review(EXPECTED_EMPTY, "none")) == (EXPECTED_EMPTY, "none", None)
    chatty = "Sure, here it is:\n```json\n" + review(WRONG_QUERY, "bad", "SELECT 1") + "\n```"
    assert parse_review(chatty) == (WRONG_QUERY, "bad", "SELECT 1")


@pytest.mark.parametrize(
    "reply",
    [
        "no json here",
        "{not json}",
        '{"verdict": "maybe", "reason": "?"}',
        '{"verdict": "wrong_query", "reason": "but no query"}',
    ],
)
def test_parse_review_rejects_what_it_cannot_act_on(reply):
    with pytest.raises(ValueError):
        parse_review(reply)


# --- the review in the chain ---------------------------------------------------------------


def test_rows_mean_no_review_and_no_record(views_connection, miss_log):
    client = ScriptedClient(FIXED_SQL)
    answer = TextToSQL(views_connection, client, miss_log=miss_log).answer("plays in 2023?")
    assert answer.rows and answer.review is None
    assert len(client.prompts) == 1
    assert logged(miss_log) == []


def test_an_expected_empty_answer_says_why(views_connection, miss_log, total):
    client = ScriptedClient(EMPTY_SQL, review(EXPECTED_EMPTY, "There were no 2023 playoff plays."))
    answer = TextToSQL(views_connection, client, miss_log=miss_log).answer("2023 playoff plays?")
    assert answer.review is not None and answer.review.verdict == EXPECTED_EMPTY
    assert answer.note == "There were no 2023 playoff plays."
    # The reviewer saw which filter emptied the query.
    assert f"without season_type = 'POST': {total} rows" in client.prompts[1]
    (record,) = logged(miss_log)
    assert record["verdict"] == EXPECTED_EMPTY
    assert record["question"] == "2023 playoff plays?"
    assert record["filters"]["probes"]["season_type = 'POST'"] == total


def test_a_wrong_query_is_replaced_by_its_correction(views_connection, miss_log, total):
    client = ScriptedClient(
        EMPTY_SQL, review(WRONG_QUERY, "The question asked about the regular season.", FIXED_SQL)
    )
    chain = TextToSQL(views_connection, client, miss_log=miss_log)
    answer = chain.answer("plays in 2023?")
    assert answer.ok
    assert len(answer.rows) == total
    assert "'REG'" in (answer.sql or "")
    assert answer.review is not None and answer.review.revised
    assert "'POST'" in answer.review.original_sql
    assert answer.note == "Revised query: The question asked about the regular season."
    assert [attempt.error for attempt in answer.attempts] == [None, None]
    (record,) = logged(miss_log)
    assert record["verdict"] == WRONG_QUERY
    assert record["revised_rows"] == total
    assert "'POST'" in record["sql"]


def test_stats_tool_shows_the_revision_and_the_original(views_connection, total):
    client = ScriptedClient(EMPTY_SQL, review(WRONG_QUERY, "It is the regular season.", FIXED_SQL))
    evidence = StatsTool(TextToSQL(views_connection, client)).run("plays in 2023?")
    assert not evidence.failed
    assert evidence.table is not None and len(evidence.table.rows) == total
    assert "Revised query: It is the regular season." in evidence.answer
    assert evidence.detail["review"] == WRONG_QUERY
    assert "'POST'" in evidence.detail["original_sql"]
    assert "'REG'" in evidence.detail["sql"]


def test_a_correction_that_finds_nothing_keeps_the_empty_answer(views_connection, miss_log):
    client = ScriptedClient(
        EMPTY_SQL, review(WRONG_QUERY, "week 99 does not exist", STILL_EMPTY_SQL)
    )
    answer = TextToSQL(views_connection, client, miss_log=miss_log).answer("plays in 2023?")
    assert answer.review is not None and not answer.review.revised
    assert "'POST'" in (answer.sql or "")
    assert answer.note == CORRECTION_FOUND_NOTHING
    (record,) = logged(miss_log)
    assert record["revised_rows"] == 0
    assert record["reason"] == "week 99 does not exist"


def test_a_correction_that_counts_zero_is_not_a_revision(views_connection, miss_log):
    zero = "SELECT count(*) AS plays FROM plays WHERE season = 2023 AND week = 99"
    client = ScriptedClient(EMPTY_SQL, review(WRONG_QUERY, "count them instead", zero))
    answer = TextToSQL(views_connection, client, miss_log=miss_log).answer("plays?")
    assert answer.review is not None and not answer.review.revised
    assert answer.review.revised_rows == 0
    assert "'POST'" in (answer.sql or "")
    assert answer.note == CORRECTION_FOUND_NOTHING


def test_an_unsafe_correction_never_runs(views_connection, miss_log):
    client = ScriptedClient(EMPTY_SQL, review(WRONG_QUERY, "drop it", "DROP VIEW plays"))
    answer = TextToSQL(views_connection, client, miss_log=miss_log).answer("plays?")
    assert answer.review is not None and answer.review.error is not None
    assert answer.review.revised_rows is None
    assert views_connection.execute("SELECT count(*) FROM plays").fetchone()[0] > 0
    assert logged(miss_log)[0]["error"]


def test_an_unusable_review_reply_leaves_the_answer_alone(views_connection, miss_log):
    client = ScriptedClient(EMPTY_SQL, "I think the query is fine.")
    answer = TextToSQL(views_connection, client, miss_log=miss_log).answer("plays?")
    assert answer.ok and answer.review is not None
    assert answer.review.verdict == UNREVIEWED
    assert answer.note is None
    assert logged(miss_log)[0]["verdict"] == UNREVIEWED


class FailingReviewer(ScriptedClient):
    """Writes the query, then fails like an overloaded provider when asked to review."""

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str:
        if self.prompts:
            raise LLMError("provider overloaded")
        return super().complete(system=system, prompt=prompt, temperature=temperature)


def test_a_failed_review_call_is_not_a_failed_answer(views_connection, miss_log):
    answer = TextToSQL(views_connection, FailingReviewer(EMPTY_SQL), miss_log=miss_log).answer(
        "plays?"
    )
    assert answer.ok
    assert answer.review is not None and answer.review.verdict == UNREVIEWED
    assert "overloaded" in (answer.review.error or "")


def test_with_review_off_an_empty_result_is_still_recorded(views_connection, miss_log):
    client = ScriptedClient(EMPTY_SQL)
    answer = TextToSQL(views_connection, client, review=False, miss_log=miss_log).answer("plays?")
    assert answer.review is None
    assert len(client.prompts) == 1
    (record,) = logged(miss_log)
    assert "verdict" not in record
    assert record["filters"]["rows_with_all"] == 0


# --- the miss log ------------------------------------------------------------------------


def test_an_unwritable_log_does_not_break_the_answer(views_connection, tmp_path):
    log = MissLog(tmp_path / "missing" / "dir")
    client = ScriptedClient(EMPTY_SQL, review(EXPECTED_EMPTY, "none"))
    assert TextToSQL(views_connection, client, miss_log=log).answer("plays?").ok
    assert not log.path.exists()


def test_read_skips_a_corrupt_line(tmp_path):
    log = MissLog(tmp_path)
    log.record({"question": "first"})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated\n")
    log.record({"question": "second"})
    assert [record["question"] for record in feedback.read(log.path)] == ["first", "second"]


def test_misses_command_lists_and_filters(tmp_path):
    log = MissLog(tmp_path / "feedback")
    log.path.parent.mkdir()
    log.record({"question": "q1", "verdict": EXPECTED_EMPTY, "reason": "none", "sql": "SELECT 1"})
    log.record({"question": "q2", "verdict": WRONG_QUERY, "reason": "fixed", "revised_rows": 3})
    runner = CliRunner()
    every = runner.invoke(app, ["misses", "--data-dir", str(tmp_path), "--sql"])
    assert every.exit_code == 0, every.output
    assert "[expected_empty]  q1" in every.output
    assert "sql: SELECT 1" in every.output
    assert "[wrong_query, revised (3 rows)]  q2" in every.output
    only = runner.invoke(app, ["misses", "--data-dir", str(tmp_path), "--verdict", WRONG_QUERY])
    assert "q1" not in only.output and "q2" in only.output


def test_misses_command_on_an_empty_log(tmp_path):
    result = CliRunner().invoke(app, ["misses", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no empty results recorded" in result.output


def test_removing_a_filter_keeps_an_or_group_intact(views_connection, total):
    report = filters.probe(
        views_connection,
        "SELECT play_id FROM plays WHERE (posteam = 'AAA' OR posteam = 'BBB') AND season = 1999",
    )
    assert report is not None
    assert report.probes[1].condition == "season = 1999"
    assert report.probes[1].rows_without == total


# --- golden questions whose answer is nothing ---------------------------------------------


@pytest.mark.parametrize(
    ("rows", "correct"),
    [([], True), ([(None,)], True), ([(0,)], True), ([(3,)], False), ([("KC", 0)], False)],
)
def test_expect_empty_accepts_nothing_nulls_and_zeros_only(views_connection, rows, correct):
    question = harness.GoldenQuestion(
        id="q", question="q", tags=("expected-empty",), reference_sql="SELECT 1", expect_empty=True
    )
    answer = SQLAnswer(question="q", sql="SELECT 1", columns=("x",), rows=rows)
    assert harness.grade(views_connection, question, answer).correct is correct


def test_the_golden_set_carries_expected_empty_questions():
    empty = [question for question in harness.load_questions() if question.expect_empty]
    assert len(empty) >= 3
    assert all(question.reference_sql for question in empty)
