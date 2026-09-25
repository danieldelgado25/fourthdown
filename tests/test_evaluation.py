"""The grader, and the golden set's own well-formedness."""

from __future__ import annotations

import pytest
from conftest import ScriptedClient

from fourthdown.evaluation import harness
from fourthdown.rag.text_to_sql import SQLAnswer
from fourthdown.sql import guard


def test_golden_set_loads_and_is_well_formed() -> None:
    questions = harness.load_questions()
    assert len(questions) >= 10
    assert len({question.id for question in questions}) == len(questions)
    for question in questions:
        assert question.question.strip()
        assert question.tags
        assert question.expect_unanswerable == (question.reference_sql is None)


def test_every_reference_query_passes_the_guard() -> None:
    for question in harness.load_questions():
        if question.reference_sql is None:
            continue
        guard.validate(question.reference_sql)


def test_matching_ignores_float_noise() -> None:
    assert harness.results_match([(0.61694,)], [(0.616941,)])
    assert not harness.results_match([(0.61,)], [(0.62,)])


def test_matching_accepts_a_scalar_answer_carrying_extra_columns() -> None:
    assert harness.results_match([("KC",)], [("KC", 0.63)])


def test_matching_accepts_a_ranking_with_its_metric() -> None:
    assert harness.results_match(
        [("KC",), ("BUF",)],
        [("KC", 0.12), ("BUF", 0.09)],
    )


def test_matching_rejects_a_different_ranking() -> None:
    assert not harness.results_match([("KC",), ("BUF",)], [("KC", 0.12), ("NYJ", 0.09)])


def test_matching_compares_ints_and_floats_alike() -> None:
    assert harness.results_match([(3,)], [(3.0,)])


@pytest.fixture()
def question() -> harness.GoldenQuestion:
    return harness.GoldenQuestion(
        id="plays-in-2023",
        question="How many plays were there in 2023?",
        tags=("counting",),
        reference_sql="SELECT count(*) FROM plays WHERE season = 2023",
    )


def _answer(rows: list[tuple[object, ...]]) -> SQLAnswer:
    return SQLAnswer(question="q", sql="SELECT 1 FROM plays", columns=("n",), rows=rows)


def test_grades_a_matching_result_correct(views_connection, question) -> None:
    result = harness.grade(views_connection, question, _answer([(4,)]))
    assert result.correct


def test_grades_a_wrong_result_incorrect(views_connection, question) -> None:
    result = harness.grade(views_connection, question, _answer([(99,)]))
    assert not result.correct
    assert "expected" in result.detail


def test_grades_a_declined_answerable_question_incorrect(views_connection, question) -> None:
    declined = SQLAnswer(question="q", sql=None, columns=(), rows=[], unanswerable=True)
    assert not harness.grade(views_connection, question, declined).correct


def test_grades_a_correctly_declined_question_correct(views_connection) -> None:
    trap = harness.GoldenQuestion(
        id="payroll",
        question="Which team had the highest payroll?",
        tags=("unanswerable",),
        reference_sql=None,
        expect_unanswerable=True,
    )
    declined = SQLAnswer(question="q", sql=None, columns=(), rows=[], unanswerable=True)
    assert harness.grade(views_connection, trap, declined).correct


def test_report_renders_a_scoreboard(views_connection, question) -> None:
    from fourthdown.rag.text_to_sql import TextToSQL

    chain = TextToSQL(views_connection, ScriptedClient("SELECT count(*) FROM plays"))
    report = harness.evaluate(views_connection, chain, [question])
    rendered = report.render()
    assert "correct result: 1/1" in rendered
    assert "plays-in-2023" in rendered
