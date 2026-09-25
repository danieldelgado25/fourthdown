from __future__ import annotations

import pytest
from conftest import TEST_EMBEDDER as EMBEDDER

from fourthdown.evaluation import retrieval_harness
from fourthdown.evaluation.retrieval_harness import RetrievalQuestion
from fourthdown.retrieval import index
from fourthdown.retrieval.store import DocumentStore

pytestmark = pytest.mark.postgres


def test_build_indexes_both_grains(views_connection, store: DocumentStore) -> None:
    report = index.build(views_connection, store, EMBEDDER, reset=True)
    assert report.documents == store.count() > 0
    grains = {grain for grain, _, _, _ in report.coverage}
    assert grains == {"game", "drive"}
    assert "documents in" in report.render()


def test_rebuilding_does_not_duplicate(views_connection, store: DocumentStore) -> None:
    first = index.build(views_connection, store, EMBEDDER, reset=True)
    index.build(views_connection, store, EMBEDDER)
    assert store.count() == first.documents


def test_build_can_select_a_grain_and_season(views_connection, store: DocumentStore) -> None:
    report = index.build(
        views_connection, store, EMBEDDER, grains=("game",), seasons=[2023], reset=True
    )
    assert report.documents == 1
    assert store.count(grain="drive") == 0


def test_indexing_a_missing_season_writes_nothing(views_connection, store: DocumentStore) -> None:
    report = index.build(views_connection, store, EMBEDDER, seasons=[1999], reset=True)
    assert report.documents == 0
    assert store.count() == 0


def test_retrieval_harness_scores_each_mode(views_connection, store: DocumentStore) -> None:
    index.build(views_connection, store, EMBEDDER, grains=("game",), reset=True)
    questions = [
        RetrievalQuestion(
            id="found",
            question="AAA at BBB in week 1 of 2023",
            tags=("game",),
            grain="game",
            expect_game_ids=frozenset({"2023_01_AAA_BBB"}),
        ),
        RetrievalQuestion(
            id="missed",
            question="AAA at BBB in week 1 of 2023",
            tags=("game",),
            grain="game",
            expect_game_ids=frozenset({"nonexistent"}),
        ),
    ]
    report = retrieval_harness.evaluate(store, EMBEDDER, questions=questions)

    assert report.total == 2
    assert report.hit_at_1() == 0.5
    assert report.recall() == 0.5
    assert report.mrr() == 0.5
    assert report.results[0].hybrid.rank == 1
    assert report.results[1].hybrid.rank is None
    assert "missed" in report.results[1].detail
    rendered = report.render()
    assert "| hybrid |" in rendered and "| dense |" in rendered and "| lexical |" in rendered


def test_golden_retrieval_set_is_well_formed() -> None:
    questions = retrieval_harness.load_questions()
    assert len(questions) >= 15
    assert len({question.id for question in questions}) == len(questions)
    for question in questions:
        assert question.question.strip()
        assert question.tags
        assert question.grain in {"game", "drive", None}
        assert question.expect_game_ids
