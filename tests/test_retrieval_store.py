"""Store tests against a real pgvector Postgres.

Nothing here is mocked: the whole point of the store is the SQL -- generated tsvectors,
the vector cast, and the rank fusion -- and a fake connection would test none of it.
The suite skips when no server is reachable, and runs in CI through the pgvector service.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from conftest import TEST_EMBEDDER as EMBEDDER

from fourthdown.narrative.render import Document
from fourthdown.retrieval.store import DocumentStore, StoreError

pytestmark = pytest.mark.postgres

DOCUMENTS = [
    Document(
        doc_id="game:2014_21_NE_SEA",
        grain="game",
        game_id="2014_21_NE_SEA",
        season=2014,
        week=21,
        teams=("NE", "SEA"),
        title="2014 Super Bowl XLIX (49): New England Patriots at Seattle Seahawks",
        body="New England Patriots beat Seattle Seahawks by 4 after an interception at the "
        "goal line.",
    ),
    Document(
        doc_id="game:2013_21_SEA_DEN",
        grain="game",
        game_id="2013_21_SEA_DEN",
        season=2013,
        week=21,
        teams=("SEA", "DEN"),
        title="2013 Super Bowl XLVIII (48): Seattle Seahawks at Denver Broncos",
        body="Seattle Seahawks beat Denver Broncos by 35 in a rout.",
    ),
    Document(
        doc_id="drive:2014_21_NE_SEA_23",
        grain="drive",
        game_id="2014_21_NE_SEA",
        season=2014,
        week=21,
        teams=("SEA", "NE"),
        title="2014 Super Bowl XLIX (49), Seattle Seahawks drive 23 against New England",
        body="Fourth quarter: Seattle Seahawks started at their own 20. Result: interception.",
    ),
]


def _embedded(documents: Sequence[Document]) -> list[tuple[Document, list[float]]]:
    vectors = EMBEDDER.encode([document.text for document in documents])
    return list(zip(documents, vectors, strict=True))


@pytest.fixture()
def loaded(store: DocumentStore) -> DocumentStore:
    store.upsert(_embedded(DOCUMENTS))
    return store


def _search(store: DocumentStore, question: str, **kwargs: object) -> list[str]:
    embedding = EMBEDDER.encode([question])[0]
    hits = store.search(question, embedding, **kwargs)  # type: ignore[arg-type]
    return [hit.doc_id for hit in hits]


def test_initialise_records_the_embedding_space(store: DocumentStore) -> None:
    assert store.embedding_space() == (EMBEDDER.name, EMBEDDER.dimensions)


def test_reindexing_with_a_different_model_needs_reset(store: DocumentStore) -> None:
    with pytest.raises(StoreError, match="Re-index with --reset"):
        store.initialise(model="other-model", dimensions=8)
    store.initialise(model="other-model", dimensions=8, reset=True)
    assert store.embedding_space() == ("other-model", 8)


def test_upsert_is_idempotent_and_updates_in_place(store: DocumentStore) -> None:
    store.upsert(_embedded(DOCUMENTS))
    store.upsert(_embedded(DOCUMENTS))
    assert store.count() == len(DOCUMENTS)

    rewritten = {**DOCUMENTS[0].__dict__, "body": "Rewritten body about a goal line stand."}
    revised = Document(**rewritten)
    store.upsert(_embedded([revised]))
    assert store.count() == len(DOCUMENTS)
    hits = _search(store, "goal line stand", grain="game")
    assert hits[0] == revised.doc_id


def test_counts_and_coverage_by_grain(loaded: DocumentStore) -> None:
    assert loaded.count() == 3
    assert loaded.count(grain="drive") == 1
    assert loaded.coverage() == [("drive", 1, 2014, 2014), ("game", 2, 2013, 2014)]


def test_lexical_search_finds_an_exact_phrase(loaded: DocumentStore) -> None:
    embedding = EMBEDDER.encode(["Denver Broncos rout"])[0]
    hits = loaded.search("Denver Broncos rout", embedding, mode="lexical")
    assert hits[0].doc_id == "game:2013_21_SEA_DEN"
    assert hits[0].found_by == "lexical"
    assert all(hit.dense_rank is None for hit in hits)


def test_dense_search_returns_everything_ranked(loaded: DocumentStore) -> None:
    embedding = EMBEDDER.encode(["Seahawks"])[0]
    hits = loaded.search("Seahawks", embedding, mode="dense")
    assert len(hits) == len(DOCUMENTS), "dense search has no @@ filter, so nothing drops out"
    assert all(hit.lexical_rank is None for hit in hits)


def test_hybrid_fuses_both_lists(loaded: DocumentStore) -> None:
    embedding = EMBEDDER.encode(["Seattle Seahawks interception"])[0]
    hits = loaded.search("Seattle Seahawks interception", embedding)
    assert any(hit.found_by == "both" for hit in hits)
    assert hits == sorted(hits, key=lambda hit: -hit.score)
    # A document in both lists must outscore the same rank in only one of them.
    both = next(hit for hit in hits if hit.found_by == "both")
    assert both.score > 1 / (60 + 1)


def test_filters_restrict_the_candidate_pool(loaded: DocumentStore) -> None:
    assert _search(loaded, "Super Bowl", grain="drive") == ["drive:2014_21_NE_SEA_23"]
    assert _search(loaded, "Super Bowl", season=2013) == ["game:2013_21_SEA_DEN"]
    assert set(_search(loaded, "Super Bowl", team="DEN")) == {"game:2013_21_SEA_DEN"}
    assert _search(loaded, "Super Bowl", season=1999) == []


def test_k_limits_the_result_count(loaded: DocumentStore) -> None:
    assert len(_search(loaded, "Super Bowl", k=1)) == 1


def test_unknown_mode_is_rejected(loaded: DocumentStore) -> None:
    with pytest.raises(ValueError, match="unknown search mode"):
        loaded.search("x", EMBEDDER.encode(["x"])[0], mode="bm25")
