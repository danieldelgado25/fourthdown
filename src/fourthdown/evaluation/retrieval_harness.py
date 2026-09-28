"""Score hybrid retrieval on a golden set of "which game was this" questions.

The unit of correctness is the *game*, not the document: for a drive question there are
often several defensible passages from the same game, and demanding one exact drive would
measure the question's phrasing rather than the retriever. So a hit is a returned document
whose ``game_id`` is one of the expected ones, and the metrics are the standard three:

- hit@1, the share of questions whose top passage is from the right game;
- recall@k, the share with a right-game passage anywhere in the top k;
- MRR, 1/rank of the first right-game passage, which distinguishes "second" from "fifth".

Each question is also scored with dense-only and lexical-only retrieval, because the claim
that hybrid beats either half is the one thing this harness exists to check.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib import resources

from fourthdown.retrieval.embed import Embedder
from fourthdown.retrieval.store import DocumentStore, Hit

LOGGER = logging.getLogger(__name__)

GOLDEN_FILE = "golden_retrieval.json"
DEFAULT_K = 5


@dataclass(frozen=True)
class RetrievalQuestion:
    id: str
    question: str
    tags: tuple[str, ...]
    grain: str | None
    expect_game_ids: frozenset[str]


@dataclass(frozen=True)
class Scored:
    """Where the first correct passage landed, for one retrieval mode."""

    rank: int | None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


@dataclass(frozen=True)
class QuestionResult:
    question: RetrievalQuestion
    hybrid: Scored
    dense: Scored
    lexical: Scored
    top_hit: Hit | None

    @property
    def detail(self) -> str:
        if self.hybrid.rank is None:
            got = self.top_hit.title if self.top_hit else "nothing"
            return f"missed; top passage was {got}"
        return f"rank {self.hybrid.rank}" + (f" ({self.top_hit.found_by})" if self.top_hit else "")


@dataclass(frozen=True)
class RetrievalReport:
    results: list[QuestionResult]
    k: int

    @property
    def total(self) -> int:
        return len(self.results)

    def hit_at_1(self, mode: str = "hybrid") -> float:
        return _mean(
            1.0 if self._scored(result, mode).rank == 1 else 0.0 for result in self.results
        )

    def recall(self, mode: str = "hybrid") -> float:
        return _mean(
            1.0 if self._scored(result, mode).rank is not None else 0.0 for result in self.results
        )

    def mrr(self, mode: str = "hybrid") -> float:
        return _mean(self._scored(result, mode).reciprocal_rank for result in self.results)

    def __iter__(self) -> Iterator[QuestionResult]:
        return iter(self.results)

    @staticmethod
    def _scored(result: QuestionResult, mode: str) -> Scored:
        return {"hybrid": result.hybrid, "dense": result.dense, "lexical": result.lexical}[mode]

    def render(self) -> str:
        lines = [
            "# Retrieval evaluation",
            "",
            f"- questions: {self.total}",
            f"- k: {self.k}",
            "",
            "| retrieval | hit@1 | recall@k | MRR |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {mode} | {self.hit_at_1(mode):.2f} | {self.recall(mode):.2f} | "
            f"{self.mrr(mode):.3f} |"
            for mode in ("hybrid", "dense", "lexical")
        ]
        lines += [
            "",
            "| question | hybrid | dense | lexical | detail |",
            "| --- | --- | --- | --- | --- |",
        ]
        for result in self.results:
            lines.append(
                f"| {result.question.id} | {_rank(result.hybrid)} | {_rank(result.dense)} | "
                f"{_rank(result.lexical)} | {result.detail} |"
            )
        return "\n".join(lines) + "\n"


def _rank(scored: Scored) -> str:
    return "-" if scored.rank is None else str(scored.rank)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def load_questions() -> list[RetrievalQuestion]:
    payload = json.loads(
        resources.files("fourthdown.evaluation").joinpath(GOLDEN_FILE).read_text(encoding="utf-8")
    )
    return [
        RetrievalQuestion(
            id=item["id"],
            question=item["question"],
            tags=tuple(item["tags"]),
            grain=item.get("grain"),
            expect_game_ids=frozenset(item["expect_game_ids"]),
        )
        for item in payload
    ]


def _first_correct(hits: Sequence[Hit], expected: frozenset[str]) -> Scored:
    for rank, hit in enumerate(hits, start=1):
        if hit.game_id in expected:
            return Scored(rank=rank)
    return Scored(rank=None)


def evaluate(
    store: DocumentStore,
    embedder: Embedder,
    *,
    k: int = DEFAULT_K,
    questions: Sequence[RetrievalQuestion] | None = None,
) -> RetrievalReport:
    """Run every golden question through hybrid, dense-only, and lexical-only search."""
    wanted = list(questions) if questions is not None else load_questions()
    results: list[QuestionResult] = []
    for question in wanted:
        embedding = embedder.encode([question.question])[0]
        searched = {
            mode: store.search(question.question, embedding, k=k, grain=question.grain, mode=mode)
            for mode in ("hybrid", "dense", "lexical")
        }
        hybrid = searched["hybrid"]
        results.append(
            QuestionResult(
                question=question,
                hybrid=_first_correct(hybrid, question.expect_game_ids),
                dense=_first_correct(searched["dense"], question.expect_game_ids),
                lexical=_first_correct(searched["lexical"], question.expect_game_ids),
                top_hit=hybrid[0] if hybrid else None,
            )
        )
        LOGGER.info("%s: %s", question.id, results[-1].detail)
    return RetrievalReport(results=results, k=k)
