"""Retrieve passages and have the LLM answer from them, with citations.

The prompt is deliberately strict: the model is told the passages are the only evidence
and that it must cite. That is the whole safety property of this half of the system --
the SQL half cannot hallucinate a number because the number comes from DuckDB, and this
half cannot hallucinate an event because an uncited claim is visible as one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from fourthdown.llm import LLMClient
from fourthdown.retrieval.embed import Embedder
from fourthdown.retrieval.store import DocumentStore, Hit

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an NFL analyst. You answer only from the numbered passages you are given, which \
are generated summaries of games and drives. Cite every claim with its passage number, \
like [2]. If the passages do not contain the answer, say so plainly instead of guessing. \
Answer in at most four sentences."""

PROMPT = """Passages:
{passages}

Question: {question}

Answer, citing passage numbers:"""


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    hits: list[Hit]

    def render(self) -> str:
        sources = "\n".join(
            f"[{index}] {hit.title} ({hit.found_by}, score {hit.score:.4f})"
            for index, hit in enumerate(self.hits, start=1)
        )
        return f"{self.answer}\n\nSources:\n{sources}"


def format_passages(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1))


class Retriever:
    """Embeds the question and runs hybrid search; the only entry point the CLI needs."""

    def __init__(self, store: DocumentStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(
        self,
        question: str,
        *,
        k: int = 5,
        grain: str | None = None,
        season: int | None = None,
        team: str | None = None,
    ) -> list[Hit]:
        embedding = self._embedder.encode([question])[0]
        return self._store.search(question, embedding, k=k, grain=grain, season=season, team=team)


def answer(
    retriever: Retriever,
    client: LLMClient,
    question: str,
    *,
    k: int = 5,
    grain: str | None = None,
    season: int | None = None,
    team: str | None = None,
) -> GroundedAnswer:
    hits = retriever.search(question, k=k, grain=grain, season=season, team=team)
    if not hits:
        return GroundedAnswer(
            question=question,
            answer="Nothing in the narrative index matches that question.",
            hits=[],
        )
    response = client.complete(
        system=SYSTEM_PROMPT,
        prompt=PROMPT.format(passages=format_passages(hits), question=question),
        temperature=0.1,
    )
    return GroundedAnswer(question=question, answer=response.strip(), hits=hits)
