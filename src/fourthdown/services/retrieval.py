"""The retrieval service: narrative answers from the pgvector index.

Tools:
    ``narrative`` -- hybrid (dense + lexical) search over game and drive narratives,
    then an LLM answer that must cite the retrieved passages.

Endpoints: `common.tool_service`'s ``/health`` and ``/tools/narrative``.

Dependencies: the ``retrieval`` extra (psycopg). Reads ``FOURTHDOWN_PG_DSN``,
``OLLAMA_HOST``, ``FOURTHDOWN_LLM_MODEL``, ``FOURTHDOWN_EMBED_MODEL``. Default port 8002.

Run: ``python -m fourthdown.services.retrieval`` or ``fourthdown serve retrieval``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack

from fourthdown.agent.tools import Evidence, Passage
from fourthdown.llm import LLMClient, LLMError, OllamaClient
from fourthdown.retrieval import recall
from fourthdown.retrieval.embed import Embedder, EmbeddingError, default_embedder
from fourthdown.retrieval.recall import Retriever
from fourthdown.retrieval.store import DocumentStore, Hit, StoreError
from fourthdown.services.common import (
    Capabilities,
    Reloading,
    configure_logging,
    llm_available,
    serve,
    tool_service,
)

NAME = "retrieval"
PORT = 8002


def passages_from(hits: Sequence[Hit]) -> list[Passage]:
    """Convert store hits into wire-format `Passage` citations, keeping their order."""
    return [
        Passage(title=hit.title, text=hit.body, score=hit.score, found_by=hit.found_by)
        for hit in hits
    ]


class NarrativeTool:
    """What happened in a game, answered from retrieved narratives with citations.

    Args:
        retriever: Hybrid search over the document store.
        client: The LLM that writes the cited answer.
        k: Passages retrieved per question.

    `run(question)` returns the answer with its passages; failed when nothing matched.
    """

    name = "narrative"
    description = (
        "What happened in a specific game, drive, or moment: comebacks, upsets, famous "
        "finishes, 'how did X lose to Y', 'tell me about the 2018 NFC championship'. "
        "Answers from generated game summaries and cites them."
    )

    def __init__(self, retriever: Retriever, client: LLMClient, *, k: int = 5) -> None:
        self._retriever = retriever
        self._client = client
        self._k = k

    def run(self, question: str) -> Evidence:
        """Retrieve, then answer from the passages; see the class docstring."""
        grounded = recall.answer(self._retriever, self._client, question, k=self._k)
        return Evidence(
            answer=grounded.answer,
            passages=passages_from(grounded.hits),
            failed=not grounded.hits,
        )


def opener(
    client: LLMClient | None = None, embedder: Embedder | None = None, dsn: str | None = None
) -> Callable[[ExitStack], Capabilities]:
    """Build the function `Reloading` calls to open this service's resources.

    Args:
        client: The answering LLM; an `OllamaClient` from the environment if ``None``.
        embedder: Query embedder; must match the one the index was built with.
        dsn: Postgres URL; ``FOURTHDOWN_PG_DSN`` when ``None``.

    Returns:
        A callable that connects to Postgres on the given stack and reports the
        ``retrieval`` and ``llm`` components. ``narrative`` needs both.
    """
    llm = client or OllamaClient()
    query_embedder = embedder or default_embedder()

    def open_(stack: ExitStack) -> Capabilities:
        state = Capabilities()
        state.components["llm"] = llm_available(llm)
        try:
            store = stack.enter_context(DocumentStore.open(dsn))
        except StoreError as error:
            state.components["retrieval"] = False
            state.skipped["retrieval"] = str(error)
            return state
        state.components["retrieval"] = True
        if state.components["llm"]:
            state.tools.append(NarrativeTool(Retriever(store, query_embedder), llm))
        else:
            state.skipped["narrative"] = "grounded answers need the LLM, which is unavailable"
        return state

    return open_


def main(host: str | None = None, port: int | None = None, model: str | None = None) -> None:
    """Connect to the index and serve until interrupted.

    Args:
        host: Bind address (see `common.serve`).
        port: Port (see `common.serve`); 8002 by default.
        model: Ollama model for answers; ``FOURTHDOWN_LLM_MODEL`` when ``None``.
    """
    configure_logging()
    resources = Reloading(opener(client=OllamaClient(model)))
    app = tool_service(NAME, resources, errors=(LLMError, StoreError, EmbeddingError))
    try:
        serve(app, host=host, port=port, default_port=PORT)
    finally:
        resources.close()


if __name__ == "__main__":
    main()
