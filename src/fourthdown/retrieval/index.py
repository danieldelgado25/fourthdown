"""Warehouse -> narratives -> vectors -> Postgres."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import duckdb

from fourthdown.narrative import Document
from fourthdown.narrative.render import GRAINS, documents
from fourthdown.retrieval.embed import Embedder
from fourthdown.retrieval.store import DocumentStore

LOGGER = logging.getLogger(__name__)

ENCODE_BATCH = 64


@dataclass(frozen=True)
class IndexReport:
    documents: int
    elapsed_seconds: float
    coverage: list[tuple[str, int, int, int]]

    def render(self) -> str:
        lines = [f"indexed {self.documents} documents in {self.elapsed_seconds:.1f}s"]
        lines += [
            f"  {grain}: {count} documents, seasons {first}-{last}"
            for grain, count, first, last in self.coverage
        ]
        return "\n".join(lines)


def _batched(items: Iterable[Document], size: int) -> Iterator[list[Document]]:
    batch: list[Document] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build(
    connection: duckdb.DuckDBPyConnection,
    store: DocumentStore,
    embedder: Embedder,
    *,
    grains: Sequence[str] = GRAINS,
    seasons: Iterable[int] | None = None,
    postseason_drives_only: bool = False,
    reset: bool = False,
) -> IndexReport:
    """Generate, embed, and upsert every document for the requested grains and seasons.

    Upserts are keyed on ``doc_id``, so re-running after a new season lands adds that
    season without touching the rest of the index.
    """
    started = time.perf_counter()
    store.initialise(model=embedder.name, dimensions=embedder.dimensions, reset=reset)
    written = 0
    stream = documents(
        connection,
        grains=grains,
        seasons=seasons,
        postseason_drives_only=postseason_drives_only,
    )
    for batch in _batched(stream, ENCODE_BATCH):
        vectors = embedder.encode([document.text for document in batch])
        written += store.upsert(zip(batch, vectors, strict=True))
        if written % (ENCODE_BATCH * 20) == 0:
            LOGGER.info("indexed %d documents", written)
    return IndexReport(
        documents=written,
        elapsed_seconds=time.perf_counter() - started,
        coverage=store.coverage(),
    )
