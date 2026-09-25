"""Narrative retrieval: embeddings, a pgvector store, and hybrid search."""

from fourthdown.retrieval.embed import (
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    OllamaEmbedder,
    default_embedder,
)
from fourthdown.retrieval.recall import GroundedAnswer, Retriever, answer
from fourthdown.retrieval.store import DocumentStore, Hit, StoreError

__all__ = [
    "DocumentStore",
    "Embedder",
    "EmbeddingError",
    "GroundedAnswer",
    "HashingEmbedder",
    "Hit",
    "OllamaEmbedder",
    "Retriever",
    "StoreError",
    "answer",
    "default_embedder",
]
