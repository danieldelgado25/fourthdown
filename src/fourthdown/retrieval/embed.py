"""Embedding backends behind a two-property protocol.

`OllamaEmbedder` is the real one: it reuses the Ollama server the SQL layer already
needs, so the project stays a single local runtime instead of pulling a ~2 GB torch
wheel in to run a 137M-parameter encoder. `HashingEmbedder` is deterministic, has no
dependencies, and exists so the store, the fusion, and the CLI are all testable offline
and in CI.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import requests

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0
BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Embeddings from a local Ollama model (``nomic-embed-text`` by default)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model or os.environ.get("FOURTHDOWN_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self._host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self._timeout = timeout
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.encode(["dimension probe"])[0])
        return self._dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[start : start + BATCH_SIZE])
            try:
                response = requests.post(
                    f"{self._host}/api/embed",
                    json={"model": self._model, "input": batch},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as error:
                raise EmbeddingError(
                    f"could not reach Ollama at {self._host}: {error}. "
                    f"Is it running, and is `ollama pull {self._model}` done?"
                ) from error
            except ValueError as error:
                raise EmbeddingError(f"Ollama returned invalid JSON: {error}") from error
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise EmbeddingError(f"Ollama returned {embeddings!r} for {len(batch)} inputs")
            vectors.extend([float(value) for value in vector] for vector in embeddings)
        if vectors:
            self._dimensions = len(vectors[0])
        return vectors


class HashingEmbedder:
    """A hashed bag-of-words encoder: no model, no network, same vector every time.

    It has no semantics -- "touchdown" and "score" land in unrelated buckets -- so it is
    a test double and a CI stand-in, never a retrieval backend for real questions.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing:{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def default_embedder() -> Embedder:
    """The embedder the CLI uses unless told otherwise."""
    return OllamaEmbedder()
