"""Block until the things a container depends on exist.

Kubernetes has no "start after that Job finished", so init containers run
`fourthdown wait` with the checks their main container needs. Each check returns why it
is not ready yet, or None, so a timeout says exactly what never came up.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping

import psycopg
import requests

from fourthdown.config import Paths
from fourthdown.models.card import MODEL_CARD
from fourthdown.retrieval import DocumentStore
from fourthdown.retrieval.embed import DEFAULT_HOST
from fourthdown.retrieval.store import StoreError

LOGGER = logging.getLogger(__name__)

Check = Callable[[], str | None]


def data_check(paths: Paths) -> Check:
    """The warehouse exists and training finished (the model card is written last)."""

    def check() -> str | None:
        for required in (paths.database, paths.models / MODEL_CARD):
            if not required.exists():
                return f"{required} does not exist yet"
        return None

    return check


def ollama_check(model: str, host: str | None = None) -> Check:
    """Ollama itself has pulled ``model``: embeddings come from Ollama even when
    ``FOURTHDOWN_LLM_BASE_URL`` sends chat to a hosted API, whose key Ollama never sees."""
    root = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")

    def check() -> str | None:
        try:
            response = requests.get(f"{root}/api/tags", timeout=5)
            response.raise_for_status()
            names = {str(entry.get("name", "")) for entry in response.json().get("models", [])}
        except (requests.RequestException, ValueError, AttributeError):
            return f"Ollama at {root} is unreachable"
        if model in names or f"{model}:latest" in names:
            return None
        return f"Ollama at {root} has no {model}"

    return check


def postgres_check(url: str | None = None, *, min_documents: int = 0) -> Check:
    """Postgres answers, and when asked, the narrative index holds documents."""

    def check() -> str | None:
        try:
            with DocumentStore.open(url) as store:
                if min_documents <= 0:
                    return None
                documents = store.count()
        except StoreError as error:
            return str(error)
        except psycopg.Error as error:
            return f"narrative index is not built: {error}".strip()
        if documents < min_documents:
            return f"narrative index has {documents} documents, want {min_documents}"
        return None

    return check


def wait_for(
    checks: Mapping[str, Check],
    *,
    timeout: float,
    interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, str]:
    """Poll until every check passes or the timeout lapses; return what is still pending."""
    deadline = clock() + timeout
    pending = dict(checks)
    while True:
        reasons = {name: reason for name, check in pending.items() if (reason := check())}
        pending = {name: pending[name] for name in reasons}
        if not pending or clock() >= deadline:
            return reasons
        for name, reason in reasons.items():
            LOGGER.info("waiting for %s: %s", name, reason)
        sleep(interval)
