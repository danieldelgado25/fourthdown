"""Block until the things a container depends on exist.

Kubernetes has no "start after that Job finished", so init containers run
`fourthdown wait` with the checks their main container needs. Each check returns why it
is not ready yet, or None, so a timeout says exactly what never came up.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

import psycopg

from fourthdown.config import Paths
from fourthdown.llm import OllamaClient
from fourthdown.models.card import MODEL_CARD
from fourthdown.retrieval import DocumentStore
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
    client = OllamaClient(model, host=host)

    def check() -> str | None:
        return None if client.available() else f"Ollama at {client.host} has no {model}"

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
