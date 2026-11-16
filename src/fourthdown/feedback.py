"""A durable record of questions whose query came back empty.

Each empty result is one JSON line in ``misses.jsonl`` under the data root's ``feedback``
directory: the question, the query, what each filter left, the reviewer's verdict, and any
revision. ``fourthdown misses`` lists them. A record that turns out to be a real miss
becomes a golden question, a schema-card rule, or a fix in name matching.

The log is best effort. The warehouse service mounts the data read-only with only this
directory writable; if it is missing or unwritable, the answer still goes out and the
failure is logged once.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

FILENAME = "misses.jsonl"
NOT_REVIEWED = "not reviewed"
"""How a record with no verdict is listed: the review is switched off for that chain."""


class MissLog:
    """Appends one JSON object per line; safe to share across request threads."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / FILENAME
        self._lock = threading.Lock()
        self._warned = False

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}, default=str
        )
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as error:
                if not self._warned:
                    LOGGER.warning("cannot write the miss log at %s: %s", self.path, error)
                    self._warned = True


def read(path: Path) -> Iterator[dict[str, Any]]:
    """Every record in ``path``, oldest first; a corrupt line is skipped, not fatal."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("skipping unreadable line %d of %s", number, path)
