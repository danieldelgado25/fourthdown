"""Durable, local records of how questions were answered.

Two logs live under the data root's ``feedback`` directory, one JSON object per line:

- ``misses.jsonl`` (`MissLog`, written by the warehouse service): every SQL query that came
  back empty, with the query, what each filter left, the reviewer's verdict, and any
  revision. ``fourthdown misses`` lists them; a real miss becomes a golden question, a
  schema-card rule, or a fix in name matching.
- ``queries.jsonl`` (`QueryLog`, written by the gateway): every question asked, with its
  route, latency, whether SQL ran, how many rows came back, and any error.
  ``fourthdown queries`` summarises it (`summarise`): p50/p95 latency, error rate, and
  the route mix.

Both are best effort. The services mount the data read-only with only this directory
writable; if it is missing or unwritable, the answer still goes out and the failure is
logged once.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

FILENAME = "misses.jsonl"
QUERIES_FILENAME = "queries.jsonl"
NOT_REVIEWED = "not reviewed"
"""How a miss with no verdict is listed: the review is switched off for that chain."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JsonlLog:
    """Appends one JSON object per line, stamped with ``at``; safe across request threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._warned = False

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps({"at": _now().isoformat(timespec="seconds"), **entry}, default=str)
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as error:
                if not self._warned:
                    LOGGER.warning("cannot write %s: %s", self.path, error)
                    self._warned = True


class MissLog(JsonlLog):
    """Empty SQL results, written by `fourthdown.rag.text_to_sql.TextToSQL`."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory / FILENAME)


class QueryLog(JsonlLog):
    """Every question the gateway answered (see `query_entry` for the fields)."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory / QUERIES_FILENAME)


def query_entry(
    *,
    question: str,
    tool: str | None,
    decided_by: str | None,
    elapsed_seconds: float,
    detail: dict[str, str] | None = None,
    rows: list[list[Any]] | None = None,
    failed: bool = False,
    error: str | None = None,
    guest: bool = False,
) -> dict[str, Any]:
    """One ``queries.jsonl`` record.

    ``sql_ran`` is whether the answer carries executed SQL; ``rows`` is how many rows its
    table holds (``None`` without a table) and ``empty`` whether they hold only NULLs, the
    shape a sum over nothing returns. ``review`` is the empty-result verdict, if any.
    """
    detail = detail or {}
    count = None if rows is None else len(rows)
    empty = None if rows is None else all(value is None for row in rows for value in row)
    return {
        "question": question,
        "tool": tool,
        "decided_by": decided_by,
        "elapsed_ms": round(elapsed_seconds * 1000, 1),
        "sql_ran": "sql" in detail,
        "rows": count,
        "empty": empty,
        "review": detail.get("review"),
        "failed": failed,
        "error": error,
        "guest": guest,
    }


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


def since(records: Iterable[dict[str, Any]], hours: float | None) -> list[dict[str, Any]]:
    """The records stamped within the last ``hours`` (all of them when ``None``)."""
    if hours is None:
        return list(records)
    cutoff = _now() - timedelta(hours=hours)
    kept = []
    for record in records:
        try:
            stamped = datetime.fromisoformat(str(record.get("at")))
        except ValueError:
            continue
        if stamped >= cutoff:
            kept.append(record)
    return kept


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile: the smallest value at or above ``fraction`` of the data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class QuerySummary:
    """What `summarise` reports over a set of ``queries.jsonl`` records."""

    total: int
    p50_ms: float | None
    p95_ms: float | None
    error_rate: float
    sql_ran: int
    empty_rate: float | None
    """Of the answers that ran SQL, the share that came back empty."""
    tools: dict[str, int]
    decided_by: dict[str, int]
    slowest: list[dict[str, Any]]

    def render(self) -> str:
        if not self.total:
            return "no queries recorded"

        def ms(value: float | None) -> str:
            return "-" if value is None else f"{value:,.0f} ms"

        def mix(counts: dict[str, int]) -> str:
            return ", ".join(
                f"{name} {count} ({count / self.total:.0%})" for name, count in counts.items()
            )

        empty = "-" if self.empty_rate is None else f"{self.empty_rate:.0%}"
        lines = [
            f"queries      {self.total}",
            f"latency      p50 {ms(self.p50_ms)}, p95 {ms(self.p95_ms)}",
            f"errors       {self.error_rate:.1%}",
            f"sql ran      {self.sql_ran}, empty {empty}",
            f"tools        {mix(self.tools)}",
            f"decided by   {mix(self.decided_by)}",
        ]
        if self.slowest:
            lines.append("slowest:")
            lines += [
                f"  {ms(record.get('elapsed_ms'))}  [{record.get('tool')}]  "
                f"{record.get('question', '')}"
                for record in self.slowest
            ]
        return "\n".join(lines)


def summarise(records: list[dict[str, Any]], *, slowest: int = 3) -> QuerySummary:
    """Latency percentiles, error and empty rates, and the route mix over ``records``."""
    latencies = [
        float(record["elapsed_ms"])
        for record in records
        if isinstance(record.get("elapsed_ms"), int | float)
    ]
    ran = [record for record in records if record.get("sql_ran")]
    tools = Counter(str(record.get("tool") or "none") for record in records)
    deciders = Counter(str(record.get("decided_by") or "none") for record in records)
    return QuerySummary(
        total=len(records),
        p50_ms=percentile(latencies, 0.5),
        p95_ms=percentile(latencies, 0.95),
        error_rate=(
            sum(bool(record.get("failed")) for record in records) / len(records) if records else 0.0
        ),
        sql_ran=len(ran),
        empty_rate=sum(bool(record.get("empty")) for record in ran) / len(ran) if ran else None,
        tools=dict(tools.most_common()),
        decided_by=dict(deciders.most_common()),
        slowest=sorted(records, key=lambda record: -float(record.get("elapsed_ms") or 0))[:slowest],
    )
