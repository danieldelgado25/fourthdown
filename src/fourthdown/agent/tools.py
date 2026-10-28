"""The contract every tool honours, wherever it runs.

A tool is a name, a description the router sees, and a `run` that returns evidence. The
descriptions are the router's entire world model, so they are written for a reader who
has to choose between them, not for a reader who already knows what the code does.

This module is deliberately dependency-free: the gateway, which routes but never computes,
imports it without pulling in DuckDB, Postgres, or torch. The concrete tools live with
the service that owns their backing store (`fourthdown.services.*`), and cross the
network as the JSON produced by `evidence_json` and read back by `evidence_from_json`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolError(RuntimeError):
    """Raised by a tool whose backing service failed, so the assistant can report it.

    Services translate their own failures (a DuckDB error, an unreachable Ollama) into
    failed `Evidence` before replying; this is what is left for the caller to catch --
    chiefly a remote tool whose service did not answer at all.
    """


@dataclass
class Table:
    """A result grid.

    Attributes:
        columns: Column names, in display order.
        rows: One list of JSON-serialisable values per row, aligned with ``columns``.
    """

    columns: tuple[str, ...]
    rows: list[list[Any]]


@dataclass
class Passage:
    """A retrieved document, as cited evidence for a narrative answer.

    Attributes:
        title: The document's headline (game or drive).
        text: The passage body the answer was grounded in.
        score: Fused retrieval score; higher ranked first.
        found_by: Which retriever found it: ``dense``, ``lexical``, or ``hybrid``.
    """

    title: str
    text: str
    score: float
    found_by: str


@dataclass
class Evidence:
    """What a tool produced, in the shape the API and the dashboard both want.

    Attributes:
        answer: Prose, when there is prose to give; always a one-line headline at least.
        table: The machine-readable result grid, so clients need not re-parse English.
        passages: Retrieved documents backing a narrative answer, in citation order.
        detail: Whatever the tool shows its work with (the SQL it ran, the option values).
        failed: True when the tool could not answer; ``answer`` then says why.
    """

    answer: str
    table: Table | None = None
    passages: list[Passage] = field(default_factory=list)
    detail: dict[str, str] = field(default_factory=dict)
    failed: bool = False


class Tool(Protocol):
    """A capability the router can pick.

    Attributes:
        name: Stable identifier the router returns and the API accepts as ``tool``.
        description: What questions the tool answers, written for the router.

    `run` takes the raw question and returns `Evidence`; it may raise `ToolError`.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, question: str) -> Evidence: ...


def summarise(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A one-line reading of a result set, for clients that only want the headline.

    Args:
        columns: Column names of the result.
        rows: The result rows.

    Returns:
        ``"column: value"`` for a single cell, otherwise the row count and the first row.
    """
    if not rows:
        return "No rows matched."
    if len(rows) == 1 and len(columns) == 1:
        return f"{columns[0]}: {rows[0][0]}"
    leader = ", ".join(f"{column} {value}" for column, value in zip(columns, rows[0], strict=True))
    return f"{len(rows)} row(s); top: {leader}"


def table_json(table: Table | None) -> dict[str, Any] | None:
    """Serialise a `Table` for the wire, or pass ``None`` through."""
    if table is None:
        return None
    return {"columns": list(table.columns), "rows": table.rows}


def passage_json(passage: Passage) -> dict[str, Any]:
    """Serialise a `Passage` for the wire."""
    return {
        "title": passage.title,
        "text": passage.text,
        "score": passage.score,
        "found_by": passage.found_by,
    }


def evidence_json(evidence: Evidence) -> dict[str, Any]:
    """Serialise `Evidence` into the JSON body a tool service returns.

    Args:
        evidence: What the tool produced.

    Returns:
        A dict with ``answer``, ``table``, ``passages``, ``detail`` and ``failed``;
        `evidence_from_json` is its exact inverse.
    """
    return {
        "answer": evidence.answer,
        "table": table_json(evidence.table),
        "passages": [passage_json(passage) for passage in evidence.passages],
        "detail": evidence.detail,
        "failed": evidence.failed,
    }


def evidence_from_json(body: dict[str, Any]) -> Evidence:
    """Rebuild `Evidence` from a tool service's JSON reply.

    Args:
        body: The dict produced by `evidence_json` on the other side of the wire.

    Returns:
        The equivalent `Evidence`.

    Raises:
        KeyError: If ``answer`` is missing, i.e. the body is not a tool reply.
    """
    table = body.get("table")
    return Evidence(
        answer=body["answer"],
        table=None if table is None else Table(tuple(table["columns"]), table["rows"]),
        passages=[Passage(**passage) for passage in body.get("passages", [])],
        detail={str(key): str(value) for key, value in body.get("detail", {}).items()},
        failed=bool(body.get("failed", False)),
    )
