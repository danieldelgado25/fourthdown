"""Route a question to one tool, run it, and return the evidence with its provenance.

The assistant deliberately does not paraphrase numbers. Whatever the tool returned is
what the caller gets, plus the route that produced it; the LLM's jobs are choosing the
tool and, for narrative questions, writing prose from passages it must cite. That split
is what makes the answers checkable: every number traces to SQL or to a model, and every
sentence traces to a passage.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from fourthdown.agent.router import Route, Router
from fourthdown.agent.tools import Evidence, Passage, Table, Tool
from fourthdown.llm import LLMError
from fourthdown.retrieval.store import StoreError

LOGGER = logging.getLogger(__name__)


@dataclass
class Answer:
    """One question, one route, one tool's evidence."""

    question: str
    tool: str
    route_reason: str
    decided_by: str
    answer: str
    table: Table | None = None
    passages: list[Passage] = field(default_factory=list)
    detail: dict[str, str] = field(default_factory=dict)
    failed: bool = False
    elapsed_seconds: float = 0.0


class Assistant:
    """Router plus tools. The only object the API needs to hold."""

    def __init__(self, tools: Sequence[Tool], router: Router) -> None:
        if not tools:
            raise ValueError("an assistant needs at least one tool")
        self._tools = {tool.name: tool for tool in tools}
        self._router = router

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def ask(self, question: str, *, tool: str | None = None) -> Answer:
        """Answer a question, optionally forcing a tool (the dashboard lets you)."""
        started = time.perf_counter()
        route = self._resolve(question, tool)
        evidence = self._run(route.tool, question)
        return Answer(
            question=question,
            tool=route.tool,
            route_reason=route.reason,
            decided_by=route.decided_by,
            answer=evidence.answer,
            table=evidence.table,
            passages=evidence.passages,
            detail=evidence.detail,
            failed=evidence.failed,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _resolve(self, question: str, tool: str | None) -> Route:
        if tool is None:
            route = self._router.route(question)
            if route.tool in self._tools:
                return route
            LOGGER.warning("router picked unavailable tool %s", route.tool)
            name = next(iter(self._tools))
            return Route(tool=name, reason=f"{route.tool} is not loaded", decided_by="fallback")
        if tool not in self._tools:
            raise KeyError(tool)
        return Route(tool=tool, reason="requested explicitly", decided_by="caller")

    def _run(self, name: str, question: str) -> Evidence:
        """A tool that raises is a failed answer, not a failed request.

        The dashboard shows one panel per question, and a 500 there tells the user
        nothing about which half of the system is down; a failed `Evidence` carrying the
        error does.
        """
        try:
            return self._tools[name].run(question)
        except (LLMError, StoreError, duckdb.Error) as error:
            LOGGER.exception("%s failed", name)
            return Evidence(answer=f"The {name} tool failed: {error}", failed=True)
