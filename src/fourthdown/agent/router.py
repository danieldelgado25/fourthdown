"""Pick the tool that should answer a question.

Two routers, and the difference is the point of the module.

`KeywordRouter` is rules over the question text. It is fast, free, deterministic, and
gets the obvious cases right; it is also the fallback, so the system degrades to
something usable when Ollama is down rather than erroring.

`LLMRouter` asks the model to choose from the tool descriptions and returns its reason.
It handles the phrasings no rule anticipated, and when it answers with something that is
not a tool name -- which a 7B model does -- the keyword router takes over. Routing
accuracy for both is measured on a golden set (`docs/routing_eval.md`), which is the
only honest way to claim the LLM is worth the round trip.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from fourthdown.agent.tools import Tool
from fourthdown.llm import LLMClient, LLMError

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You route NFL analytics questions to exactly one tool. You reply with the tool name and \
nothing else: no punctuation, no explanation."""

PROMPT = """Tools:
{tools}

Question: {question}

Tool name:"""

ADVISOR_PATTERNS = (
    r"\b(4th|fourth)\s*(and|&)\b",
    r"\bgo for it\b",
    r"\bpunt\b",
    r"\bfield goal\b.*\b(or|vs|versus)\b",
)
NARRATIVE_PATTERNS = (
    r"\bwhat happened\b",
    r"\bhow did\b.*\b(win|lose|beat|collapse|comeback)\b",
    r"\btell me about\b",
    r"\bdescribe\b",
    r"\bsuper bowl\b",
    r"\b(comeback|collapse|upset|meltdown)\b",
)
TENDENCY_PATTERNS = (
    r"\b(pass|run|rush)[- ]?(rate|heavy|happy)\b",
    r"\btendenc(y|ies)\b",
    r"\bpredictab",
    r"\bhow often do(es)?\b.*\b(pass|run)\b",
)


@dataclass(frozen=True)
class Route:
    """The chosen tool, and why -- the reason is shown in the UI, not just logged."""

    tool: str
    reason: str
    decided_by: str


class Router(Protocol):
    def route(self, question: str) -> Route: ...


def _matches(question: str, patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return pattern
    return None


class KeywordRouter:
    """Rules first, statistics last: anything unmatched is a numbers question."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self._names = {tool.name for tool in tools}
        if not self._names:
            raise ValueError("a router needs at least one tool to route to")
        self._default = "stats" if "stats" in self._names else sorted(self._names)[0]

    def route(self, question: str) -> Route:
        for name, patterns in (
            ("advisor", ADVISOR_PATTERNS),
            ("tendency", TENDENCY_PATTERNS),
            ("narrative", NARRATIVE_PATTERNS),
        ):
            pattern = _matches(question, patterns)
            if pattern is not None and name in self._names:
                return Route(tool=name, reason=f"matched /{pattern}/", decided_by="keyword")
        return Route(
            tool=self._default,
            reason="no narrative or decision cue; treated as a statistics question",
            decided_by="keyword",
        )


class LLMRouter:
    """Asks the model to choose, and falls back to rules when it answers badly."""

    def __init__(self, tools: Sequence[Tool], client: LLMClient) -> None:
        self._tools = list(tools)
        self._names = {tool.name for tool in self._tools}
        self._client = client
        self._fallback = KeywordRouter(tools)

    @property
    def catalog(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools)

    def route(self, question: str) -> Route:
        try:
            reply = self._client.complete(
                system=SYSTEM_PROMPT,
                prompt=PROMPT.format(tools=self.catalog, question=question),
                temperature=0.0,
            )
        except LLMError as error:
            LOGGER.warning("router unavailable, falling back to keywords: %s", error)
            return self._fallback.route(question)
        chosen = _first_tool_name(reply, self._names)
        if chosen is None:
            LOGGER.info("router replied %r, which is not a tool; falling back", reply.strip()[:60])
            return self._fallback.route(question)
        return Route(tool=chosen, reason=f"model chose {chosen}", decided_by="llm")


def _first_tool_name(reply: str, names: set[str]) -> str | None:
    """Accept a bare name, a quoted one, or a name buried in a sentence."""
    for word in re.findall(r"[a-z_]+", reply.lower()):
        if word in names:
            return word
    return None
