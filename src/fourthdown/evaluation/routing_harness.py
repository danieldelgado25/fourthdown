"""Score the router: does the question reach the tool that can answer it?

Routing is the one new failure mode phase 05 introduces. A perfect text-to-SQL chain is
worthless if "what happened in the 2017 Super Bowl" is sent to it, and the failure is
invisible from the answer alone -- SQL will happily return an empty table and the model
will happily narrate one. So the router is measured the same way retrieval was: a golden
set with a labelled correct tool, scored per tool so a router that collapses everything
onto `stats` is obvious rather than merely mediocre.

Both routers are scored on every question, which is what justifies (or refutes) paying
for an LLM round trip on the way in.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

from fourthdown.agent.router import Router
from fourthdown.agent.tools import (
    AdvisorTool,
    Evidence,
    NarrativeTool,
    StatsTool,
    TendencyTool,
    Tool,
)

LOGGER = logging.getLogger(__name__)

GOLDEN_FILE = "golden_routing.json"


@dataclass(frozen=True)
class RoutingQuestion:
    question: str
    tool: str


@dataclass(frozen=True)
class RoutingResult:
    question: RoutingQuestion
    chosen: str
    reason: str

    @property
    def correct(self) -> bool:
        return self.chosen == self.question.tool


@dataclass(frozen=True)
class RoutingReport:
    name: str
    results: list[RoutingResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for result in self.results if result.correct)

    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def by_tool(self) -> dict[str, tuple[int, int]]:
        """Correct and total per expected tool."""
        tally: dict[str, tuple[int, int]] = {}
        for result in self.results:
            hits, seen = tally.get(result.question.tool, (0, 0))
            tally[result.question.tool] = (hits + int(result.correct), seen + 1)
        return dict(sorted(tally.items()))


def load_questions() -> list[RoutingQuestion]:
    raw = resources.files("fourthdown.evaluation").joinpath(GOLDEN_FILE).read_text()
    return [
        RoutingQuestion(question=item["question"], tool=item["tool"]) for item in json.loads(raw)
    ]


def evaluate(
    router: Router,
    *,
    name: str,
    questions: Sequence[RoutingQuestion] | None = None,
) -> RoutingReport:
    wanted = list(questions) if questions is not None else load_questions()
    results = []
    for question in wanted:
        route = router.route(question.question)
        LOGGER.debug("%s -> %s", question.question, route.tool)
        results.append(RoutingResult(question=question, chosen=route.tool, reason=route.reason))
    return RoutingReport(name=name, results=results)


def render(reports: Sequence[RoutingReport]) -> str:
    """A markdown report comparing the routers question by question."""
    lines = ["# Routing evaluation", ""]
    for report in reports:
        lines.append(
            f"- **{report.name}**: {report.correct}/{report.total} ({report.accuracy():.0%})"
        )
    lines.extend(
        ["", "## By expected tool", "", "| tool | " + " | ".join(r.name for r in reports) + " |"]
    )
    lines.append("| --- | " + " | ".join("---" for _ in reports) + " |")
    tools = sorted({tool for report in reports for tool in report.by_tool()})
    for tool in tools:
        cells = []
        for report in reports:
            hits, seen = report.by_tool().get(tool, (0, 0))
            cells.append(f"{hits}/{seen}")
        lines.append(f"| {tool} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Misroutes", ""])
    misses = [
        (report.name, result)
        for report in reports
        for result in report.results
        if not result.correct
    ]
    if not misses:
        lines.append("None.")
    else:
        lines.append("| router | question | expected | chosen | reason |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, result in misses:
            lines.append(
                f"| {name} | {result.question.question} | {result.question.tool} "
                f"| {result.chosen} | {result.reason} |"
            )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CatalogTool:
    """A tool as the router sees it: a name and a description, no backing service.

    Routing can therefore be scored on a laptop with no warehouse, no Postgres, and no
    trained models -- only Ollama, and only for the LLM router.
    """

    name: str
    description: str

    def run(self, question: str) -> Evidence:
        raise NotImplementedError("catalog tools exist to be routed to, not to run")


def catalog() -> list[Tool]:
    """The real tool catalog, built from the classes the API would load."""
    return [
        CatalogTool(name=tool.name, description=tool.description)
        for tool in (StatsTool, NarrativeTool, AdvisorTool, TendencyTool)
    ]
