"""The four things the assistant can actually do, behind one uniform interface.

A tool is a name, a description the router sees, and a `run` that returns evidence. The
descriptions are the router's entire world model, so they are written for a reader who
has to choose between them, not for a reader who already knows what the code does.

Nothing here talks to the router and the router does not import any concrete tool: both
depend on `Tool`, which is what keeps "what the system can do" separate from "how it
decides".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import duckdb

from fourthdown.agent.parse import parse_season, parse_state, parse_team
from fourthdown.llm import LLMClient
from fourthdown.models.fourth_down import FourthDownAdvisor
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.retrieval import recall
from fourthdown.retrieval.recall import Retriever
from fourthdown.retrieval.store import Hit

LOGGER = logging.getLogger(__name__)

TENDENCY_SQL = """
SELECT
    posteam AS team,
    count(*) AS plays,
    round(avg(is_pass_call::INT), 3) AS pass_rate,
    round(avg(CASE WHEN down IN (1, 2) THEN is_pass_call::INT END), 3) AS early_down_pass_rate,
    round(avg(epa), 3) AS epa_per_play
FROM plays
WHERE season = ? AND is_designed_play AND is_neutral_script
  AND (? IS NULL OR posteam = ?)
GROUP BY posteam
HAVING count(*) >= 50
ORDER BY pass_rate DESC
"""


@dataclass
class Evidence:
    """What a tool produced, in the shape the API and the dashboard both want.

    `answer` is prose when there is prose to give. `table` and `passages` carry the
    machine-readable half so the UI can render a grid or a citation list instead of
    re-parsing English, and `detail` carries whatever the tool wants to show its work
    with (the SQL it ran, the option values it compared).
    """

    answer: str
    table: Table | None = None
    passages: list[Passage] = field(default_factory=list)
    detail: dict[str, str] = field(default_factory=dict)
    failed: bool = False


@dataclass
class Table:
    columns: tuple[str, ...]
    rows: list[list[object]]


@dataclass
class Passage:
    title: str
    text: str
    score: float
    found_by: str


class Tool(Protocol):
    """A capability the router can pick. `run` takes the raw question, nothing else."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, question: str) -> Evidence: ...


def passages_from(hits: Sequence[Hit]) -> list[Passage]:
    return [
        Passage(title=hit.title, text=hit.body, score=hit.score, found_by=hit.found_by)
        for hit in hits
    ]


class StatsTool:
    """Exact aggregates, by writing SQL and running it."""

    name = "stats"
    description = (
        "Counts, rates, totals, leaders, and any question whose answer is a number or a "
        "ranking computed over the play-by-play table (2009-2024). Use for 'how many', "
        "'which team led', 'what rate', 'compare X and Y'."
    )

    def __init__(self, chain: TextToSQL, *, max_rows: int = 50) -> None:
        self._chain = chain
        self._max_rows = max_rows

    def run(self, question: str) -> Evidence:
        answer = self._chain.answer(question)
        if answer.unanswerable:
            return Evidence(
                answer="The warehouse does not hold the data that question needs.",
                failed=True,
            )
        if not answer.ok or answer.sql is None:
            return Evidence(
                answer=f"Could not write working SQL for that: {answer.error}",
                detail={"attempts": str(len(answer.attempts))},
                failed=True,
            )
        rows = [list(row) for row in answer.rows[: self._max_rows]]
        return Evidence(
            answer=_summarise(answer.columns, rows),
            table=Table(columns=answer.columns, rows=rows),
            detail={"sql": answer.sql, "attempts": str(len(answer.attempts))},
        )


class NarrativeTool:
    """What happened in a game, answered from retrieved narratives with citations."""

    name = "narrative"
    description = (
        "What happened in a specific game, drive, or moment: comebacks, upsets, famous "
        "finishes, 'how did X lose to Y', 'tell me about the 2018 NFC championship'. "
        "Answers from generated game summaries and cites them."
    )

    def __init__(self, retriever: Retriever, client: LLMClient, *, k: int = 5) -> None:
        self._retriever = retriever
        self._client = client
        self._k = k

    def run(self, question: str) -> Evidence:
        grounded = recall.answer(self._retriever, self._client, question, k=self._k)
        return Evidence(
            answer=grounded.answer,
            passages=passages_from(grounded.hits),
            failed=not grounded.hits,
        )


class AdvisorTool:
    """Fourth-down decisions, priced in win probability."""

    name = "advisor"
    description = (
        "Whether to go for it, kick a field goal, or punt on fourth down, given a "
        "situation: yard line, yards to go, time, and score. Use whenever the question "
        "asks what a team should do on fourth down."
    )

    def __init__(self, advisor: FourthDownAdvisor) -> None:
        self._advisor = advisor

    def run(self, question: str) -> Evidence:
        state = parse_state(question)
        if state is None:
            return Evidence(
                answer=(
                    "I need the situation to advise: yard line, yards to go, time left, "
                    "and the score margin."
                ),
                failed=True,
            )
        recommendation = self._advisor.recommend(state)
        return Evidence(
            answer=recommendation.render(),
            table=Table(
                columns=("option", "win probability", "success", "detail"),
                rows=[
                    [
                        option.name,
                        round(option.win_probability, 4),
                        None
                        if option.success_probability is None
                        else round(option.success_probability, 4),
                        option.detail,
                    ]
                    for option in recommendation.options
                ],
            ),
            detail={
                "situation": recommendation.render().splitlines()[0],
                "best": recommendation.best.name,
                "edge": f"{recommendation.edge:.4f}",
            },
        )


class TendencyTool:
    """Team pass/run tendency and efficiency, on neutral-script plays."""

    name = "tendency"
    description = (
        "How pass-happy or run-heavy a team is, and how well it did doing that: "
        "neutral-script pass rate and EPA per play for a season, by team. Use for "
        "'how often do the Ravens run', 'who is the most pass-heavy team in 2024'."
    )

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def run(self, question: str) -> Evidence:
        season = parse_season(question)
        team = parse_team(question)
        cursor = self._connection.execute(TENDENCY_SQL, [season, team, team])
        columns = tuple(description[0] for description in cursor.description or ())
        rows = [list(row) for row in cursor.fetchall()]
        detail = {"season": str(season), "team": team or "all"}
        if not rows:
            return Evidence(
                answer=f"No neutral-script plays for {season}.", detail=detail, failed=True
            )
        return Evidence(
            answer=_summarise(columns, rows),
            table=Table(columns=columns, rows=rows),
            detail=detail,
        )


def _summarise(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A one-line reading of a result set, for clients that only want the headline."""
    if not rows:
        return "No rows matched."
    if len(rows) == 1 and len(columns) == 1:
        return f"{columns[0]}: {rows[0][0]}"
    leader = ", ".join(f"{column} {value}" for column, value in zip(columns, rows[0], strict=True))
    return f"{len(rows)} row(s); top: {leader}"
