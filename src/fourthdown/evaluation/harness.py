"""Execution-accuracy grading for the golden question set.

Grading compares *results*, not query text: `avg(is_pass_call::INT)` and
`count(*) FILTER (WHERE is_pass_call) / count(*)` are the same answer, and a text
comparison would reject one of them. This is the standard execution-accuracy metric used
for text-to-SQL benchmarks, with two concessions to the domain:

- floats are rounded, because a rate computed two ways differs in the last bits;
- a scalar reference matches if the value appears anywhere in the candidate's rows, since
  models answer "which team" with either one column or the team plus the metric.

Some questions are traps rather than tests of SQL skill: San Diego must map to LAC, and a
rushing-play count must exclude kneels. Those are where the schema card earns its length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import duckdb

from fourthdown.rag.text_to_sql import SQLAnswer, TextToSQL

FLOAT_PRECISION = 4


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    tags: tuple[str, ...]
    reference_sql: str | None
    expect_unanswerable: bool = False


@dataclass
class QuestionResult:
    question: GoldenQuestion
    answer: SQLAnswer
    correct: bool
    detail: str

    @property
    def executed(self) -> bool:
        return self.answer.ok or self.answer.unanswerable


@dataclass
class EvaluationReport:
    results: list[QuestionResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def executed(self) -> int:
        return sum(1 for result in self.results if result.executed)

    @property
    def correct(self) -> int:
        return sum(1 for result in self.results if result.correct)

    @property
    def repairs(self) -> int:
        return sum(max(len(result.answer.attempts) - 1, 0) for result in self.results)

    def render(self) -> str:
        lines = [
            "# Text-to-SQL evaluation",
            "",
            f"- questions: {self.total}",
            f"- produced a runnable query: {self.executed}/{self.total}",
            f"- correct result: {self.correct}/{self.total}",
            f"- repair attempts used: {self.repairs}",
            "",
            "| question | correct | detail |",
            "| --- | --- | --- |",
        ]
        for result in self.results:
            mark = "yes" if result.correct else "no"
            detail = result.detail.replace("|", "\\|")
            lines.append(f"| {result.question.id} | {mark} | {detail} |")
        return "\n".join(lines) + "\n"


def load_questions(path: Path | None = None) -> list[GoldenQuestion]:
    """Read the golden set, from the packaged file unless a path is given."""
    if path is None:
        text = resources.files("fourthdown.evaluation").joinpath("golden.json").read_text()
    else:
        text = path.read_text()
    payload = json.loads(text)
    return [
        GoldenQuestion(
            id=entry["id"],
            question=entry["question"],
            tags=tuple(entry.get("tags", ())),
            reference_sql=entry.get("reference_sql"),
            expect_unanswerable=bool(entry.get("expect_unanswerable", False)),
        )
        for entry in payload["questions"]
    ]


def _normalise(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_PRECISION)
    if isinstance(value, int):
        return float(value)
    return value


def _rows(rows: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    return sorted(
        (tuple(_normalise(value) for value in row) for row in rows),
        key=lambda row: tuple(str(value) for value in row),
    )


def results_match(reference: list[tuple[object, ...]], candidate: list[tuple[object, ...]]) -> bool:
    """True when the candidate's rows carry the same answer as the reference's."""
    expected = _rows(reference)
    actual = _rows(candidate)
    if expected == actual:
        return True
    if len(expected) == 1 and len(expected[0]) == 1:
        wanted = expected[0][0]
        return any(_normalise(value) == wanted for row in candidate for value in row)
    # A ranking reference of one column matches a candidate that also carries the metric.
    if expected and all(len(row) == 1 for row in expected) and len(actual) == len(expected):
        wanted_set = {row[0] for row in expected}
        for position in range(max(len(row) for row in actual)):
            column = {row[position] for row in actual if position < len(row)}
            if column == wanted_set:
                return True
    return False


def grade(
    connection: duckdb.DuckDBPyConnection,
    question: GoldenQuestion,
    answer: SQLAnswer,
) -> QuestionResult:
    """Compare one answer against its reference query executed on the same warehouse."""
    if question.expect_unanswerable:
        correct = answer.unanswerable
        if correct:
            detail = "declined"
        elif answer.ok:
            detail = "answered a question it cannot answer"
        else:
            detail = f"failed instead of declining: {answer.error}"
        return QuestionResult(question=question, answer=answer, correct=correct, detail=detail)
    if answer.unanswerable:
        return QuestionResult(
            question=question, answer=answer, correct=False, detail="declined to answer"
        )
    if not answer.ok:
        return QuestionResult(
            question=question,
            answer=answer,
            correct=False,
            detail=answer.error or "no query produced",
        )
    assert question.reference_sql is not None
    reference = connection.execute(question.reference_sql).fetchall()
    correct = results_match(reference, answer.rows)
    detail = "matches reference" if correct else f"expected {reference[:3]}, got {answer.rows[:3]}"
    return QuestionResult(question=question, answer=answer, correct=correct, detail=detail)


def evaluate(
    connection: duckdb.DuckDBPyConnection,
    chain: TextToSQL,
    questions: list[GoldenQuestion] | None = None,
) -> EvaluationReport:
    """Run every golden question through the chain and grade the results."""
    selected = questions if questions is not None else load_questions()
    return EvaluationReport(
        results=[
            grade(connection, question, chain.answer(question.question)) for question in selected
        ]
    )
