"""Every number a stats answer states must come from the SQL result behind it.

The architecture's rule is that nothing the LLM writes carries a number the tools did not
compute. For the stats tool that is checkable: take the exact text a reader gets
(`StatsTool.evidence`), pull out every number, and look for each one in what the query
returned. A number is supported when it is

- a numeric cell, at the precision the answer printed it (``0.37`` supports ``0.3679``);
- part of a text cell or a column name (``2022`` in ``2022_07_ATL_CIN``);
- the row count, which the headline states; or
- in the question itself ("in 2019" may be repeated back).

Answers with no rows make no claim about the data and are not scored; the
`expect_empty` golden questions already check those.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
"""A number standing on its own: not the tail of an identifier like ``Q4`` or ``x.5``."""


def numbers(text: str) -> list[str]:
    """Every number in ``text``, as written (thousands separators kept)."""
    return NUMBER.findall(text)


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _value(token: str) -> float:
    return float(token.replace(",", ""))


@dataclass(frozen=True)
class Support:
    """What a result set can vouch for."""

    values: tuple[float, ...]
    texts: frozenset[str]

    @classmethod
    def of(
        cls,
        columns: Sequence[str],
        rows: Sequence[Sequence[object]],
        question: str = "",
    ) -> Support:
        values: list[float] = [float(len(rows))]
        texts: set[str] = set(numbers(question))
        for name in columns:
            texts.update(numbers(name))
        for row in rows:
            for cell in row:
                if isinstance(cell, bool) or cell is None:
                    continue
                if isinstance(cell, int | float):
                    if math.isfinite(float(cell)):
                        values.append(float(cell))
                    texts.update(numbers(str(cell)))
                else:
                    # Digits inside text count by value too: '07' in a game id supports 'week 7'.
                    found = re.findall(r"\d+(?:\.\d+)?", str(cell))
                    texts.update(found)
                    values.extend(float(token) for token in found)
        return cls(values=tuple(values), texts=frozenset(texts))

    def covers(self, token: str) -> bool:
        if token in self.texts or token.replace(",", "") in self.texts:
            return True
        wanted = _value(token)
        tolerance = 0.5 * 10 ** -_decimals(token) + 1e-9
        return any(abs(value - wanted) <= tolerance for value in self.values)


def unsupported(
    answer: str, columns: Sequence[str], rows: Sequence[Sequence[object]], question: str = ""
) -> list[str]:
    """The numbers in ``answer`` that nothing in the result (or question) accounts for."""
    support = Support.of(columns, rows, question)
    return [token for token in numbers(answer) if not support.covers(token)]


@dataclass(frozen=True)
class GroundingResult:
    """One answer's check: which numbers it stated without support."""

    id: str
    answer: str
    unsupported: tuple[str, ...]

    @property
    def grounded(self) -> bool:
        return not self.unsupported


@dataclass(frozen=True)
class GroundingReport:
    results: tuple[GroundingResult, ...]
    skipped: int
    """Answers with no rows, which state nothing about the data."""

    @property
    def checked(self) -> int:
        return len(self.results)

    @property
    def grounded(self) -> int:
        return sum(result.grounded for result in self.results)

    @property
    def numbers_unsupported(self) -> int:
        return sum(len(result.unsupported) for result in self.results)

    def findings(self) -> list[str]:
        return [
            f"`{result.id}` stated {', '.join(result.unsupported)} not in its result: "
            f"{result.answer[:120]}"
            for result in self.results
            if not result.grounded
        ]


Checked = tuple[str, str, str, Sequence[str], Sequence[Sequence[object]]]
"""``(id, question, answer text, columns, rows)``: one answer to check."""


def check(answers: Iterable[Checked]) -> GroundingReport:
    """Check each answer's numbers against its own rows."""
    results: list[GroundingResult] = []
    skipped = 0
    for id_, question, text, columns, rows in answers:
        if not rows or all(cell is None for row in rows for cell in row):
            skipped += 1
            continue
        results.append(
            GroundingResult(
                id=id_, answer=text, unsupported=tuple(unsupported(text, columns, rows, question))
            )
        )
    return GroundingReport(results=tuple(results), skipped=skipped)
