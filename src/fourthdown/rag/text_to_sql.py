"""Question -> SQL -> rows, with a bounded repair loop.

Three failure modes matter and they are handled differently:

- *Unparseable or unsafe SQL* never reaches DuckDB; sqlglot rejects it and the error text
  is fed back to the model.
- *Valid SQL that DuckDB refuses* (a misspelled column, a bad cast) comes back as the
  database's own error message, which is usually specific enough to fix in one attempt.
- *SQL that runs and is wrong* is not detectable here at all. That is what the golden set
  in `fourthdown.evaluation` is for, and why the answer always ships with its query.

Each attempt is recorded, so a failure is inspectable rather than a shrug.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import duckdb

from fourthdown.llm import LLMClient
from fourthdown.rag import schema_card
from fourthdown.sql import guard

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a careful NFL analytics engineer. You translate questions into a single DuckDB \
SELECT over a fixed set of views. You reply with SQL only: no prose, no explanation, no \
markdown fences. If the question cannot be answered with these views, reply with the \
single word UNANSWERABLE."""

PROMPT = """{card}

Question: {question}

SQL:"""

REPAIR = """{card}

Question: {question}

You wrote this SQL:
{sql}

It failed: {error}
{hint}
Write a different, corrected query using only columns that exist. SQL only.

SQL:"""

UNANSWERABLE = "UNANSWERABLE"

# Repairing at temperature 0 reproduces the failing query verbatim, so retries warm up.
REPAIR_TEMPERATURES = (0.2, 0.5, 0.8)

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Attempt:
    """One generation, and why it was rejected if it was."""

    sql: str
    error: str | None = None


@dataclass
class SQLAnswer:
    """The outcome of a question: the query that ran, its rows, and the road there."""

    question: str
    sql: str | None
    columns: tuple[str, ...]
    rows: list[tuple[object, ...]]
    attempts: list[Attempt] = field(default_factory=list)
    unanswerable: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.sql is not None and self.error is None and not self.unanswerable


def extract_sql(response: str) -> str:
    """Pull SQL out of a model reply that may be fenced, prefixed, or chatty."""
    fenced = _FENCE.search(response)
    text = fenced.group(1) if fenced else response
    text = text.strip()
    if text.upper().startswith(UNANSWERABLE):
        return UNANSWERABLE
    # Models like to preface with "Here is the query:". Start at the first SQL keyword.
    match = re.search(r"\b(WITH|SELECT)\b", text, re.IGNORECASE)
    if match:
        text = text[match.start() :]
    return text.strip().rstrip(";").strip()


def _describe(error: Exception) -> str:
    message = str(error).strip()
    return message.splitlines()[0] if message else error.__class__.__name__


class TextToSQL:
    """Generates and executes SQL for a question against the warehouse."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        client: LLMClient,
        *,
        max_attempts: int = 3,
        max_rows: int = guard.DEFAULT_ROW_LIMIT,
    ) -> None:
        self._connection = connection
        self._client = client
        self._max_attempts = max_attempts
        self._max_rows = max_rows
        self._card = schema_card.build(connection)
        self._view_columns = schema_card.view_columns(connection)

    @property
    def schema_card(self) -> str:
        return self._card

    def _column_hint(self, sql: str) -> str:
        """Spell out the columns of the views a failed query touched.

        Hallucinated columns are the dominant failure, and the model cannot see which of
        its names were invented from the error alone.
        """
        words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)}
        lines = [
            f"{view}: {', '.join(self._view_columns[view])}"
            for view in sorted(guard.ALLOWED_VIEWS & words)
        ]
        return "\nColumns that exist:\n" + "\n".join(lines) + "\n" if lines else ""

    def _generate(self, question: str, previous: Attempt | None, temperature: float) -> str:
        prompt = (
            PROMPT.format(card=self._card, question=question)
            if previous is None
            else REPAIR.format(
                card=self._card,
                question=question,
                sql=previous.sql,
                error=previous.error,
                hint=self._column_hint(previous.sql),
            )
        )
        response = self._client.complete(
            system=SYSTEM_PROMPT, prompt=prompt, temperature=temperature
        )
        return extract_sql(response)

    def _execute(self, sql: str) -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
        cursor = self._connection.execute(sql)
        columns = tuple(description[0] for description in cursor.description or ())
        return columns, cursor.fetchall()

    def answer(self, question: str) -> SQLAnswer:
        started = time.perf_counter()
        attempts: list[Attempt] = []
        previous: Attempt | None = None
        for attempt_number in range(1, self._max_attempts + 1):
            temperature = (
                0.0
                if previous is None
                else REPAIR_TEMPERATURES[min(attempt_number - 2, len(REPAIR_TEMPERATURES) - 1)]
            )
            raw = self._generate(question, previous, temperature)
            if raw == UNANSWERABLE:
                return SQLAnswer(
                    question=question,
                    sql=None,
                    columns=(),
                    rows=[],
                    attempts=attempts,
                    unanswerable=True,
                    elapsed_seconds=time.perf_counter() - started,
                )
            try:
                validated = guard.validate(raw, max_rows=self._max_rows)
                columns, rows = self._execute(validated.sql)
            except (guard.UnsafeSQLError, duckdb.Error) as error:
                previous = Attempt(sql=raw, error=_describe(error))
                attempts.append(previous)
                LOGGER.info("attempt %d rejected: %s", attempt_number, previous.error)
                continue
            attempts.append(Attempt(sql=validated.sql))
            return SQLAnswer(
                question=question,
                sql=validated.sql,
                columns=columns,
                rows=rows,
                attempts=attempts,
                elapsed_seconds=time.perf_counter() - started,
            )
        return SQLAnswer(
            question=question,
            sql=None,
            columns=(),
            rows=[],
            attempts=attempts,
            error=attempts[-1].error if attempts else "no SQL generated",
            elapsed_seconds=time.perf_counter() - started,
        )
