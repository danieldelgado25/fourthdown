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

# Worked examples of the query shapes the model gets wrong when left to guess: extreme
# rows, per-game rollups, comparisons, and the football terms in the glossary. None is a
# golden question, so the golden sets still measure generalisation rather than recall.
EXEMPLARS: tuple[tuple[str, str], ...] = (
    (
        "Which game had the strongest wind in 2019, and how strong was it?",
        "SELECT game_id, wind FROM games WHERE season = 2019 AND wind IS NOT NULL "
        "ORDER BY wind DESC LIMIT 1",
    ),
    (
        "Which receiver gained the most yards in 2021 among those with at least 100 targets?",
        "SELECT player_name, sum(yards) AS yards FROM player_game WHERE season = 2021 "
        "AND role = 'receiver' GROUP BY player_name HAVING sum(plays) >= 100 "
        "ORDER BY yards DESC LIMIT 1",
    ),
    (
        "How does the Eagles' pass rate on third down compare with other downs in 2022?",
        "SELECT down = 3 AS third_down, avg(is_pass_call::INT) AS pass_rate FROM plays "
        "WHERE season = 2022 AND posteam = 'PHI' AND is_designed_play "
        "GROUP BY third_down ORDER BY third_down",
    ),
    (
        "What was the Packers' success rate on third and short in 2019?",
        "SELECT avg(success::INT) AS success_rate FROM plays WHERE season = 2019 "
        "AND posteam = 'GB' AND down = 3 AND distance_bucket = 'short' AND is_designed_play",
    ),
    (
        "What was the Dolphins' EPA per play over the 2022 season?",
        "SELECT avg(epa) AS epa_per_play FROM plays WHERE season = 2022 AND posteam = 'MIA' "
        "AND is_designed_play",
    ),
    ("How many Pro Bowl selections did the Steelers have in 2019?", "UNANSWERABLE"),
)

PROMPT = """{card}

{examples}

Question: {question}

SQL:"""

REPAIR = """{card}

Question: {question}

These queries failed, most recent last:
{failures}
{hint}
Write a different, corrected query using only columns that exist. SQL only.

SQL:"""

UNANSWERABLE = "UNANSWERABLE"

# Repairing at temperature 0 reproduces the failing query verbatim, so retries warm up.
REPAIR_TEMPERATURES = (0.2, 0.5, 0.8)

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MISSING_COLUMN = re.compile(r'column (?:named )?"([^"]+)"', re.IGNORECASE)


def render_examples() -> str:
    pairs = "\n\n".join(f"Question: {question}\nSQL: {sql}" for question, sql in EXEMPLARS)
    return f"Examples:\n\n{pairs}"


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


MAX_ERROR_CHARS = 300


def _describe(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        return error.__class__.__name__
    first = message.splitlines()[0]
    if "STRUCT(" in first:
        # DuckDB reads a bare table name as a whole-row struct and echoes every column.
        kind = first.split(":")[0]
        return f"{kind}: a table name was used as a column; count rows with count(*)"
    return first if len(first) <= MAX_ERROR_CHARS else first[:MAX_ERROR_CHARS] + "..."


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

    def _column_hint(self, failed: Attempt) -> str:
        """Spell out the columns of the views a failed query touched.

        Hallucinated columns are the dominant failure, and the model cannot see which of
        its names were invented from the error alone. A column borrowed from the wrong view
        is named outright, with the view that does have it.
        """
        words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", failed.sql)}
        lines = [
            f"{view}: {', '.join(self._view_columns[view])}"
            for view in sorted(guard.ALLOWED_VIEWS & words)
        ]
        hint = "\nColumns that exist:\n" + "\n".join(lines) + "\n" if lines else ""
        missing = _MISSING_COLUMN.search(failed.error or "")
        if missing:
            column = missing.group(1).lower()
            owners = [view for view, names in self._view_columns.items() if column in names]
            hint += (
                f"{column} is only a column of {', '.join(owners)}.\n"
                if owners
                else f"{column} is not a column of any view.\n"
            )
        return hint

    def _generate(self, question: str, failures: list[Attempt], temperature: float) -> str:
        prompt = (
            PROMPT.format(card=self._card, examples=render_examples(), question=question)
            if not failures
            else REPAIR.format(
                card=self._card,
                question=question,
                failures="\n".join(
                    f"SQL: {failed.sql}\nError: {failed.error}" for failed in failures
                ),
                hint=self._column_hint(failures[-1]),
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
        for attempt_number in range(1, self._max_attempts + 1):
            temperature = (
                0.0
                if not attempts
                else REPAIR_TEMPERATURES[min(attempt_number - 2, len(REPAIR_TEMPERATURES) - 1)]
            )
            raw = self._generate(question, attempts, temperature)
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
                attempts.append(Attempt(sql=raw, error=_describe(error)))
                LOGGER.info("attempt %d rejected: %s", attempt_number, attempts[-1].error)
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
