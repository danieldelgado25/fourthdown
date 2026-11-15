"""Question -> SQL -> rows, with a bounded repair loop.

Three failure modes matter and they are handled differently:

- *Unparseable or unsafe SQL* never reaches DuckDB; sqlglot rejects it and the error text
  is fed back to the model.
- *Valid SQL that DuckDB refuses* (a misspelled column, a bad cast) comes back as the
  database's own error message, which is usually specific enough to fix in one attempt.
- *SQL that runs and matches nothing* is caught in one case: a ``player_name`` filter that
  no row satisfies, which is nearly always a full name or a nickname where nflverse stores
  'P.Mahomes'. The names that do exist are fed back, the same way an error is.
- *SQL that runs and is wrong* is otherwise not detectable here at all. That is what the golden set
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
from fourthdown.narrative.teams import TEAM_NAMES
from fourthdown.rag import names, schema_card
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
        "AND role = 'receiver' GROUP BY player_id, player_name HAVING sum(plays) >= 100 "
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
    (
        "How many receptions did Stefon Diggs have in 2020?",
        "SELECT player_name, string_agg(DISTINCT team, '/') AS team, "
        "sum(completions) AS receptions, sum(plays) AS targets "
        "FROM player_game WHERE season = 2020 AND season_type = 'REG' AND role = 'receiver' "
        "AND player_name ILIKE 'S.%Diggs' GROUP BY player_id, player_name",
    ),
    (
        "Who did the Bears play in week 3 of 2015, and what was the score?",
        "SELECT game_date, away_team, away_score, home_team, home_score FROM games "
        "WHERE season = 2015 AND week = 3 AND (home_team = 'CHI' OR away_team = 'CHI')",
    ),
    (
        "What was the Saints' record in 2011?",
        "SELECT count(*) FILTER (WHERE (home_team = 'NO') = (home_margin > 0)) AS wins, "
        "count(*) FILTER (WHERE (home_team = 'NO') = (home_margin < 0)) AS losses, "
        "count(*) FILTER (WHERE home_margin = 0) AS ties FROM games "
        "WHERE season = 2011 AND season_type = 'REG' AND 'NO' IN (home_team, away_team)",
    ),
    (
        "How many passing yards did Joe Flacco have in the 2012 divisional round?",
        "SELECT player_name, sum(yards) AS passing_yards FROM player_game "
        "WHERE season = 2012 AND playoff_round = 'divisional' AND role = 'passer' "
        "AND player_name ILIKE 'J.%Flacco' GROUP BY player_id, player_name",
    ),
    ("How many Pro Bowl selections did the Steelers have in 2019?", "UNANSWERABLE"),
)

PROMPT = """{card}

{examples}

Question: {question}
{players}
SQL:"""

REPAIR = """{card}

Question: {question}
{players}
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
_NAME_FILTER = re.compile(r"player_name\s*(?:=|I?LIKE)\s*'([^']*)'", re.IGNORECASE)
_TEAM_COLUMNS = r"(?:home_team|away_team|posteam|defteam|team|opponent)"
_TEAM_FILTERS = (
    re.compile(rf"\b{_TEAM_COLUMNS}\s*=\s*'([^']*)'", re.IGNORECASE),
    re.compile(rf"'([^']*)'\s*=\s*{_TEAM_COLUMNS}\b", re.IGNORECASE),
    re.compile(rf"'([^']*)'\s+IN\s*\(\s*{_TEAM_COLUMNS}\b", re.IGNORECASE),
)
# Codes other sources use, mapped to the one the warehouse normalises every season to.
TEAM_CODE_ALIASES: dict[str, str] = {
    "LAR": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "JAC": "JAX",
    "WSH": "WAS",
    "ARZ": "ARI",
    "KAN": "KC",
    "GNB": "GB",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "CLV": "CLE",
    "BLT": "BAL",
    "HST": "HOU",
}


def unknown_teams(sql: str) -> list[str]:
    """Team literals in ``sql`` that are not one of the warehouse's 32 codes."""
    found = [match for pattern in _TEAM_FILTERS for match in pattern.findall(sql)]
    return [code for code in dict.fromkeys(found) if code not in TEAM_NAMES]


def describe_teams(codes: list[str]) -> str:
    parts = []
    for code in codes:
        fix = TEAM_CODE_ALIASES.get(code.upper())
        parts.append(
            f"'{code}' is not a team code; use '{fix}'"
            if fix
            else f"'{code}' is not a team code; codes are {', '.join(sorted(TEAM_NAMES))}"
        )
    return "The query ran but matched nothing: " + "; ".join(parts)


_YEAR = re.compile(r"\b(19\d\d|20\d\d)\b")
NAME_SUGGESTIONS = 8

NAME_CANDIDATES = """
SELECT player_name, string_agg(DISTINCT team, '/' ORDER BY team) AS teams,
       min(season) AS first_season, max(season) AS last_season
FROM player_game
WHERE player_name ILIKE ?
GROUP BY player_name
ORDER BY count(*) DESC
LIMIT ?
"""

# Misspellings ("Mahommes") miss the ILIKE above; a close surname still finds them.
SIMILAR_NAMES = """
SELECT player_name, string_agg(DISTINCT team, '/' ORDER BY team) AS teams,
       min(season) AS first_season, max(season) AS last_season
FROM player_game
WHERE jaro_winkler_similarity(lower(regexp_extract(player_name, '[^.]+$')), lower(?)) >= 0.88
GROUP BY player_name
ORDER BY count(*) DESC
LIMIT ?
"""


def _surname(filter_value: str) -> str:
    """The last word of a name filter: 'Chad Ochocinco' and 'C.%Ochocinco' -> 'Ochocinco'."""
    words = [word for word in re.split(r"[\s.%_]+", filter_value) if word]
    return words[-1] if words else ""


def _is_empty(rows: list[tuple[object, ...]]) -> bool:
    """No rows, or only NULLs -- what an aggregate over zero matching rows returns."""
    return all(value is None for row in rows for value in row)


def mentioned_seasons(question: str) -> list[int]:
    """Every four-digit year in the question, in order."""
    return [int(year) for year in _YEAR.findall(question)]


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
    note: str | None = None
    """Why an answer is empty or refused, in words a reader can act on."""

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
        self._names = names.NameResolver(names.load_roster(connection))
        span = connection.execute("SELECT min(season), max(season) FROM plays").fetchone()
        self.coverage: tuple[int, int] | None = (
            (int(span[0]), int(span[1])) if span and span[0] is not None else None
        )

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

    def out_of_range(self, question: str) -> str | None:
        """A refusal when every year the question names is outside the warehouse."""
        years = mentioned_seasons(question)
        if self.coverage is None or not years:
            return None
        first, last = self.coverage
        if any(first <= year <= last for year in years):
            return None
        named = ", ".join(str(year) for year in dict.fromkeys(years))
        return f"The warehouse covers the {first}-{last} seasons; {named} is outside that."

    def _name_candidates(self, surname: str) -> list[tuple[object, ...]]:
        if not surname:
            return []
        with self._connection.cursor() as cursor:
            rows = cursor.execute(NAME_CANDIDATES, [f"%{surname}%", NAME_SUGGESTIONS]).fetchall()
            if not rows and len(surname) >= 4:
                rows = cursor.execute(SIMILAR_NAMES, [surname, NAME_SUGGESTIONS]).fetchall()
        return rows

    def unmatched_names(self, sql: str) -> list[tuple[str, list[tuple[object, ...]]]]:
        """Each player_name filter in ``sql`` that matches no row, with names that do exist."""
        unmatched = []
        for value in dict.fromkeys(_NAME_FILTER.findall(sql)):
            with self._connection.cursor() as cursor:
                found = cursor.execute(
                    "SELECT 1 FROM player_game WHERE player_name ILIKE ? LIMIT 1", [value]
                ).fetchone()
            if found is None:
                unmatched.append((value, self._name_candidates(_surname(value))))
        return unmatched

    @staticmethod
    def describe_names(unmatched: list[tuple[str, list[tuple[object, ...]]]]) -> str:
        lines = []
        for value, candidates in unmatched:
            if candidates:
                listed = "; ".join(
                    f"'{name}' ({teams}, {first}-{last})" for name, teams, first, last in candidates
                )
                lines.append(f"no player_name matches '{value}'. Names that exist: {listed}")
            else:
                lines.append(f"no player named like '{_surname(value)}' is in the warehouse")
        return "The query ran but matched nothing: " + " | ".join(lines)

    def _generate(
        self, question: str, players: str, failures: list[Attempt], temperature: float
    ) -> str:
        prompt = (
            PROMPT.format(
                card=self._card, examples=render_examples(), question=question, players=players
            )
            if not failures
            else REPAIR.format(
                card=self._card,
                question=question,
                players=players,
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
        # A cursor per query: DuckDB connections are not safe to share across threads.
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            columns = tuple(description[0] for description in cursor.description or ())
            return columns, cursor.fetchall()

    def answer(self, question: str) -> SQLAnswer:
        started = time.perf_counter()
        attempts: list[Attempt] = []
        refusal = self.out_of_range(question)
        if refusal is not None:
            return SQLAnswer(
                question=question,
                sql=None,
                columns=(),
                rows=[],
                unanswerable=True,
                note=refusal,
                elapsed_seconds=time.perf_counter() - started,
            )
        players = names.hint(self._names.resolve(question))
        for attempt_number in range(1, self._max_attempts + 1):
            temperature = (
                0.0
                if not attempts
                else REPAIR_TEMPERATURES[min(attempt_number - 2, len(REPAIR_TEMPERATURES) - 1)]
            )
            raw = self._generate(question, players, attempts, temperature)
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
            note: str | None = None
            if _is_empty(rows):
                problems: list[str] = []
                retryable = False
                bad_teams = unknown_teams(validated.sql)
                if bad_teams:
                    problems.append(describe_teams(bad_teams))
                    retryable = True
                unmatched = self.unmatched_names(validated.sql)
                if unmatched:
                    problems.append(self.describe_names(unmatched))
                    retryable = retryable or any(candidates for _, candidates in unmatched)
                if problems:
                    note = " | ".join(problems)
                    if retryable and attempt_number < self._max_attempts:
                        attempts.append(Attempt(sql=validated.sql, error=note))
                        LOGGER.info("attempt %d matched nothing: %s", attempt_number, note)
                        continue
            attempts.append(Attempt(sql=validated.sql))
            return SQLAnswer(
                question=question,
                sql=validated.sql,
                columns=columns,
                rows=rows,
                attempts=attempts,
                elapsed_seconds=time.perf_counter() - started,
                note=note,
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
