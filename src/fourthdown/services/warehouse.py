"""The warehouse service: everything answered by querying DuckDB.

Tools:
    ``stats`` -- text-to-SQL over the semantic views (needs Ollama for the SQL).
    ``tendency`` -- neutral-script pass rate and EPA by team (no LLM).

Endpoints, beyond `common.tool_service`'s ``/health`` and ``/tools/<name>``:
    ``GET /tendencies?season=2024&team=KC`` -- the tendency table as structured JSON.

Dependencies: the ``warehouse`` extra (duckdb, sqlglot). Reads ``FOURTHDOWN_DATA_DIR``,
``OLLAMA_HOST``, ``FOURTHDOWN_LLM_MODEL``. Default port 8001.

Run: ``python -m fourthdown.services.warehouse`` or ``fourthdown serve warehouse``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import duckdb
from flask import Flask, Response, jsonify, request

from fourthdown.agent.parse import parse_season, parse_team
from fourthdown.agent.tools import Evidence, Table, summarise
from fourthdown.config import Paths, data_paths
from fourthdown.data import warehouse
from fourthdown.llm import LLMClient, LLMError, default_client
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.services.common import (
    Capabilities,
    Reloading,
    configure_logging,
    llm_available,
    serve,
    tool_service,
)

NAME = "warehouse"
PORT = 8001

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


def tendencies(
    connection: duckdb.DuckDBPyConnection, season: int, team: str | None = None
) -> Table:
    """Neutral-script tendency and efficiency by team.

    Args:
        connection: An open warehouse connection.
        season: The season to summarise.
        team: Restrict to one team abbreviation, or ``None`` for the league.

    Returns:
        One row per team with at least 50 qualifying plays, most pass-heavy first.
    """
    with connection.cursor() as cursor:  # one per request; connections are not thread-safe
        cursor.execute(TENDENCY_SQL, [season, team, team])
        columns = tuple(description[0] for description in cursor.description or ())
        return Table(columns=columns, rows=[list(row) for row in cursor.fetchall()])


class StatsTool:
    """Exact aggregates, by writing SQL and running it.

    Args:
        chain: The text-to-SQL chain (generate, guard, execute, repair).
        max_rows: Rows returned to the caller; the full result is never shipped.

    `run(question)` returns `Evidence` with the result table and the SQL in ``detail``,
    or a failed `Evidence` when the question is unanswerable or no valid SQL was found.
    """

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
        """Answer ``question`` with SQL; see the class docstring for the outputs."""
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
            answer=summarise(answer.columns, rows),
            table=Table(columns=answer.columns, rows=rows),
            detail={"sql": answer.sql, "attempts": str(len(answer.attempts))},
        )


class TendencyTool:
    """Team pass/run tendency and efficiency, on neutral-script plays.

    Args:
        connection: An open warehouse connection.

    `run(question)` parses a season (default 2024) and an optional team out of the
    question and returns the `tendencies` table, or a failed `Evidence` if it is empty.
    """

    name = "tendency"
    description = (
        "How pass-happy or run-heavy a team is, and how well it did doing that: "
        "neutral-script pass rate and EPA per play for a season, by team. Use for "
        "'how often do the Ravens run', 'who is the most pass-heavy team in 2024'."
    )

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def run(self, question: str) -> Evidence:
        """Answer ``question`` from the tendency query; see the class docstring."""
        season = parse_season(question)
        team = parse_team(question)
        table = tendencies(self._connection, season, team)
        detail = {"season": str(season), "team": team or "all"}
        if not table.rows:
            return Evidence(
                answer=f"No neutral-script plays for {season}.", detail=detail, failed=True
            )
        return Evidence(answer=summarise(table.columns, table.rows), table=table, detail=detail)


@dataclass
class WarehouseCapabilities(Capabilities):
    """`Capabilities` plus the open connection, for the ``/tendencies`` endpoint."""

    connection: duckdb.DuckDBPyConnection | None = None


def opener(
    paths: Paths | None = None, client: LLMClient | None = None
) -> Callable[[ExitStack], WarehouseCapabilities]:
    """Build the function `Reloading` calls to open this service's resources.

    Args:
        paths: Data layout; ``FOURTHDOWN_DATA_DIR`` or the repo default when ``None``.
        client: The LLM for text-to-SQL; a `ChatClient` from the environment if ``None``.

    Returns:
        A callable that opens the warehouse read-only on the given stack and reports the
        ``warehouse`` and ``llm`` components. ``tendency`` needs only the warehouse;
        ``stats`` also needs the LLM.
    """
    resolved = paths or data_paths()
    llm = client or default_client()

    def open_(stack: ExitStack) -> WarehouseCapabilities:
        state = WarehouseCapabilities()
        state.components["llm"] = llm_ready = llm_available(llm)
        try:
            connection = stack.enter_context(warehouse.connect(resolved))
        except (FileNotFoundError, duckdb.Error) as error:
            state.components["warehouse"] = False
            state.skipped["warehouse"] = str(error)
            return state
        state.connection = connection
        state.components["warehouse"] = True
        state.tools.append(TendencyTool(connection))
        if llm_ready:
            state.tools.append(StatsTool(TextToSQL(connection, llm)))
        else:
            state.skipped["stats"] = "text-to-SQL needs the LLM, which is unavailable"
        return state

    return open_


def create_app(resources: Reloading[WarehouseCapabilities]) -> Flask:
    """The warehouse service app.

    Args:
        resources: The service's reloading resources (see `opener`).

    Returns:
        `common.tool_service` plus ``GET /tendencies``: query ``season`` (int, default
        2024) and optional ``team``; replies ``{"season", "columns", "rows"}``, 400 on a
        non-integer season, 503 when the warehouse is not open.
    """
    app = tool_service(NAME, resources, errors=(duckdb.Error, LLMError))

    @app.get("/tendencies")
    def tendencies_endpoint() -> tuple[Response, int] | Response:
        connection = resources.current().connection
        if connection is None:
            return jsonify({"error": "no warehouse; run `fourthdown build`"}), 503
        try:
            season = int(request.args.get("season", 2024))
        except ValueError:
            return jsonify({"error": "season must be an integer"}), 400
        table = tendencies(connection, season, request.args.get("team") or None)
        body: dict[str, Any] = {"season": season, "columns": table.columns, "rows": table.rows}
        return jsonify(body)

    return app


def main(host: str | None = None, port: int | None = None, model: str | None = None) -> None:
    """Open the warehouse and serve until interrupted.

    Args:
        host: Bind address (see `common.serve`).
        port: Port (see `common.serve`); 8001 by default.
        model: Ollama model for text-to-SQL; ``FOURTHDOWN_LLM_MODEL`` when ``None``.
    """
    configure_logging()
    resources = Reloading(opener(client=default_client(model)))
    try:
        serve(create_app(resources), host=host, port=port, default_port=PORT)
    finally:
        resources.close()


if __name__ == "__main__":
    main()
