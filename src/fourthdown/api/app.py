"""The HTTP surface: an application factory over one long-lived `Services`.

The services are opened once and held for the life of the process, because the DuckDB
connection, the pgvector pool, and a 400 MB torch checkpoint are all expensive to open
and all read-only afterwards. That makes the app single-worker by design; the dev server
and `flask run` are both single-process, and scaling past that is a phase-06 problem
(gunicorn with preload, or a model server behind the API).

Every response is JSON with a stable shape, and every answer carries the route that
produced it -- the dashboard shows which tool ran, and that is a feature, not debug
output.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from fourthdown.agent import Answer, Readiness, Services, services
from fourthdown.agent.tools import TENDENCY_SQL, Passage, Table
from fourthdown.models.winprob import GameState

LOGGER = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500


def create_app(*, provided: Services | None = None) -> Flask:
    """Build the app, opening the backing services unless they were handed in (tests)."""
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    stack = ExitStack()
    resolved = provided if provided is not None else stack.enter_context(services())
    app.extensions["fourthdown"] = resolved
    app.extensions["fourthdown_stack"] = stack
    _register(app, resolved)
    return app


def _register(app: Flask, resolved: Services) -> None:
    @app.get("/api/health")
    def health() -> Response:
        return jsonify(readiness_json(resolved.readiness))

    @app.post("/api/ask")
    def ask() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": f"question exceeds {MAX_QUESTION_CHARS} characters"}), 400
        tool = payload.get("tool")
        try:
            answer = resolved.assistant.ask(question, tool=None if tool is None else str(tool))
        except KeyError:
            return jsonify(
                {"error": f"unknown tool {tool!r}", "tools": list(resolved.assistant.tool_names)}
            ), 400
        return jsonify(answer_json(answer))

    @app.post("/api/advise")
    def advise() -> tuple[Response, int] | Response:
        if resolved.advisor is None:
            return jsonify({"error": "no trained models; run `fourthdown train`"}), 503
        payload = request.get_json(silent=True) or {}
        try:
            state = _state_from(payload)
        except KeyError as error:
            return jsonify({"error": f"{error.args[0]} is required"}), 400
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        recommendation = resolved.advisor.recommend(state)
        return jsonify(
            {
                "situation": recommendation.render().splitlines()[0],
                "best": recommendation.best.name,
                "edge": recommendation.edge,
                "options": [
                    {
                        "name": option.name,
                        "win_probability": option.win_probability,
                        "success_probability": option.success_probability,
                        "detail": option.detail,
                    }
                    for option in recommendation.options
                ],
            }
        )

    @app.get("/api/tendencies")
    def tendencies() -> tuple[Response, int] | Response:
        if resolved.connection is None:
            return jsonify({"error": "no warehouse; run `fourthdown build`"}), 503
        try:
            season = int(request.args.get("season", 2024))
        except ValueError:
            return jsonify({"error": "season must be an integer"}), 400
        team = request.args.get("team") or None
        cursor = resolved.connection.execute(TENDENCY_SQL, [season, team, team])
        columns = tuple(description[0] for description in cursor.description or ())
        rows = [list(row) for row in cursor.fetchall()]
        return jsonify({"season": season, "columns": columns, "rows": rows})

    @app.errorhandler(500)
    def server_error(error: Exception) -> tuple[Response, int]:
        LOGGER.exception("unhandled error", exc_info=error)
        return jsonify({"error": "internal error"}), 500


def _state_from(payload: dict[str, Any]) -> GameState:
    yardline = float(payload["yardline_100"])
    togo = float(payload["ydstogo"])
    if not 1.0 <= yardline <= 99.0:
        raise ValueError("yardline_100 must be between 1 and 99")
    if togo < 0.0:
        raise ValueError("ydstogo must not be negative")
    return GameState(
        yardline_100=yardline,
        down=4,
        ydstogo=togo,
        game_seconds_remaining=float(payload.get("game_seconds_remaining", 900.0)),
        score_differential=float(payload.get("score_differential", 0.0)),
        posteam_is_home=bool(payload.get("posteam_is_home", True)),
        posteam_spread=float(payload.get("posteam_spread", 0.0)),
        goal_to_go=bool(payload.get("goal_to_go", yardline <= togo)),
    )


def readiness_json(readiness: Readiness) -> dict[str, Any]:
    return {
        "status": "degraded" if readiness.degraded else "ok",
        "warehouse": readiness.warehouse,
        "llm": readiness.llm,
        "retrieval": readiness.retrieval,
        "models": readiness.models,
        "tools": list(readiness.tools),
        "skipped": readiness.skipped,
    }


def answer_json(answer: Answer) -> dict[str, Any]:
    return {
        "question": answer.question,
        "tool": answer.tool,
        "route": {"reason": answer.route_reason, "decided_by": answer.decided_by},
        "answer": answer.answer,
        "table": _table_json(answer.table),
        "passages": [_passage_json(passage) for passage in answer.passages],
        "detail": answer.detail,
        "failed": answer.failed,
        "elapsed_seconds": round(answer.elapsed_seconds, 3),
    }


def _table_json(table: Table | None) -> dict[str, Any] | None:
    if table is None:
        return None
    return {"columns": list(table.columns), "rows": table.rows}


def _passage_json(passage: Passage) -> dict[str, Any]:
    return {
        "title": passage.title,
        "text": passage.text,
        "score": passage.score,
        "found_by": passage.found_by,
    }
