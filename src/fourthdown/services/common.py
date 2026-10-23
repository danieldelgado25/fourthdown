"""What every tool service has in common: health, tool execution, and serving.

A tool service owns one backing store and the tools that read it. It exposes:

* ``GET /health`` -- which of its tools are loaded, which components answered, and why
  anything is missing. 200 when at least one tool is loaded, 503 otherwise, so it doubles
  as a readiness probe.
* ``POST /tools/<name>`` -- body ``{"question": "..."}``, reply `evidence_json`.

Services start degraded rather than refusing to start, and retry what is missing on a
timer: Postgres coming up after the retrieval service, or a model pulled into Ollama
after boot, heals without a restart. That is what lets containers start in any order.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import waitress
from flask import Flask, Response, jsonify, request

from fourthdown.agent.tools import Evidence, Tool, evidence_json

LOGGER = logging.getLogger(__name__)

RETRY_SECONDS = 30.0
MAX_QUESTION_CHARS = 500


@dataclass
class Capabilities:
    """What a service managed to open.

    Attributes:
        tools: The tools it can run right now.
        components: Backing services by name (``warehouse``, ``llm``...) and whether
            each answered; the gateway merges these into ``/api/health``.
        skipped: Component or tool name -> human-readable reason it is unavailable.
    """

    tools: list[Tool] = field(default_factory=list)
    components: dict[str, bool] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


C = TypeVar("C", bound=Capabilities)


class Reloading(Generic[C]):
    """Holds a service's open resources and reopens them while anything is missing.

    Args:
        opener: Opens the resources, registering anything that needs closing on the given
            `ExitStack`, and returns the service's `Capabilities` (or a subclass carrying
            extra handles such as a DuckDB connection).
        retry_seconds: Minimum interval between reopen attempts while degraded.

    Use `current()` on every request; it is cheap when nothing is missing.
    """

    def __init__(
        self, opener: Callable[[ExitStack], C], *, retry_seconds: float = RETRY_SECONDS
    ) -> None:
        self._opener = opener
        self._retry_seconds = retry_seconds
        self._stack = ExitStack()
        self._state = opener(self._stack)
        self._opened = time.monotonic()

    def current(self) -> C:
        """The open capabilities, reopened first if degraded and the retry interval passed.

        Returns:
            The most recent `Capabilities`. A failed reopen keeps the previous state.
        """
        stale = time.monotonic() - self._opened >= self._retry_seconds
        if self._state.skipped and stale:
            self._opened = time.monotonic()
            stack = ExitStack()
            try:
                state = self._opener(stack)
            except Exception:
                LOGGER.exception("reopen failed; keeping the previous state")
                stack.close()
                return self._state
            self._stack.close()
            self._stack, self._state = stack, state
        return self._state

    def close(self) -> None:
        """Release everything the opener registered."""
        self._stack.close()


def health_json(name: str, capabilities: Capabilities) -> dict[str, object]:
    """The ``/health`` body of a tool service.

    Args:
        name: The service name (``warehouse``, ``retrieval``, ``models``).
        capabilities: What it currently has open.

    Returns:
        ``service``, ``tools`` (each with ``name`` and ``description``, which is how the
        gateway learns the catalog), ``components``, and ``skipped``.
    """
    return {
        "service": name,
        "tools": [
            {"name": tool.name, "description": tool.description} for tool in capabilities.tools
        ],
        "components": capabilities.components,
        "skipped": capabilities.skipped,
    }


def tool_service(
    name: str,
    resources: Reloading[C],
    *,
    errors: tuple[type[BaseException], ...],
) -> Flask:
    """A Flask app exposing ``/health`` and ``/tools/<tool>`` over ``resources``.

    Args:
        name: The service name reported on ``/health``.
        resources: The service's reloading resources.
        errors: Exception types a tool may raise when its backing store fails (a DuckDB
            error, an unreachable LLM). They become failed `Evidence` with a 200, because
            the request was fine and the answer is "this tool is down"; anything else is a
            bug and a 500.

    Returns:
        The app; service modules add their structured endpoints to it.
    """
    app = Flask(name)

    @app.get("/health")
    def health() -> tuple[Response, int]:
        capabilities = resources.current()
        return jsonify(health_json(name, capabilities)), 200 if capabilities.tools else 503

    @app.post("/tools/<tool>")
    def run(tool: str) -> tuple[Response, int]:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": f"question exceeds {MAX_QUESTION_CHARS} characters"}), 400
        tools = {loaded.name: loaded for loaded in resources.current().tools}
        if tool not in tools:
            return jsonify({"error": f"{tool} is not loaded", "tools": list(tools)}), 404
        try:
            evidence = tools[tool].run(question)
        except errors as error:
            LOGGER.exception("%s failed", tool)
            evidence = Evidence(answer=f"The {tool} tool failed: {error}", failed=True)
        return jsonify(evidence_json(evidence)), 200

    return app


def llm_available(client: object) -> bool:
    """Whether an LLM client's model answers.

    Args:
        client: Any `LLMClient`; one with an ``available()`` probe (`OllamaClient`) is
            asked, and one without (a test double) is assumed up.
    """
    probe = getattr(client, "available", None)
    return bool(probe()) if callable(probe) else True


def configure_logging(verbose: bool = False) -> None:
    """One log format for every service, so container logs read alike."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def serve(app: Flask, *, host: str | None, port: int | None, default_port: int) -> None:
    """Run ``app`` under waitress.

    Args:
        app: The service's Flask app.
        host: Bind address; ``FOURTHDOWN_HOST`` or ``0.0.0.0`` when ``None``.
        port: Port; ``FOURTHDOWN_PORT`` or ``default_port`` when ``None``.
        default_port: The service's conventional port.

    One thread, because each service holds a single DuckDB connection, Postgres
    connection, or torch model that is not safe to share across threads. Scale a service
    by running more replicas of it, not more threads.
    """
    resolved_host = host or os.environ.get("FOURTHDOWN_HOST", "0.0.0.0")
    resolved_port = port or int(os.environ.get("FOURTHDOWN_PORT", default_port))
    LOGGER.info("serving %s on %s:%d", app.name, resolved_host, resolved_port)
    waitress.serve(app, host=resolved_host, port=resolved_port, threads=1)
