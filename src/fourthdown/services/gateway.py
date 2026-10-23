"""The gateway: the public API and the dashboard, in front of the tool services.

The gateway routes and never computes. It learns the tool catalog by polling each tool
service's ``/health`` (names and descriptions come from the service that owns the tool),
routes a question with the LLM router -- which falls back to keywords on its own when
Ollama is down -- and forwards the question to the owning service. No DuckDB, Postgres,
or torch is imported here; the gateway image installs none of them.

Public endpoints (the contract `web/src/api.ts` types):
    ``GET  /api/health``     -- merged readiness of every service.
    ``POST /api/ask``        -- ``{"question", "tool"?}`` -> routed answer with evidence.
    ``POST /api/advise``     -- proxied to the service owning ``advisor``.
    ``GET  /api/tendencies`` -- proxied to the service owning ``tendency``.
    ``GET  /``               -- the built dashboard, when ``web/dist`` exists.

Configuration: ``FOURTHDOWN_SERVICES`` (comma-separated base URLs of the tool services),
``OLLAMA_HOST``, ``FOURTHDOWN_LLM_MODEL``, ``FOURTHDOWN_WEB_DIR``. Default port 8000.

Run: ``python -m fourthdown.services.gateway`` or ``fourthdown serve gateway``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from flask import Flask, Response, jsonify, request

from fourthdown.agent.assistant import Assistant, answer_json
from fourthdown.agent.router import LLMRouter
from fourthdown.agent.tools import Evidence, ToolError, evidence_from_json
from fourthdown.llm import LLMClient, OllamaClient
from fourthdown.services.common import (
    MAX_QUESTION_CHARS,
    configure_logging,
    llm_available,
    serve,
)

LOGGER = logging.getLogger(__name__)

PORT = 8000
DEFAULT_SERVICES = "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003"
COMPONENTS = ("warehouse", "llm", "retrieval", "models")
CATALOG_TTL_SECONDS = 15.0
HEALTH_TIMEOUT_SECONDS = 2.0
TOOL_TIMEOUT_SECONDS = 180.0
"""Generous: a text-to-SQL answer with two repairs on a CPU-bound 7B model is slow."""


HTTP = Any
"""The transport: the `requests` module in production (only ``get`` and ``post`` are
called, with ``timeout`` and ``json``/``params``); tests pass an in-process stand-in."""


class RemoteTool:
    """A `Tool` whose `run` is a POST to the service that owns it.

    Args:
        name: Tool name, as the owning service reported it.
        description: Router-facing description, as the owning service reported it.
        base_url: The owning service's base URL.
        http: Transport (the `requests` module in production).

    `run(question)` returns the service's `Evidence`, and raises `ToolError` when the
    service is unreachable or rejects the request.
    """

    def __init__(self, name: str, description: str, base_url: str, http: HTTP) -> None:
        self.name = name
        self.description = description
        self.base_url = base_url
        self._http = http

    def run(self, question: str) -> Evidence:
        """Forward ``question`` to ``POST {base_url}/tools/{name}``."""
        try:
            response = self._http.post(
                f"{self.base_url}/tools/{self.name}",
                json={"question": question},
                timeout=TOOL_TIMEOUT_SECONDS,
            )
            body = response.json()
        except (requests.RequestException, ValueError) as error:
            raise ToolError(f"{self.name} service unreachable: {error}") from error
        if response.status_code != 200:
            raise ToolError(f"{self.name} service replied {response.status_code}: {body}")
        return evidence_from_json(body)


@dataclass
class ServiceStatus:
    """One tool service as last seen by the gateway.

    Attributes:
        url: Its base URL.
        name: Its self-reported name, or the URL when it did not answer.
        tools: Its loaded tools, as `RemoteTool`s.
        components: Its component readiness (see `common.Capabilities`).
        skipped: What it reported missing, plus ``unreachable`` if it did not answer.
    """

    url: str
    name: str
    tools: list[RemoteTool] = field(default_factory=list)
    components: dict[str, bool] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


class Catalog:
    """The tools available right now, discovered from the services' ``/health``.

    Args:
        urls: Base URLs of the tool services.
        http: Transport (the `requests` module in production).
        ttl: Seconds a discovery result is reused before polling again.

    A service that is down simply contributes no tools until it answers again, so the
    router never picks a tool that cannot run.
    """

    def __init__(self, urls: list[str], http: HTTP, *, ttl: float = CATALOG_TTL_SECONDS) -> None:
        self._urls = [url.rstrip("/") for url in urls if url.strip()]
        self._http = http
        self._ttl = ttl
        self._statuses: list[ServiceStatus] = []
        self._polled = float("-inf")

    def statuses(self) -> list[ServiceStatus]:
        """Every service's status, re-polled when older than the TTL."""
        if time.monotonic() - self._polled >= self._ttl:
            self._statuses = [self._poll(url) for url in self._urls]
            self._polled = time.monotonic()
        return self._statuses

    def tools(self) -> list[RemoteTool]:
        """Every loaded tool across all services, first service first."""
        return [tool for status in self.statuses() for tool in status.tools]

    def owner(self, tool: str) -> str | None:
        """Base URL of the service that has ``tool`` loaded, or ``None``."""
        return next((t.base_url for t in self.tools() if t.name == tool), None)

    def _poll(self, url: str) -> ServiceStatus:
        try:
            body = self._http.get(f"{url}/health", timeout=HEALTH_TIMEOUT_SECONDS).json()
        except (requests.RequestException, ValueError) as error:
            LOGGER.warning("service %s unreachable: %s", url, error)
            return ServiceStatus(url=url, name=url, skipped={url: f"unreachable: {error}"})
        return ServiceStatus(
            url=url,
            name=str(body.get("service", url)),
            tools=[
                RemoteTool(tool["name"], tool["description"], url, self._http)
                for tool in body.get("tools", [])
            ],
            components={str(k): bool(v) for k, v in body.get("components", {}).items()},
            skipped={str(k): str(v) for k, v in body.get("skipped", {}).items()},
        )


def health_json(catalog: Catalog, llm: bool) -> dict[str, Any]:
    """The merged ``/api/health`` body (the dashboard's `Health` type).

    Args:
        catalog: The discovered services.
        llm: Whether the gateway's own routing LLM answers.

    Returns:
        ``status`` (``ok`` only when nothing is skipped), one boolean per component
        (``warehouse``, ``llm``, ``retrieval``, ``models``) true if any service reports
        it, ``tools``, ``skipped`` reasons, and ``services`` -- name -> loaded tools.
    """
    statuses = catalog.statuses()
    components = dict.fromkeys(COMPONENTS, False)
    skipped: dict[str, str] = {}
    for status in statuses:
        for name, ready in status.components.items():
            components[name] = components.get(name, False) or ready
        skipped.update(status.skipped)
    components["llm"] = llm
    if not llm:
        skipped["llm"] = "the routing LLM is unavailable; routing by keywords"
    return {
        "status": "degraded" if skipped else "ok",
        **components,
        "tools": [tool.name for status in statuses for tool in status.tools],
        "skipped": skipped,
        "services": {status.name: [tool.name for tool in status.tools] for status in statuses},
    }


def web_dist() -> Path:
    """The built dashboard: ``FOURTHDOWN_WEB_DIR``, or ``web/dist`` in a checkout."""
    env = os.environ.get("FOURTHDOWN_WEB_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "web" / "dist"


def _proxy(http: HTTP, method: str, url: str, **kwargs: Any) -> tuple[Response, int]:
    """Forward a request and relay the JSON body and status, or 502 if unreachable."""
    try:
        response = getattr(http, method)(url, timeout=TOOL_TIMEOUT_SECONDS, **kwargs)
        return jsonify(response.json()), response.status_code
    except (requests.RequestException, ValueError) as error:
        return jsonify({"error": f"upstream unavailable: {error}"}), 502


def create_app(catalog: Catalog, client: LLMClient, http: HTTP = requests) -> Flask:
    """The gateway app.

    Args:
        catalog: Tool discovery over the configured services.
        client: The routing LLM (the router falls back to keywords if it fails).
        http: Transport for the proxied endpoints (the `requests` module in production).

    Returns:
        The Flask app serving the endpoints listed in the module docstring. The dashboard
        is served from the same origin as the API, so no CORS is configured.
    """
    dist = web_dist()
    app = Flask(__name__, static_folder=dist if dist.is_dir() else None, static_url_path="")
    if app.static_folder:
        app.add_url_rule("/", "index", lambda: app.send_static_file("index.html"))

    @app.get("/api/health")
    def health() -> Response:
        return jsonify(health_json(catalog, llm_available(client)))

    @app.post("/api/ask")
    def ask() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": f"question exceeds {MAX_QUESTION_CHARS} characters"}), 400
        tools = catalog.tools()
        if not tools:
            return jsonify({"error": "no tool service is available", "tools": []}), 503
        tool = payload.get("tool")
        assistant = Assistant(tools, LLMRouter(tools, client))
        try:
            answer = assistant.ask(question, tool=None if tool is None else str(tool))
        except KeyError:
            return jsonify(
                {"error": f"unknown tool {tool!r}", "tools": list(assistant.tool_names)}
            ), 400
        return jsonify(answer_json(answer))

    @app.post("/api/advise")
    def advise() -> tuple[Response, int]:
        owner = catalog.owner("advisor")
        if owner is None:
            return jsonify({"error": "no trained models; run `fourthdown train`"}), 503
        return _proxy(http, "post", f"{owner}/advise", json=request.get_json(silent=True) or {})

    @app.get("/api/tendencies")
    def tendencies() -> tuple[Response, int]:
        owner = catalog.owner("tendency")
        if owner is None:
            return jsonify({"error": "no warehouse; run `fourthdown build`"}), 503
        return _proxy(http, "get", f"{owner}/tendencies", params=request.args.to_dict())

    @app.errorhandler(500)
    def server_error(error: Exception) -> tuple[Response, int]:
        LOGGER.exception("unhandled error", exc_info=error)
        return jsonify({"error": "internal error"}), 500

    return app


def main(host: str | None = None, port: int | None = None, model: str | None = None) -> None:
    """Serve the gateway until interrupted.

    Args:
        host: Bind address (see `common.serve`).
        port: Port (see `common.serve`); 8000 by default.
        model: Ollama model for routing; ``FOURTHDOWN_LLM_MODEL`` when ``None``.
    """
    configure_logging()
    urls = os.environ.get("FOURTHDOWN_SERVICES", DEFAULT_SERVICES).split(",")
    app = create_app(Catalog(urls, requests), OllamaClient(model))
    serve(app, host=host, port=port, default_port=PORT)


if __name__ == "__main__":
    main()
