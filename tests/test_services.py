"""The services in isolation, and the gateway talking to them over (simulated) HTTP."""

from __future__ import annotations

from typing import Any

import pytest
import requests
from flask import Flask

from fourthdown.agent.tools import Evidence, Passage, Table, evidence_from_json, evidence_json
from fourthdown.llm import LLMError
from fourthdown.models.fourth_down import FourthDownAdvisor, PuntModel
from fourthdown.models.winprob import GameState
from fourthdown.services import gateway, models, warehouse
from fourthdown.services.common import Capabilities, Reloading, tool_service


class StubTool:
    def __init__(self, name: str, evidence: Evidence) -> None:
        self.name = name
        self.description = f"the {name} tool"
        self._evidence = evidence

    def run(self, question: str) -> Evidence:
        return self._evidence


class FailingTool:
    name = "stats"
    description = "raises"

    def run(self, question: str) -> Evidence:
        raise LLMError("ollama is not running")


class LinearWinProbability:
    """A closed-form stand-in for the network, so the endpoint is tested, not the model."""

    def probability(self, state: GameState) -> float:
        value = 0.5 + 0.02 * state.score_differential + 0.004 * (100.0 - state.yardline_100)
        return min(0.99, max(0.01, value))


class Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def probability(self, *_: object) -> float:
        return self.value


class DownLLM:
    """An LLM that is never there, so routing falls back to keywords."""

    model = "down"

    def available(self) -> bool:
        return False

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str:
        raise LLMError("connection refused")


class Reply:
    """A `requests.Response` look-alike over a Flask test response."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class FlaskHTTP:
    """Routes `requests`-style calls to in-process Flask apps by base URL."""

    def __init__(self, apps: dict[str, Flask]) -> None:
        self._clients = {url: app.test_client() for url, app in apps.items()}

    def _client(self, url: str) -> tuple[Any, str]:
        for base, client in self._clients.items():
            if url.startswith(base):
                return client, url[len(base) :]
        raise requests.ConnectionError(f"no route to {url}")

    def get(self, url: str, **kwargs: Any) -> Reply:
        client, path = self._client(url)
        response = client.get(path, query_string=kwargs.get("params"))
        return Reply(response.status_code, response.get_json())

    def post(self, url: str, **kwargs: Any) -> Reply:
        client, path = self._client(url)
        response = client.post(path, json=kwargs.get("json"))
        return Reply(response.status_code, response.get_json())


def _advisor() -> FourthDownAdvisor:
    return FourthDownAdvisor(
        win_probability=LinearWinProbability(),
        conversion=Constant(0.6),
        field_goal=Constant(0.5),
        punt=PuntModel({}, 75.0),
    )


def _stats_tool() -> StubTool:
    return StubTool(
        "stats",
        Evidence(
            answer="1 row",
            table=Table(columns=("sacks",), rows=[[42]]),
            detail={"sql": "SELECT count(*) AS sacks FROM plays LIMIT 100"},
        ),
    )


def _narrative_tool() -> StubTool:
    return StubTool(
        "narrative",
        Evidence(
            answer="They lost [1].",
            passages=[Passage(title="Week 1", text="a game", score=0.5, found_by="hybrid")],
        ),
    )


def _service(name: str, state: Capabilities, errors: tuple[type[BaseException], ...] = ()):
    return tool_service(name, Reloading(lambda stack: state), errors=errors)


@pytest.fixture()
def warehouse_app(views_connection) -> Flask:
    state = warehouse.WarehouseCapabilities(
        tools=[_stats_tool(), warehouse.TendencyTool(views_connection)],
        components={"warehouse": True, "llm": True},
        connection=views_connection,
    )
    return warehouse.create_app(Reloading(lambda stack: state))


@pytest.fixture()
def models_app() -> Flask:
    advisor = _advisor()
    state = models.ModelCapabilities(
        tools=[models.AdvisorTool(advisor)], components={"models": True}, advisor=advisor
    )
    return models.create_app(Reloading(lambda stack: state))


@pytest.fixture()
def http(warehouse_app, models_app) -> FlaskHTTP:
    retrieval = _service(
        "retrieval", Capabilities(tools=[_narrative_tool()], components={"retrieval": True})
    )
    return FlaskHTTP(
        {
            "http://warehouse": warehouse_app,
            "http://retrieval": retrieval,
            "http://models": models_app,
        }
    )


@pytest.fixture()
def api(http):
    catalog = gateway.Catalog(["http://warehouse", "http://retrieval", "http://models"], http)
    return gateway.create_app(catalog, DownLLM(), http).test_client()


def test_evidence_survives_the_wire():
    evidence = _narrative_tool().run("")
    evidence.table = Table(columns=("a",), rows=[[1]])
    assert evidence_from_json(evidence_json(evidence)) == evidence


def test_tool_service_reports_its_tools_and_runs_them():
    app = _service("svc", Capabilities(tools=[_stats_tool()], components={"warehouse": True}))
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["tools"] == [{"name": "stats", "description": "the stats tool"}]
    body = client.post("/tools/stats", json={"question": "sacks"}).get_json()
    assert body["table"] == {"columns": ["sacks"], "rows": [[42]]}


def test_tool_service_is_not_ready_without_tools():
    app = _service("svc", Capabilities(skipped={"warehouse": "missing"}))
    assert app.test_client().get("/health").status_code == 503


def test_tool_service_validates_the_question_and_the_tool():
    client = _service("svc", Capabilities(tools=[_stats_tool()])).test_client()
    assert client.post("/tools/stats", json={}).status_code == 400
    assert client.post("/tools/stats", json={"question": "x" * 501}).status_code == 400
    assert client.post("/tools/oracle", json={"question": "hi"}).status_code == 404


def test_tool_service_turns_a_backing_failure_into_failed_evidence():
    app = _service("svc", Capabilities(tools=[FailingTool()]), errors=(LLMError,))
    body = app.test_client().post("/tools/stats", json={"question": "sacks"}).get_json()
    assert body["failed"] and "ollama is not running" in body["answer"]


def test_reloading_retries_while_degraded():
    opened: list[int] = []

    def open_(stack):
        opened.append(1)
        tools = [_stats_tool()] if len(opened) > 1 else []
        return Capabilities(tools=tools, skipped={} if tools else {"warehouse": "not yet"})

    resources = Reloading(open_, retry_seconds=0.0)
    assert resources.current().tools
    resources.current()
    assert len(opened) == 2, "a healthy service is not reopened"


def test_reloading_never_reopens_concurrently():
    import threading
    import time

    active, peak = [0], [0]

    def open_(stack):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.02)  # long enough for other request threads to arrive mid-reopen
        active[0] -= 1
        return Capabilities(skipped={"warehouse": "still missing"})

    resources = Reloading(open_, retry_seconds=0.0)
    threads = [threading.Thread(target=resources.current) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1, "request threads must not reopen resources at the same time"


def test_warehouse_tendencies_endpoint(warehouse_app):
    client = warehouse_app.test_client()
    body = client.get("/tendencies?season=2023").get_json()
    assert body["season"] == 2023 and "team" in body["columns"]
    assert client.get("/tendencies?season=last").status_code == 400


def test_models_advise_ranks_the_three_options(models_app):
    body = (
        models_app.test_client()
        .post("/advise", json={"yardline_100": 40, "ydstogo": 2, "score_differential": -3})
        .get_json()
    )
    assert {option["name"] for option in body["options"]} == {"go", "field_goal", "punt"}
    assert body["best"] == body["options"][0]["name"]
    assert body["edge"] >= 0.0


def test_models_advise_validates_the_situation(models_app):
    client = models_app.test_client()
    assert client.post("/advise", json={"ydstogo": 2}).status_code == 400
    assert client.post("/advise", json={"yardline_100": 120, "ydstogo": 2}).status_code == 400


def test_gateway_health_merges_every_service(api):
    body = api.get("/api/health").get_json()
    assert body["tools"] == ["stats", "tendency", "narrative", "advisor"]
    assert (body["warehouse"], body["retrieval"], body["models"]) == (True, True, True)
    assert body["services"]["models"] == ["advisor"]


def test_gateway_routes_and_forwards_questions(api):
    stats = api.post("/api/ask", json={"question": "how many sacks in 2020"}).get_json()
    assert stats["tool"] == "stats"
    assert stats["table"] == {"columns": ["sacks"], "rows": [[42]]}
    assert stats["route"]["decided_by"] == "keyword"
    narrative = api.post("/api/ask", json={"question": "what happened in week 1"}).get_json()
    assert narrative["passages"][0]["found_by"] == "hybrid"
    advice = api.post("/api/ask", json={"question": "4th and 2 from the 40, go for it?"})
    assert advice.get_json()["tool"] == "advisor"


def test_gateway_validates_questions_and_forced_tools(api):
    assert api.post("/api/ask", json={"question": "  "}).status_code == 400
    assert api.post("/api/ask", json={"question": "x" * 501}).status_code == 400
    forced = api.post("/api/ask", json={"question": "anything", "tool": "oracle"})
    assert forced.status_code == 400


def test_gateway_proxies_structured_endpoints(api):
    advice = api.post("/api/advise", json={"yardline_100": 40, "ydstogo": 2}).get_json()
    assert advice["best"] in {"go", "field_goal", "punt"}
    assert api.post("/api/advise", json={"ydstogo": 2}).status_code == 400
    assert api.get("/api/tendencies?season=2023").get_json()["season"] == 2023


def test_gateway_survives_a_service_that_is_down(warehouse_app):
    http = FlaskHTTP({"http://warehouse": warehouse_app})
    catalog = gateway.Catalog(["http://warehouse", "http://models"], http)
    api = gateway.create_app(catalog, DownLLM(), http).test_client()
    health = api.get("/api/health").get_json()
    assert health["status"] == "degraded" and not health["models"]
    assert "unreachable" in health["skipped"]["http://models"]
    assert api.post("/api/advise", json={"yardline_100": 40, "ydstogo": 2}).status_code == 503
    assert api.post("/api/ask", json={"question": "how many sacks"}).status_code == 200


def test_gateway_reports_a_tool_whose_service_vanished(http):
    catalog = gateway.Catalog(["http://retrieval"], http)
    catalog.statuses()
    http._clients.clear()
    api = gateway.create_app(catalog, DownLLM(), http).test_client()
    body = api.post("/api/ask", json={"question": "what happened in week 1"}).get_json()
    assert body["failed"] and "unreachable" in body["answer"]


def test_gateway_serves_the_dashboard_without_cors(http, tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<div id=root></div>")
    monkeypatch.setenv("FOURTHDOWN_WEB_DIR", str(tmp_path))
    catalog = gateway.Catalog([], http)
    client = gateway.create_app(catalog, DownLLM(), http).test_client()
    assert b"root" in client.get("/").data
    assert "Access-Control-Allow-Origin" not in client.get("/api/health").headers
