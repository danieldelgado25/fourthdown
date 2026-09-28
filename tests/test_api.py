from __future__ import annotations

import pytest

from fourthdown.agent.assistant import Assistant
from fourthdown.agent.build import Readiness, Services
from fourthdown.agent.router import KeywordRouter
from fourthdown.agent.tools import Evidence, Passage, Table
from fourthdown.api import create_app
from fourthdown.models.fourth_down import FourthDownAdvisor, PuntModel
from fourthdown.models.winprob import GameState


class StubTool:
    def __init__(self, name: str, evidence: Evidence) -> None:
        self.name = name
        self.description = f"the {name} tool"
        self._evidence = evidence

    def run(self, question: str) -> Evidence:
        return self._evidence


class LinearWinProbability:
    """A closed-form stand-in for the network, so the endpoint is tested, not the model."""

    def probability(self, state: GameState) -> float:
        value = 0.5 + 0.02 * state.score_differential + 0.004 * (100.0 - state.yardline_100)
        return min(0.99, max(0.01, value))


class Constant:
    """A conversion or field-goal model with a fixed probability."""

    def __init__(self, value: float) -> None:
        self.value = value

    def probability(self, *_: object) -> float:
        return self.value


def _advisor() -> FourthDownAdvisor:
    return FourthDownAdvisor(
        win_probability=LinearWinProbability(),
        conversion=Constant(0.6),
        field_goal=Constant(0.5),
        punt=PuntModel({}, 75.0),
    )


@pytest.fixture()
def stub_services(views_connection) -> Services:
    stats = StubTool(
        "stats",
        Evidence(
            answer="1 row",
            table=Table(columns=("sacks",), rows=[[42]]),
            detail={"sql": "SELECT count(*) AS sacks FROM plays LIMIT 100"},
        ),
    )
    narrative = StubTool(
        "narrative",
        Evidence(
            answer="They lost [1].",
            passages=[Passage(title="Week 1", text="a game", score=0.5, found_by="hybrid")],
        ),
    )
    tools = [stats, narrative]
    return Services(
        assistant=Assistant(tools, KeywordRouter(tools)),
        readiness=Readiness(
            warehouse=True,
            llm=True,
            tools=("stats", "narrative"),
            skipped={"models": "no trained artifacts"},
        ),
        advisor=None,
        connection=views_connection,
    )


@pytest.fixture()
def client(stub_services):
    app = create_app(provided=stub_services)
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_reports_what_is_loaded_and_what_is_missing(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "degraded"
    assert body["tools"] == ["stats", "narrative"]
    assert body["skipped"]["models"] == "no trained artifacts"


def test_ask_returns_the_table_and_the_sql_that_produced_it(client):
    body = client.post("/api/ask", json={"question": "how many sacks in 2020"}).get_json()
    assert body["tool"] == "stats"
    assert body["table"] == {"columns": ["sacks"], "rows": [[42]]}
    assert body["detail"]["sql"].startswith("SELECT")
    assert body["route"]["decided_by"] == "keyword"
    assert body["failed"] is False


def test_ask_returns_cited_passages_for_narrative_questions(client):
    body = client.post("/api/ask", json={"question": "what happened in week 1"}).get_json()
    assert body["tool"] == "narrative"
    assert body["passages"][0]["found_by"] == "hybrid"


def test_ask_can_force_a_tool(client):
    body = client.post(
        "/api/ask", json={"question": "what happened in week 1", "tool": "stats"}
    ).get_json()
    assert body["tool"] == "stats"


def test_ask_rejects_an_unknown_tool(client):
    response = client.post("/api/ask", json={"question": "anything", "tool": "oracle"})
    assert response.status_code == 400
    assert response.get_json()["tools"] == ["stats", "narrative"]


def test_ask_requires_a_question(client):
    assert client.post("/api/ask", json={}).status_code == 400
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400


def test_ask_rejects_an_oversized_question(client):
    response = client.post("/api/ask", json={"question": "x" * 501})
    assert response.status_code == 400


def test_advise_is_unavailable_without_trained_models(client):
    response = client.post("/api/advise", json={"yardline_100": 40, "ydstogo": 2})
    assert response.status_code == 503


def test_advise_ranks_the_three_options(stub_services):
    stub_services.advisor = _advisor()
    api = create_app(provided=stub_services).test_client()
    body = api.post(
        "/api/advise", json={"yardline_100": 40, "ydstogo": 2, "score_differential": -3}
    ).get_json()
    assert {option["name"] for option in body["options"]} == {"go", "field_goal", "punt"}
    assert body["best"] == body["options"][0]["name"]
    assert body["edge"] >= 0.0


def test_advise_validates_the_situation(stub_services):
    stub_services.advisor = _advisor()
    api = create_app(provided=stub_services).test_client()
    assert api.post("/api/advise", json={"ydstogo": 2}).status_code == 400
    assert api.post("/api/advise", json={"yardline_100": 120, "ydstogo": 2}).status_code == 400


def test_tendencies_returns_columns_even_when_empty(client):
    body = client.get("/api/tendencies?season=2023").get_json()
    assert body["season"] == 2023
    assert "team" in body["columns"]


def test_tendencies_rejects_a_non_numeric_season(client):
    assert client.get("/api/tendencies?season=last").status_code == 400
