from __future__ import annotations

import pytest

from fourthdown.agent.assistant import Assistant
from fourthdown.agent.parse import parse_season, parse_state, parse_team
from fourthdown.agent.router import KeywordRouter, LLMRouter
from fourthdown.agent.tools import Evidence, Table, TendencyTool
from fourthdown.evaluation import routing_harness
from fourthdown.llm import LLMError
from tests.conftest import ScriptedClient


class RecordingTool:
    def __init__(self, name: str, *, evidence: Evidence | None = None) -> None:
        self.name = name
        self.description = f"the {name} tool"
        self.questions: list[str] = []
        self._evidence = evidence or Evidence(answer=f"{name} answered")

    def run(self, question: str) -> Evidence:
        self.questions.append(question)
        return self._evidence


class ExplodingTool:
    name = "stats"
    description = "raises"

    def run(self, question: str) -> Evidence:
        raise LLMError("ollama is not running")


@pytest.fixture()
def tools() -> list[RecordingTool]:
    return [RecordingTool(name) for name in ("stats", "narrative", "advisor", "tendency")]


def test_keyword_router_sends_fourth_down_questions_to_the_advisor(tools):
    route = KeywordRouter(tools).route("4th and 2 at the 38, should they go for it?")
    assert route.tool == "advisor"
    assert route.decided_by == "keyword"


def test_keyword_router_sends_what_happened_questions_to_narrative(tools):
    assert KeywordRouter(tools).route("What happened in the 2017 Super Bowl?").tool == "narrative"


def test_keyword_router_defaults_to_stats(tools):
    route = KeywordRouter(tools).route("How many completions did Brady have in 2012?")
    assert route.tool == "stats"
    assert "statistics" in route.reason


def test_keyword_router_only_picks_loaded_tools():
    only_stats = [RecordingTool("stats")]
    assert KeywordRouter(only_stats).route("4th and 1, go for it?").tool == "stats"


def test_llm_router_uses_the_model_choice(tools):
    client = ScriptedClient("tendency")
    route = LLMRouter(tools, client).route("how run heavy are the Titans")
    assert (route.tool, route.decided_by) == ("tendency", "llm")
    assert "tendency:" in client.prompts[0]


def test_llm_router_extracts_a_name_from_a_chatty_reply(tools):
    route = LLMRouter(tools, ScriptedClient("I would use the narrative tool.")).route("anything")
    assert route.tool == "narrative"


def test_llm_router_falls_back_when_the_reply_is_not_a_tool(tools):
    route = LLMRouter(tools, ScriptedClient("sql_database")).route("What happened in that game?")
    assert (route.tool, route.decided_by) == ("narrative", "keyword")


class FailingClient:
    model = "down"

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str:
        raise LLMError("connection refused")


def test_llm_router_falls_back_when_the_model_is_unreachable(tools):
    route = LLMRouter(tools, FailingClient()).route("how many sacks in 2020")
    assert (route.tool, route.decided_by) == ("stats", "keyword")


def test_assistant_runs_the_routed_tool(tools):
    assistant = Assistant(tools, KeywordRouter(tools))
    answer = assistant.ask("What happened in the 2017 Super Bowl?")
    assert answer.tool == "narrative"
    assert tools[1].questions == ["What happened in the 2017 Super Bowl?"]
    assert answer.elapsed_seconds >= 0.0


def test_assistant_honours_a_forced_tool(tools):
    answer = Assistant(tools, KeywordRouter(tools)).ask("4th and 1 at the 40", tool="stats")
    assert (answer.tool, answer.decided_by) == ("stats", "caller")
    assert tools[0].questions == ["4th and 1 at the 40"]


def test_assistant_rejects_an_unknown_forced_tool(tools):
    with pytest.raises(KeyError):
        Assistant(tools, KeywordRouter(tools)).ask("anything", tool="oracle")


def test_assistant_turns_a_tool_failure_into_a_failed_answer():
    tool = ExplodingTool()
    answer = Assistant([tool], KeywordRouter([tool])).ask("how many sacks in 2020")
    assert answer.failed
    assert "ollama is not running" in answer.answer


def test_an_assistant_and_a_router_both_need_a_tool(tools):
    with pytest.raises(ValueError):
        Assistant([], KeywordRouter(tools))
    with pytest.raises(ValueError):
        KeywordRouter([])


def test_parse_state_reads_distance_field_position_clock_and_margin():
    state = parse_state("4th and 2 from the 38, down 3 with 4 minutes left")
    assert state is not None
    assert (state.ydstogo, state.yardline_100, state.down) == (2.0, 38.0, 4)
    assert state.game_seconds_remaining == pytest.approx(240.0)
    assert state.score_differential == pytest.approx(-3.0)


def test_parse_state_handles_spelled_out_numbers_and_leading():
    state = parse_state("fourth and one at the 45, up seven with two minutes to go")
    assert state is not None
    assert (state.ydstogo, state.score_differential) == (1.0, 7.0)


def test_parse_state_defaults_the_clock_and_score_when_unstated():
    state = parse_state("4th and 4 from the 40")
    assert state is not None
    assert (state.game_seconds_remaining, state.score_differential) == (900.0, 0.0)


def test_parse_state_returns_none_without_a_situation():
    assert parse_state("who led the league in sacks in 2021") is None
    assert parse_state("should we go for it on fourth down") is None


def test_parse_state_flags_goal_to_go():
    state = parse_state("4th and 3 at the 2")
    assert state is not None and state.goal_to_go


def test_parse_team_matches_nickname_city_and_abbreviation():
    assert parse_team("how often do the Ravens run") == "BAL"
    assert parse_team("Kansas City on early downs") == "KC"
    assert parse_team("tendencies for SEA in 2022") == "SEA"
    assert parse_team("league wide pass rate") is None


def test_parse_season_takes_the_last_year_or_the_default():
    assert parse_season("pass rate between 2015 and 2019") == 2019
    assert parse_season("pass rate", default=2024) == 2024


def test_tendency_tool_queries_the_warehouse(views_connection):
    evidence = TendencyTool(views_connection).run("pass rate in 2023")
    assert evidence.detail["season"] == "2023"
    assert evidence.failed or isinstance(evidence.table, Table)


def test_tendency_tool_reports_an_empty_season(views_connection):
    evidence = TendencyTool(views_connection).run("pass rate in 1999")
    assert evidence.failed


def test_routing_golden_set_covers_every_tool():
    questions = routing_harness.load_questions()
    assert {question.tool for question in questions} == {
        "stats",
        "narrative",
        "advisor",
        "tendency",
    }


def test_keyword_router_scores_on_the_golden_set():
    report = routing_harness.evaluate(KeywordRouter(routing_harness.catalog()), name="keyword")
    assert report.total == len(routing_harness.load_questions())
    assert report.accuracy() >= 0.7, routing_harness.render([report])


def test_routing_report_renders_misroutes():
    questions = [routing_harness.RoutingQuestion(question="how many sacks", tool="narrative")]
    report = routing_harness.evaluate(
        KeywordRouter(routing_harness.catalog()), name="keyword", questions=questions
    )
    rendered = routing_harness.render([report])
    assert "0/1" in rendered
    assert "how many sacks" in rendered
