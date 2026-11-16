"""Regressions for questions the live demo got wrong.

- "how many catches did chad ochocinco have in 2010" ran `player_name = 'Chad Ochocinco'`
  against nflverse's 'C.Johnson'-style names, matched nothing, and showed ``None``; and a
  receiver's `plays` were targets, so even a matching name would have overcounted catches.
- "... in 2008" asked about a season the warehouse does not hold.
- "who did the patriots play week 1 of 2016" and "the 2018 chiefs vs rams score" went to
  narrative search, which ignored the season and teams and returned 2024 games.
"""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from conftest import ScriptedClient

from fourthdown.agent.parse import is_postseason, parse_seasons, parse_teams, parse_week
from fourthdown.agent.router import KeywordRouter
from fourthdown.data import etl, warehouse
from fourthdown.rag import text_to_sql
from fourthdown.rag.text_to_sql import TextToSQL, mentioned_seasons
from fourthdown.retrieval.store import Hit
from fourthdown.services.retrieval import NarrativeTool, search_filters
from fourthdown.services.warehouse import StatsTool

# (receiver, team, caught) per targeted pass.
TARGETS = [
    ("C.Johnson", "CIN", True),
    ("C.Johnson", "CIN", True),
    ("C.Johnson", "CIN", False),
    ("T.Houshmandzadeh", "CIN", True),
]


@pytest.fixture()
def receivers(raw_frame, tmp_paths):
    """A season of four targeted passes, three of them to C.Johnson and two caught."""
    template = raw_frame.head(1)
    rows = [
        template.with_columns(
            play_id=pl.lit(float(index)),
            posteam=pl.lit(team),
            passer_player_id=pl.lit("QB1"),
            passer_player_name=pl.lit("C.Palmer"),
            receiver_player_id=pl.lit(name),
            receiver_player_name=pl.lit(name),
            complete_pass=pl.lit(1.0 if caught else 0.0),
            yards_gained=pl.lit(10.0 if caught else 0.0),
        )
        for index, (name, team, caught) in enumerate(TARGETS, start=1)
    ]
    pl.concat(rows, how="vertical_relaxed").write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    with duckdb.connect(":memory:") as connection:
        warehouse.create_views(connection, tmp_paths)
        yield connection


# --- catches are not targets ---------------------------------------------------------


def test_player_game_separates_targets_from_catches(receivers):
    row = receivers.execute(
        "SELECT sum(plays), sum(completions) FROM player_game "
        "WHERE role = 'receiver' AND player_name = 'C.Johnson'"
    ).fetchone()
    assert row == (3, 2)


def test_passer_completions_and_rusher_nulls(receivers):
    passer = receivers.execute(
        "SELECT sum(plays), sum(completions) FROM player_game WHERE role = 'passer'"
    ).fetchone()
    assert passer == (4, 3)


def test_schema_card_tells_the_model_how_names_and_catches_work(receivers):
    card = TextToSQL(receivers, ScriptedClient()).schema_card
    assert "player_name ILIKE 'P.%Mahomes'" in card
    assert "Catches: sum(completions)" in card
    assert "GROUP BY player_id, player_name" in card


# --- names that match nothing --------------------------------------------------------


def test_a_full_name_is_retried_with_the_names_that_exist(receivers):
    client = ScriptedClient(
        "SELECT sum(completions) AS catches FROM player_game "
        "WHERE role = 'receiver' AND player_name = 'Chad Johnson'",
        "SELECT sum(completions) AS catches FROM player_game "
        "WHERE role = 'receiver' AND player_name ILIKE 'C.%Johnson'",
    )
    answer = TextToSQL(receivers, client).answer("How many catches did Chad Johnson have?")
    assert answer.ok
    assert answer.rows == [(2,)]
    assert len(answer.attempts) == 2
    assert answer.note is None
    assert "'C.Johnson' (CIN, 2023-2023)" in client.prompts[1]


def test_a_misspelled_surname_still_finds_suggestions(receivers):
    client = ScriptedClient(
        "SELECT sum(completions) FROM player_game WHERE player_name = 'C.Jonhson'",
        "SELECT sum(completions) FROM player_game WHERE player_name = 'C.Johnson'",
    )
    answer = TextToSQL(receivers, client).answer("catches for Chad Jonhson?")
    assert answer.rows == [(2,)]
    assert "'C.Johnson'" in client.prompts[1]


def test_a_player_who_is_not_there_is_said_so_without_retrying(receivers):
    client = ScriptedClient(
        "SELECT sum(completions) FROM player_game WHERE player_name = 'Zzyzx Qwertyuiop'"
    )
    answer = TextToSQL(receivers, client, review=False).answer("catches for Zzyzx Qwertyuiop?")
    assert answer.ok
    assert len(client.prompts) == 1
    assert answer.note is not None
    assert "Qwertyuiop" in answer.note


def test_stats_tool_explains_an_empty_answer_instead_of_printing_none(receivers):
    client = ScriptedClient(
        "SELECT sum(completions) AS catches FROM player_game WHERE player_name = 'Zzyzx Q'"
    )
    evidence = StatsTool(TextToSQL(receivers, client, review=False)).run("catches for Zzyzx Q?")
    assert "None" not in evidence.answer
    assert "no player named like" in evidence.answer


def test_stats_tool_says_no_rows_when_a_filter_matches_nothing(views_connection):
    client = ScriptedClient("SELECT sum(yards) FROM player_game WHERE week = 99")
    evidence = StatsTool(TextToSQL(views_connection, client, review=False)).run("yards in week 99?")
    assert evidence.answer.startswith("No rows matched")


# --- seasons the warehouse does not hold ---------------------------------------------


def test_a_season_outside_the_warehouse_is_refused_without_the_model(views_connection):
    client = ScriptedClient()
    chain = TextToSQL(views_connection, client)
    answer = chain.answer("How many catches did Chad Johnson have in 2008?")
    assert answer.unanswerable
    assert answer.note == "The warehouse covers the 2023-2023 seasons; 2008 is outside that."
    assert client.prompts == []
    evidence = StatsTool(chain).run("How many catches did Chad Johnson have in 2008?")
    assert evidence.failed
    assert "2008 is outside" in evidence.answer


def test_one_covered_season_is_enough_to_run(views_connection):
    client = ScriptedClient("SELECT count(*) FROM plays")
    answer = TextToSQL(views_connection, client).answer("Compare 2008 with 2023")
    assert answer.ok


def test_mentioned_seasons_reads_every_year():
    assert mentioned_seasons("from 2015 to 2019, not 1998") == [2015, 2019, 1998]


# --- parsing the filters a question states -------------------------------------------


@pytest.mark.parametrize(
    ("question", "teams"),
    [
        ("what was the score of the 2018 chiefs vs rams game ?", ["KC", "LA"]),
        ("who did the patriots play week 1 of the 2016 nfl season?", ["NE"]),
        ("How did the Redskins do in 2012?", ["WAS"]),
        ("Is there no way NE beats SF?", ["NE", "SF"]),
        ("is there no way to win", []),
        ("Patriots' last drive against the Falcons", ["NE", "ATL"]),
        ("Green Bay at Chicago", ["GB", "CHI"]),
    ],
)
def test_parse_teams_finds_every_team_in_order(question, teams):
    assert parse_teams(question) == teams


def test_parse_week_and_seasons():
    assert parse_week("who did they play week 1 of 2016") == 1
    assert parse_week("Week 17 in 2020") == 17
    assert parse_week("weekly totals") is None
    assert parse_seasons("2016 and 2018") == [2016, 2018]
    assert parse_seasons("no year here") == []


def test_super_bowl_wording_is_postseason():
    assert is_postseason("the 2017 Super Bowl")
    assert is_postseason("AFC championship")
    assert not is_postseason("week 1 opener")


def test_search_filters_carry_season_week_and_teams():
    filters = search_filters("who did the patriots play week 1 of the 2016 nfl season?")
    assert filters == {"season": [2016], "week": 1, "team": ["NE"]}
    filters = search_filters("what was the score of the 2018 chiefs vs rams game ?")
    assert filters == {"season": [2018], "team": ["KC", "LA"]}


def test_a_super_bowl_year_also_searches_the_season_before():
    filters = search_filters(
        "What happened in the 2017 Super Bowl between the Patriots and the Falcons?"
    )
    assert filters["season"] == [2016, 2017]
    assert filters["team"] == ["NE", "ATL"]


def test_no_stated_filters_means_an_unfiltered_search():
    assert search_filters("describe the wildest comeback ever") == {}


# --- narrative search uses the filters -----------------------------------------------


class FilteringRetriever:
    def __init__(self, hits: list[Hit], coverage: tuple[int, int] | None) -> None:
        self._hits = hits
        self._coverage = coverage
        self.calls: list[dict[str, object]] = []

    def search(self, question: str, **kwargs: object) -> list[Hit]:
        self.calls.append(kwargs)
        return self._hits

    def coverage(self) -> tuple[int, int] | None:
        return self._coverage


def test_narrative_tool_passes_the_question_filters_to_search():
    retriever = FilteringRetriever([], coverage=(2024, 2024))
    tool = NarrativeTool(retriever, ScriptedClient())  # type: ignore[arg-type]
    evidence = tool.run("who did the patriots play week 1 of the 2016 nfl season?")
    assert retriever.calls[0]["season"] == [2016]
    assert retriever.calls[0]["week"] == 1
    assert retriever.calls[0]["team"] == ["NE"]
    assert evidence.failed
    assert "covers the 2024 season" in evidence.answer


def test_narrative_tool_on_an_empty_index():
    tool = NarrativeTool(FilteringRetriever([], coverage=None), ScriptedClient())  # type: ignore[arg-type]
    assert "index is empty" in tool.run("tell me about 2016").answer


# --- routing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "who did the patriots play week 1 of the 2016 nfl season?",
        "what was the score of the 2018 chiefs vs rams game ?",
        "What was the final score of the 2017 Super Bowl?",
        "how many catches did chad ochocinco have in 2010?",
    ],
)
def test_game_facts_and_stat_lines_route_to_stats(question):
    tools = [_Named(name) for name in ("stats", "narrative", "advisor", "tendency")]
    assert KeywordRouter(tools).route(question).tool == "stats"


def test_how_a_game_unfolded_still_routes_to_narrative():
    tools = [_Named(name) for name in ("stats", "narrative", "advisor", "tendency")]
    question = "What happened in the 2017 Super Bowl between the Patriots and the Falcons?"
    assert KeywordRouter(tools).route(question).tool == "narrative"


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name

    def run(self, question: str):  # pragma: no cover - routing never runs tools
        raise AssertionError


# --- playoffs ------------------------------------------------------------------------


@pytest.fixture()
def playoffs(raw_frame, tmp_paths):
    """A 2020 Super Bowl (week 21), a 2021 Super Bowl (week 22), and a 2021 week-18 game."""
    template = raw_frame.head(1)
    games = {
        2020: [("2020_21_TB_KC", "POST", 21)],
        2021: [("2021_22_LA_CIN", "POST", 22), ("2021_18_KC_DEN", "REG", 18)],
    }
    for season, rows in games.items():
        frame = pl.concat(
            [
                template.with_columns(
                    game_id=pl.lit(game_id),
                    season=pl.lit(season),
                    season_type=pl.lit(season_type),
                    week=pl.lit(week),
                    play_id=pl.lit(float(index)),
                    receiver_player_id=pl.lit("WR1"),
                    receiver_player_name=pl.lit("C.Kupp"),
                    complete_pass=pl.lit(1.0),
                )
                for index, (game_id, season_type, week) in enumerate(rows, start=1)
            ],
            how="vertical_relaxed",
        )
        frame.write_parquet(tmp_paths.season_raw(season))
        etl.transform_season(season, tmp_paths)
    with duckdb.connect(":memory:") as connection:
        warehouse.create_views(connection, tmp_paths)
        yield connection


def test_playoff_round_follows_the_2021_schedule_change(playoffs):
    rows = playoffs.execute(
        "SELECT game_id, season_type, playoff_round FROM games ORDER BY game_id"
    ).fetchall()
    assert rows == [
        ("2020_21_TB_KC", "POST", "super_bowl"),
        ("2021_18_KC_DEN", "REG", None),
        ("2021_22_LA_CIN", "POST", "super_bowl"),
    ]


def test_every_aggregate_view_can_tell_playoffs_from_the_regular_season(playoffs):
    for view in ("games", "drives", "team_game", "player_game"):
        types = playoffs.execute(f"SELECT DISTINCT season_type FROM {view}").fetchall()
        assert {row[0] for row in types} == {"REG", "POST"}, view
    rounds = playoffs.execute(
        "SELECT season_type, sum(completions) FROM player_game "
        "WHERE role = 'receiver' AND season = 2021 GROUP BY season_type ORDER BY 1"
    ).fetchall()
    assert rounds == [("POST", 1), ("REG", 1)]


def test_the_card_explains_rounds_super_bowl_numbers_and_the_regular_season_default(
    views_connection,
):
    card = TextToSQL(views_connection, ScriptedClient()).schema_card
    assert "'wild_card', 'divisional', 'conference', 'super_bowl'" in card
    assert "Super Bowl N is season 1965 + N" in card
    assert "Season totals, rates, records, and leaders use season_type = 'REG'" in card
    # Shared column notes are printed once, not once per view.
    assert card.count("NULL in REG") == 1


@pytest.mark.parametrize(
    "question",
    [
        "How many total points were scored in Super Bowl LIII?",
        "How many passing yards did Patrick Mahomes have in the 2022 playoffs?",
        "What was the Patriots' record in 2010?",
    ],
)
def test_playoff_counts_and_records_route_to_stats(question):
    tools = [_Named(name) for name in ("stats", "narrative", "advisor", "tendency")]
    assert KeywordRouter(tools).route(question).tool == "stats"


# --- team codes ----------------------------------------------------------------------


def test_unknown_team_codes_are_found_in_every_filter_shape():
    sql = (
        "SELECT 1 FROM games WHERE (home_team = 'KC' AND away_team = 'LAR') "
        "OR 'OAK' IN (home_team, away_team) OR 'XYZ' = posteam"
    )
    assert text_to_sql.unknown_teams(sql) == ["LAR", "XYZ", "OAK"]
    assert "use 'LA'" in text_to_sql.describe_teams(["LAR"])
    assert "codes are ARI" in text_to_sql.describe_teams(["XYZ"])


def test_a_wrong_team_code_is_retried_with_the_right_one(views_connection):
    client = ScriptedClient(
        "SELECT sum(total_points) FROM games WHERE home_team = 'LAR'",
        "SELECT count(*) FROM games WHERE home_team = 'BBB'",
    )
    answer = TextToSQL(views_connection, client).answer("Rams points?")
    assert len(answer.attempts) == 2
    assert "'LAR' is not a team code; use 'LA'" in client.prompts[1]
