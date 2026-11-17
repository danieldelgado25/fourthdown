"""The schema card is a prompt, so its contents are worth asserting on."""

from __future__ import annotations

from fourthdown.data.warehouse import VIEW_NAMES
from fourthdown.rag import schema_card


def test_card_describes_every_view(views_connection) -> None:
    card = schema_card.build(views_connection)
    for view in VIEW_NAMES:
        assert f"{view} — " in card


def test_card_carries_the_semantics_a_model_cannot_guess(views_connection) -> None:
    card = schema_card.build(views_connection)
    assert "is_designed_play" in card
    assert "goal line" in card
    assert "LAC (not SD)" in card


def test_card_trims_the_plays_view(views_connection) -> None:
    card = schema_card.build(views_connection)
    plays_block = card.split("plays — ")[1].split("\n\n")[0]
    assert "more columns, listed below if a question needs them" in plays_block
    assert len(plays_block.splitlines()) < 45


def test_card_reports_coverage(views_connection) -> None:
    assert "seasons 2023-2023" in schema_card.build(views_connection)


def test_card_notes_are_view_specific(views_connection) -> None:
    card = schema_card.build(views_connection)
    assert "total EPA of the drive" in card


def test_card_stays_within_a_reasonable_prompt_budget(views_connection) -> None:
    # 8000 until the playoff columns (season_type everywhere, playoff_round) and their rules.
    assert len(schema_card.build(views_connection)) < 8500


def test_card_translates_football_terms_into_filters(views_connection) -> None:
    card = schema_card.build(views_connection)
    assert "distance_bucket IN ('long', 'very_long')" in card
    assert "down = 4 AND is_designed_play" in card
    assert "BAL Ravens" in card


def test_card_warns_that_per_game_views_need_rolling_up(views_connection) -> None:
    card = schema_card.build(views_connection)
    assert "HAVING sum(plays) >= N" in card
    assert "sum(epa_per_play * plays) / sum(plays)" in card
    assert "drives has no field_zone" in card


def _names(question: str, connection) -> list[str]:
    extras = schema_card.plays_extras(connection)
    return [column.name for column in schema_card.relevant(question, extras)]


def test_left_out_columns_are_found_by_name_or_cue(views_connection) -> None:
    assert _names("How many penalties were called on DAL in 2023?", views_connection) == ["penalty"]
    assert "air_yards" in _names("Highest average air yards in 2022?", views_connection)
    assert "kick_distance" in _names("Longest field goal made in 2019?", views_connection)
    assert "interception" in _names("Who threw the most interceptions?", views_connection)
    assert _names("Two-point tries by PHI in 2024?", views_connection) == ["two_point_attempt"]


def test_a_question_needing_nothing_extra_gets_nothing(views_connection) -> None:
    assert _names("What was the score of Super Bowl LX?", views_connection) == []
    # "left" alone is the clock, not the side of the field.
    assert _names("How many seconds were left?", views_connection) == []


def test_highlighted_and_bookkeeping_columns_are_never_suggested(views_connection) -> None:
    extras = {column.name for column in schema_card.plays_extras(views_connection)}
    assert extras.isdisjoint(schema_card.PLAYS_HIGHLIGHT)
    assert extras.isdisjoint(schema_card.NEVER_SUGGESTED)


def test_suggested_columns_render_with_their_notes(views_connection) -> None:
    extras = schema_card.plays_extras(views_connection)
    block = schema_card.render_relevant(schema_card.relevant("field goal", extras))
    assert block.startswith("More plays columns, for this question:")
    assert "field_goal_result " in block
    assert "# 'made', 'missed', or 'blocked'" in block
    assert schema_card.render_relevant([]) == ""
