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
    assert "more raw nflverse columns" in plays_block
    assert len(plays_block.splitlines()) < 45


def test_card_reports_coverage(views_connection) -> None:
    assert "seasons 2023-2023" in schema_card.build(views_connection)


def test_card_notes_are_view_specific(views_connection) -> None:
    card = schema_card.build(views_connection)
    assert "total EPA of the drive" in card


def test_card_stays_within_a_reasonable_prompt_budget(views_connection) -> None:
    assert len(schema_card.build(views_connection)) < 8000
