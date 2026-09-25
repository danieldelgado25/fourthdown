from __future__ import annotations

import pytest

from fourthdown.narrative import render
from fourthdown.narrative.teams import name, nickname


def test_current_names_for_relocated_franchises() -> None:
    # The ETL rewrites SD/STL/OAK to the current codes, so only those need a name.
    assert name("LAC") == "Los Angeles Chargers"
    assert name("LV") == "Las Vegas Raiders"
    assert nickname("SF") == "49ers"
    assert name(None) == "unknown"
    assert name("XXX") == "XXX"


@pytest.mark.parametrize(
    ("season", "season_type", "week", "expected"),
    [
        (2023, "REG", 7, "Week 7"),
        (2023, "POST", 19, "Wild Card round"),
        (2023, "POST", 20, "Divisional round"),
        (2023, "POST", 21, "Conference Championship"),
        (2023, "POST", 22, "Super Bowl LVIII (58)"),
        (2014, "POST", 21, "Super Bowl XLIX (49)"),
        # 2021 was the first 18-week regular season, so the playoff weeks shift by one.
        (2020, "POST", 21, "Super Bowl LV (55)"),
        (2015, "POST", 21, "Super Bowl 50"),
    ],
)
def test_week_label(season: int, season_type: str, week: int, expected: str) -> None:
    assert render.week_label(season, season_type, week) == expected


def test_yardline_reads_from_the_offence_point_of_view() -> None:
    assert render._yardline(75, "AAA") == "their own 25"
    assert render._yardline(50, "AAA") == "midfield"
    assert render._yardline(12, "AAA") == "the AAA 12"
    assert render._yardline(None, "AAA") == "an unknown spot"


def test_game_document_carries_result_and_metadata(views_connection) -> None:
    documents = list(render.game_documents(views_connection))
    assert len(documents) == 1
    document = documents[0]
    assert document.doc_id == "game:2023_01_AAA_BBB"
    assert document.grain == "game"
    assert document.season == 2023
    assert set(document.teams) == {"AAA", "BBB"}
    assert document.title == "2023 Week 1: AAA at BBB"
    assert "Final score" in document.body
    assert document.text.startswith(document.title)


def test_drive_documents_describe_each_drive(views_connection) -> None:
    documents = list(render.drive_documents(views_connection))
    assert documents
    body = documents[0].body
    assert "plays" in body and "Result:" in body
    assert all(document.grain == "drive" for document in documents)
    assert all(document.game_id == "2023_01_AAA_BBB" for document in documents)


def test_playoff_drives_skip_the_regular_season(views_connection) -> None:
    assert list(render.drive_documents(views_connection, postseason_only=True)) == []
    assert (
        list(render.documents(views_connection, grains=("drive",), postseason_drives_only=True))
        == []
    )
    assert list(render.drive_documents(views_connection))


def test_season_filter_excludes_other_seasons(views_connection) -> None:
    assert list(render.documents(views_connection, seasons=[1999])) == []
    assert list(render.documents(views_connection, seasons=[2023]))


def test_documents_streams_only_the_requested_grains(views_connection) -> None:
    grains = {document.grain for document in render.documents(views_connection, grains=("game",))}
    assert grains == {"game"}
