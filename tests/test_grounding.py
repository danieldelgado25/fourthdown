"""Every number a stats answer states must come from its result (evaluation.grounding)."""

from __future__ import annotations

import pytest

from fourthdown.evaluation import grounding, scorecard, suites

COLUMNS = ("player_name", "game_id", "passing_yards", "epa_per_play")
ROWS = [["J.Burrow", "2022_07_ATL_CIN", 481, 0.3679261125104954]]


def test_numbers_finds_standalone_numbers_only():
    assert grounding.numbers("Q4 drive: 1,234 yards, -3.5 EPA, 12% in 2022") == [
        "1,234",
        "-3.5",
        "12",
        "2022",
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "passing_yards: 481",  # a numeric cell
        "epa 0.37",  # the same cell, at the precision printed
        "game 2022_07_ATL_CIN in 2022",  # part of a text cell
        "1 row(s); top: player_name J.Burrow",  # the row count
        "in week 7",  # '07' is part of the game id
    ],
)
def test_supported_numbers(answer):
    assert grounding.unsupported(answer, COLUMNS, ROWS) == []


@pytest.mark.parametrize(
    ("answer", "invented"),
    [
        ("passing_yards: 518", ["518"]),  # a different number
        ("epa 0.4", []),  # 0.368 to one place is 0.4
        ("epa 0.5", ["0.5"]),  # rounded wrong
        ("481 yards and 3 touchdowns", ["3"]),  # one claim too many
        ("in 2021", ["2021"]),
    ],
)
def test_numbers_are_checked_against_the_result(answer, invented):
    assert grounding.unsupported(answer, COLUMNS, ROWS) == invented


def test_the_question_may_be_repeated_back():
    assert grounding.unsupported("in 2019: 481", COLUMNS, ROWS, question="... in 2019?") == []


def test_check_skips_empty_answers_and_reports_the_rest():
    report = grounding.check(
        [
            ("ok", "q", "passing_yards: 481", COLUMNS, ROWS),
            ("bad", "q", "passing_yards: 999", COLUMNS, ROWS),
            ("empty", "q", "No rows matched. Check 2009-2024.", ("x",), [[None]]),
        ]
    )
    assert report.checked == 2 and report.skipped == 1
    assert report.grounded == 1 and report.numbers_unsupported == 1
    (finding,) = report.findings()
    assert "`bad` stated 999" in finding


def test_grounding_suite_passes_its_gate_on_the_synthetic_season(views_connection):
    result = suites.grounding_suite(views_connection)
    assert result.metrics["answers_checked"] > 0
    assert result.metrics["grounded"] == 1.0
    assert result.metrics["unsupported_numbers"] == 0.0
    gate = next(gate for gate in scorecard.load_gates() if gate.key == "grounding.grounded")
    assert gate.admits(result.metrics["grounded"])


def test_grounding_runs_offline_and_in_ci():
    assert suites.GROUNDING in suites.OFFLINE
