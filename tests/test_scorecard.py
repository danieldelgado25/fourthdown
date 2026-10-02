"""The scorecard: gate arithmetic, skip handling, and the suites that run offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fourthdown.cli import app
from fourthdown.evaluation import scorecard, suites
from fourthdown.evaluation.scorecard import Gate, Status, SuiteResult
from fourthdown.models.features import SeasonSplit

FLOOR = Gate(key="demo.accuracy", minimum=0.8, maximum=None, why="floor")
BAND = Gate(key="demo.auc", minimum=0.7, maximum=0.95, why="band")


def test_gates_file_loads_and_every_gate_names_a_known_suite() -> None:
    gates = scorecard.load_gates()
    assert gates
    for gate in gates:
        assert gate.suite in suites.ALL, gate.key
        assert gate.metric, gate.key
        assert gate.minimum is not None or gate.maximum is not None, gate.key
        assert gate.why, gate.key


def test_gate_bounds() -> None:
    assert FLOOR.admits(0.8)
    assert not FLOOR.admits(0.79)
    assert BAND.admits(0.9)
    assert not BAND.admits(0.97)
    assert not BAND.admits(float("nan"))
    assert BAND.bound() == "0.7 to 0.95"


def test_verdicts_pass_fail_and_missing_metric() -> None:
    card = scorecard.build(
        [SuiteResult("demo", metrics={"accuracy": 0.9, "auc": 0.99})], gates=[FLOOR, BAND]
    )
    statuses = {verdict.gate.key: verdict.status for verdict in card.verdicts()}
    assert statuses == {"demo.accuracy": Status.PASS, "demo.auc": Status.FAIL}
    assert not card.passed

    renamed = scorecard.build([SuiteResult("demo", metrics={"acc": 0.9})], gates=[FLOOR])
    assert [verdict.status for verdict in renamed.verdicts()] == [Status.FAIL]


def test_skipped_suite_only_fails_when_required() -> None:
    skipped = [SuiteResult("demo", skipped="no Ollama")]
    optional = scorecard.build(skipped, gates=[FLOOR])
    assert optional.passed
    assert [verdict.status for verdict in optional.verdicts()] == [Status.SKIPPED]

    required = scorecard.build(skipped, gates=[FLOOR], required=["demo"])
    assert not required.passed
    assert required.missing_required() == ["demo"]


def test_a_required_suite_that_was_never_selected_fails() -> None:
    card = scorecard.build([], gates=[FLOOR], required=["demo"])
    assert card.missing_required() == ["demo"]


def test_gates_for_unselected_suites_are_ignored() -> None:
    card = scorecard.build([SuiteResult("other", metrics={"x": 1.0})], gates=[FLOOR])
    assert card.verdicts() == []
    assert card.passed


def test_render_and_json_report_the_same_verdicts() -> None:
    card = scorecard.build(
        [
            SuiteResult("demo", metrics={"accuracy": 0.5, "extra": 3.0}, findings=("miss",)),
            SuiteResult("later", skipped="no Postgres"),
        ],
        gates=[FLOOR],
    )
    text = card.render()
    assert "**FAIL**" in text
    assert "| demo.accuracy | 0.5 | >= 0.8 | fail | floor |" in text
    assert "demo.extra" in text
    assert "skipped: no Postgres" in text
    assert "- **demo**: miss" in text

    payload = json.loads(card.to_json())
    assert payload["passed"] is False
    assert payload["gates"] == [
        {"key": "demo.accuracy", "min": 0.8, "max": None, "value": 0.5, "status": "fail"}
    ]


def test_guard_suite_blocks_every_attack_and_allows_every_legitimate_query() -> None:
    result = suites.guard_suite()
    assert result.findings == ()
    assert result.metrics["attacks_blocked"] == 1.0
    assert result.metrics["legitimate_allowed"] == 1.0
    assert result.metrics["row_caps_enforced"] == 1.0
    assert result.metrics["attack_cases"] >= 30


def test_keyword_routing_suite_reports_per_tool_recall() -> None:
    result = suites.routing_suite(
        suites.KeywordRouter(suites.routing_harness.catalog()), suites.ROUTING_KEYWORD
    )
    accuracy, worst = result.metrics["accuracy"], result.metrics["worst_tool_recall"]
    assert 0.0 <= worst <= 1.0
    assert worst <= 1.0 if accuracy == 1.0 else worst < 1.0
    assert len(result.findings) == round(
        (1 - accuracy) * len(suites.routing_harness.load_questions())
    )


def test_references_suite_runs_every_reference_query(views_connection) -> None:
    result = suites.references_suite(views_connection)
    assert result.metrics["runnable"] == 1.0, result.findings


def test_season_split_must_be_disjoint_and_in_order() -> None:
    SeasonSplit(train=(2019, 2020), valid=(2021,), test=(2022,))
    with pytest.raises(ValueError, match="time order"):
        SeasonSplit(train=(2019, 2021), valid=(2020,), test=(2022,))
    with pytest.raises(ValueError, match="at least one"):
        SeasonSplit(train=(2019,), valid=(), test=(2022,))


def test_unknown_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        suites.run(["vibes"], options=None)  # type: ignore[arg-type]


runner = CliRunner()


def test_cli_writes_a_passing_scorecard(tmp_path: Path) -> None:
    output, payload = tmp_path / "card.md", tmp_path / "card.json"
    result = runner.invoke(
        app,
        [
            "scorecard",
            "--suites",
            "guard,routing_keyword",
            "--require",
            "guard",
            "--output",
            str(output),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "**PASS**" in output.read_text()
    assert json.loads(payload.read_text())["passed"] is True


def test_cli_fails_when_a_required_suite_did_not_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "scorecard",
            "--suites",
            "guard",
            "--require",
            "guard,routing_keyword",
            "--output",
            str(tmp_path / "card.md"),
        ],
    )
    assert result.exit_code == 1
    assert "required suite routing_keyword did not run" in result.output


def test_cli_needs_all_three_splits_or_none(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "scorecard",
            "--suites",
            "guard",
            "--train-seasons",
            "2019-2021",
            "--output",
            str(tmp_path / "card.md"),
        ],
    )
    assert result.exit_code != 0
