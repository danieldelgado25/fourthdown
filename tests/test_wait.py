"""`fourthdown wait`: the readiness gate Kubernetes init containers run."""

from __future__ import annotations

from typer.testing import CliRunner

from fourthdown.cli import app
from fourthdown.config import Paths
from fourthdown.models.card import MODEL_CARD
from fourthdown.serving import wait

runner = CliRunner()


def test_data_check_needs_warehouse_and_model_card(tmp_paths: Paths) -> None:
    check = wait.data_check(tmp_paths)
    assert "fourthdown.duckdb" in (check() or "")
    tmp_paths.database.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths.database.touch()
    assert MODEL_CARD in (check() or "")
    tmp_paths.models.mkdir(parents=True, exist_ok=True)
    (tmp_paths.models / MODEL_CARD).write_text("{}")
    assert check() is None


def test_wait_for_polls_until_ready() -> None:
    answers = iter(["not yet", "not yet", None])
    naps: list[float] = []
    pending = wait.wait_for(
        {"thing": lambda: next(answers)}, timeout=60, interval=2, sleep=naps.append
    )
    assert pending == {}
    assert naps == [2, 2]


def test_wait_for_reports_what_timed_out() -> None:
    now = iter([0.0, 0.0, 5.0, 11.0])
    pending = wait.wait_for(
        {"ok": lambda: None, "slow": lambda: "still building"},
        timeout=10,
        interval=5,
        sleep=lambda _: None,
        clock=lambda: next(now),
    )
    assert pending == {"slow": "still building"}


def test_wait_for_with_zero_timeout_checks_once() -> None:
    calls: list[int] = []

    def check() -> str:
        calls.append(1)
        return "missing"

    assert wait.wait_for({"x": check}, timeout=0) == {"x": "missing"}
    assert len(calls) == 1


def test_cli_exits_nonzero_with_the_reason(tmp_paths: Paths) -> None:
    result = runner.invoke(
        app, ["wait", "--data", "--timeout", "0", "--data-dir", str(tmp_paths.root)]
    )
    assert result.exit_code == 1
    assert "not ready: data" in result.output


def test_cli_succeeds_when_data_is_present(tmp_paths: Paths) -> None:
    tmp_paths.database.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths.database.touch()
    tmp_paths.models.mkdir(parents=True, exist_ok=True)
    (tmp_paths.models / MODEL_CARD).write_text("{}")
    result = runner.invoke(
        app, ["wait", "--data", "--timeout", "0", "--data-dir", str(tmp_paths.root)]
    )
    assert result.exit_code == 0, result.output
    assert "ready: data" in result.output


def test_cli_requires_a_check() -> None:
    assert runner.invoke(app, ["wait"]).exit_code != 0
