"""The gateway's query log and its summary (fourthdown.feedback)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from fourthdown import feedback
from fourthdown.cli import app
from fourthdown.feedback import QueryLog, percentile, query_entry, summarise


def entry(ms: float, *, tool: str = "stats", by: str = "keyword", **extra) -> dict:
    record = query_entry(
        question=f"q{ms}", tool=tool, decided_by=by, elapsed_seconds=ms / 1000, **extra
    )
    return record


def test_query_entry_records_sql_rows_and_review():
    record = query_entry(
        question="sacks?",
        tool="stats",
        decided_by="llm",
        elapsed_seconds=1.2345,
        detail={"sql": "SELECT 1", "review": "expected_empty"},
        rows=[[None]],
        guest=True,
    )
    assert record == {
        "question": "sacks?",
        "tool": "stats",
        "decided_by": "llm",
        "elapsed_ms": 1234.5,
        "sql_ran": True,
        "rows": 1,
        "empty": True,
        "review": "expected_empty",
        "failed": False,
        "error": None,
        "guest": True,
    }


def test_an_answer_without_a_table_has_no_row_count():
    record = query_entry(question="q", tool="narrative", decided_by="llm", elapsed_seconds=0.1)
    assert record["rows"] is None and record["empty"] is None and not record["sql_ran"]


@pytest.mark.parametrize(
    ("fraction", "expected"), [(0.5, 50.0), (0.95, 95.0), (0.99, 99.0), (1.0, 100.0)]
)
def test_percentile_is_nearest_rank(fraction, expected):
    assert percentile([float(value) for value in range(100, 0, -1)], fraction) == expected


def test_percentile_of_nothing():
    assert percentile([], 0.5) is None


def test_summarise_reports_latency_errors_empties_and_route_mix():
    records = [
        entry(100, detail={"sql": "x"}, rows=[[1]]),
        entry(200, detail={"sql": "x"}, rows=[[None]]),
        entry(300, tool="narrative", by="llm"),
        entry(4000, tool="advisor", by="llm", failed=True, error="models down"),
    ]
    summary = summarise(records, slowest=1)
    assert summary.total == 4
    assert summary.p50_ms == 200.0
    assert summary.p95_ms == 4000.0
    assert summary.error_rate == 0.25
    assert summary.sql_ran == 2 and summary.empty_rate == 0.5
    assert summary.tools == {"stats": 2, "narrative": 1, "advisor": 1}
    assert summary.decided_by == {"keyword": 2, "llm": 2}
    assert [record["tool"] for record in summary.slowest] == ["advisor"]
    text = summary.render()
    assert "p50 200 ms, p95 4,000 ms" in text
    assert "errors       25.0%" in text
    assert "stats 2 (50%)" in text


def test_summarise_nothing():
    summary = summarise([])
    assert summary.total == 0 and summary.render() == "no queries recorded"


def test_since_keeps_recent_records_and_skips_unstamped_ones():
    now = datetime.now(timezone.utc)
    records = [
        {"at": (now - timedelta(hours=5)).isoformat()},
        {"at": (now - timedelta(minutes=5)).isoformat()},
        {"at": "not a time"},
    ]
    assert len(feedback.since(records, 1)) == 1
    assert len(feedback.since(records, None)) == 3


def test_query_log_appends_stamped_lines(tmp_path):
    log = QueryLog(tmp_path)
    log.record(entry(10))
    log.record(entry(20))
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["elapsed_ms"] for line in lines] == [10.0, 20.0]
    assert all("at" in json.loads(line) for line in lines)


def test_queries_command_summarises_the_log(tmp_path):
    log = QueryLog(tmp_path / "feedback")
    log.path.parent.mkdir()
    for ms in (100, 200, 300):
        log.record(entry(ms, detail={"sql": "x"}, rows=[[1]]))
    result = CliRunner().invoke(app, ["queries", "--data-dir", str(tmp_path), "--hours", "1"])
    assert result.exit_code == 0, result.output
    assert "queries      3" in result.output
    assert "p50 200 ms" in result.output


def test_queries_command_on_an_empty_log(tmp_path):
    result = CliRunner().invoke(app, ["queries", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no queries recorded" in result.output
