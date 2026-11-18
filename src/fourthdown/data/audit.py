"""Dataset audit: per-season coverage, plus the results of the warehouse's data tests.

What we believe about the data (a season has 32 teams and ~256 regular-season games;
league pass rate sits in the high 50s; EPA is roughly zero-sum; every key resolves to its
dimension) is encoded as dbt tests in ``dbt_project/``, which run on every warehouse
build and stop a bad one from replacing the live warehouse. This report lists their
results from the last build beside the coverage tables, so a silently broken transform
or a changed upstream release surfaces as a failure rather than as a wrong answer later.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from fourthdown.data.modeling import TestResult

SUMMARY_QUERY = """
SELECT
    season,
    count(DISTINCT game_id) AS games,
    count(*) AS plays,
    count(*) FILTER (WHERE is_designed_play) AS designed_plays,
    round(avg(is_pass_call::INT) FILTER (WHERE is_designed_play), 4) AS pass_rate,
    round(avg(is_pass_call::INT) FILTER (WHERE is_designed_play AND is_neutral_script), 4)
        AS neutral_pass_rate,
    round(avg(epa) FILTER (WHERE is_designed_play), 4) AS epa_per_play,
    round(avg(CASE WHEN epa IS NULL THEN 1 ELSE 0 END), 4) AS epa_null_rate,
    round(avg(CASE WHEN wp IS NULL THEN 1 ELSE 0 END), 4) AS wp_null_rate,
    round(avg(CASE WHEN play_desc IS NULL THEN 1 ELSE 0 END), 4) AS desc_null_rate
FROM plays
GROUP BY season
ORDER BY season
"""

PLAY_TYPE_QUERY = """
SELECT coalesce(play_type, '(null)') AS play_type, count(*) AS plays,
       round(count(*) * 1.0 / sum(count(*)) OVER (), 4) AS share
FROM plays
GROUP BY play_type
ORDER BY plays DESC
"""


def _detail(test: TestResult) -> str:
    if test.passed:
        return "0 failing rows"
    return f"{test.failures} failing rows" if test.failures else test.message


def _markdown_table(columns: list[str], rows: list[tuple]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_report(connection: duckdb.DuckDBPyConnection, tests: Sequence[TestResult] | None) -> str:
    """Build the Markdown audit report for the seasons currently in the warehouse."""
    summary = connection.execute(SUMMARY_QUERY)
    summary_columns = [description[0] for description in summary.description]
    summary_rows = summary.fetchall()

    play_types = connection.execute(PLAY_TYPE_QUERY)
    play_type_columns = [description[0] for description in play_types.description]
    play_type_rows = play_types.fetchall()

    if tests:
        failed = [test for test in tests if not test.passed]
        check_rows = [
            (test.name, "pass" if test.passed else "FAIL", _detail(test))
            for test in sorted(tests, key=lambda test: (test.passed, test.name))
        ]
        checks = (
            f"{len(tests) - len(failed)} of {len(tests)} dbt data tests passed on the last "
            "warehouse build (`dbt build` in `fourthdown warehouse`), failures first.\n\n"
            + _markdown_table(["test", "result", "detail"], check_rows)
        )
    else:
        checks = "No data test results: the last warehouse build ran without tests."

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# Data audit

Generated {generated} by `fourthdown audit`. Source: nflverse play-by-play releases.

## Data tests

{checks}

## Coverage by season

{_markdown_table(summary_columns, summary_rows)}

`pass_rate` counts designed run/pass plays only (kneels, spikes, aborted snaps, and
special teams excluded). `neutral_pass_rate` further restricts to neutral game script:
win probability between 0.2 and 0.8, first three quarters, outside the two-minute
warning. The gap between the two is the game-script effect that a naive tendency stat
mistakes for coaching philosophy.

## Play type distribution

{_markdown_table(play_type_columns, play_type_rows)}
"""


def write_report(
    connection: duckdb.DuckDBPyConnection, tests: Sequence[TestResult] | None, output: Path
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(connection, tests), encoding="utf-8")
    return output
