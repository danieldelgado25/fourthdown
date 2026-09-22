"""Dataset audit: per-season coverage plus the sanity checks the tests assert on.

The checks encode what we believe about the data (a season has 32 teams and ~256 regular
season games; league pass rate sits in the high 50s; EPA is roughly zero-sum). They run
in CI against whatever seasons are on disk, so a silently broken transform or a changed
upstream release surfaces as a failure rather than as a wrong answer three phases later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

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


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    @property
    def marker(self) -> str:
        return "pass" if self.passed else "FAIL"


def _scalar(connection: duckdb.DuckDBPyConnection, sql: str) -> float | int | None:
    row = connection.execute(sql).fetchone()
    return None if row is None else row[0]


def run_checks(connection: duckdb.DuckDBPyConnection) -> list[Check]:
    """Validate the processed table against known properties of NFL play-by-play data."""
    checks: list[Check] = []

    duplicates = _scalar(
        connection,
        "SELECT count(*) FROM (SELECT game_id, play_id FROM plays "
        "GROUP BY game_id, play_id HAVING count(*) > 1)",
    )
    checks.append(
        Check("unique play keys", duplicates == 0, f"{duplicates} duplicated (game_id, play_id)")
    )

    teams = _scalar(
        connection,
        "SELECT max(teams) FROM (SELECT season, count(DISTINCT posteam) AS teams FROM plays "
        "WHERE posteam IS NOT NULL GROUP BY season)",
    )
    checks.append(Check("32 teams per season", teams == 32, f"max distinct posteam = {teams}"))

    regular_games = _scalar(
        connection,
        "SELECT min(games) FROM (SELECT season, count(DISTINCT game_id) AS games FROM plays "
        "WHERE season_type = 'REG' GROUP BY season)",
    )
    checks.append(
        Check(
            "full regular seasons",
            regular_games is not None and regular_games >= 256,
            f"fewest regular season games in a season = {regular_games}",
        )
    )

    pass_rate = _scalar(
        connection,
        "SELECT avg(is_pass_call::INT) FROM plays WHERE is_designed_play",
    )
    checks.append(
        Check(
            "league pass rate in [0.52, 0.65]",
            pass_rate is not None and 0.52 <= pass_rate <= 0.65,
            f"pass rate = {pass_rate:.4f}" if pass_rate is not None else "no designed plays",
        )
    )

    epa = _scalar(connection, "SELECT avg(epa) FROM plays WHERE is_designed_play")
    checks.append(
        Check(
            "EPA per play near zero",
            epa is not None and abs(epa) < 0.05,
            f"mean EPA = {epa:.4f}" if epa is not None else "no EPA",
        )
    )

    down_range = _scalar(
        connection,
        "SELECT count(*) FROM plays WHERE down IS NOT NULL AND (down < 1 OR down > 4)",
    )
    checks.append(Check("downs within 1-4", down_range == 0, f"{down_range} out-of-range downs"))

    yardline_range = _scalar(
        connection,
        "SELECT count(*) FROM plays WHERE yardline_100 IS NOT NULL "
        "AND (yardline_100 < 0 OR yardline_100 > 100)",
    )
    checks.append(
        Check(
            "yardline_100 within 0-100",
            yardline_range == 0,
            f"{yardline_range} out-of-range yardlines",
        )
    )

    orphan_calls = _scalar(
        connection,
        "SELECT count(*) FROM plays WHERE is_pass_call IS NOT NULL AND NOT is_designed_play",
    )
    checks.append(
        Check(
            "play-call label only on designed plays",
            orphan_calls == 0,
            f"{orphan_calls} labelled non-designed plays",
        )
    )

    label_nulls = _scalar(
        connection,
        "SELECT avg(CASE WHEN posteam_won IS NULL THEN 1 ELSE 0 END) FROM plays "
        "WHERE posteam IS NOT NULL",
    )
    checks.append(
        Check(
            "win label present (ties excepted)",
            label_nulls is not None and label_nulls < 0.01,
            f"null rate = {label_nulls:.4f}" if label_nulls is not None else "no rows",
        )
    )

    return checks


def _markdown_table(columns: list[str], rows: list[tuple]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_report(connection: duckdb.DuckDBPyConnection) -> str:
    """Build the Markdown audit report for the seasons currently in the warehouse."""
    summary = connection.execute(SUMMARY_QUERY)
    summary_columns = [description[0] for description in summary.description]
    summary_rows = summary.fetchall()

    play_types = connection.execute(PLAY_TYPE_QUERY)
    play_type_columns = [description[0] for description in play_types.description]
    play_type_rows = play_types.fetchall()

    checks = run_checks(connection)
    check_rows = [(check.name, check.marker, check.detail) for check in checks]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# Data audit

Generated {generated} by `fourthdown audit`. Source: nflverse play-by-play releases.

## Checks

{_markdown_table(["check", "result", "detail"], check_rows)}

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


def write_report(connection: duckdb.DuckDBPyConnection, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(connection), encoding="utf-8")
    return output
