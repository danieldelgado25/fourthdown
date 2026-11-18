"""Data quality: contracts at ingestion, row counts, the leakage contract, and freshness."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl
import pytest

from fourthdown.data import etl, provenance
from fourthdown.models import features
from fourthdown.quality import DataQualityError, alerts, contracts, freshness, leakage, volume


def _processed(raw_frame: pl.DataFrame) -> pl.DataFrame:
    return etl.transform(raw_frame.lazy()).collect()


# --- the processed contract -----------------------------------------------------------------


def test_a_clean_partition_meets_the_processed_contract(raw_frame):
    contracts.validate_processed(_processed(raw_frame), season=2023, week=1)


def test_out_of_range_values_fail_with_the_column_bound_and_an_example(raw_frame):
    frame = _processed(raw_frame).with_columns(pl.lit(7, dtype=pl.Int32).alias("down"))
    with pytest.raises(DataQualityError) as raised:
        contracts.validate_processed(frame, season=2023, week=1)
    assert "down: 4 value(s) not between 1 and 4 (e.g. 7)" in str(raised.value)


def test_unknown_labels_and_duplicate_keys_both_fail(raw_frame):
    frame = _processed(raw_frame).with_columns(
        pl.when(pl.col("play_id") == 1)
        .then(pl.lit("hail_mary"))
        .otherwise("play_type")
        .alias("play_type")
    )
    frame = pl.concat([frame, frame.head(1)])
    with pytest.raises(DataQualityError) as raised:
        contracts.validate_processed(frame, season=2023, week=1)
    problems = raised.value.problems
    assert any("play_type" in p and "hail_mary" in p for p in problems)
    assert any("rows share a key" in p for p in problems)


def test_a_partition_holds_only_its_own_week(raw_frame):
    with pytest.raises(DataQualityError, match="week: 4 value"):
        contracts.validate_processed(_processed(raw_frame), season=2023, week=2)


def test_a_changed_type_fails_the_contract(raw_frame):
    frame = _processed(raw_frame).with_columns(pl.col("ydstogo").cast(pl.Float64))
    with pytest.raises(DataQualityError, match="ydstogo: expected Int32"):
        contracts.validate_processed(frame, season=2023, week=1)


# --- the raw contract -------------------------------------------------------------------------


def test_a_missing_required_column_fails_instead_of_loading_nulls(raw_frame):
    with pytest.raises(DataQualityError, match="missing column down"):
        contracts.validate_raw(raw_frame.drop("down"), season=2023, subject="raw")


def test_text_in_a_numeric_column_fails_instead_of_loading_nulls(raw_frame):
    frame = raw_frame.with_columns(pl.col("ydstogo").cast(pl.String))
    with pytest.raises(DataQualityError, match="ydstogo: expected numbers, got String"):
        contracts.validate_raw(frame, season=2023, subject="raw")


def test_a_file_from_another_season_fails(raw_frame):
    with pytest.raises(DataQualityError, match="season: 4 value"):
        contracts.validate_raw(raw_frame, season=2024, subject="raw")


def test_an_optional_column_may_be_missing(raw_frame):
    # Seasons before 2006 publish no completion probability; the ETL fills it with nulls.
    contracts.validate_raw(raw_frame.drop("cpoe", "xpass"), season=2023, subject="raw")


# --- the ETL refuses a corrupted file ---------------------------------------------------------


def test_a_corrupted_season_file_writes_nothing(raw_frame, tmp_paths):
    raw_frame.with_columns(pl.lit(9.0).alias("down")).write_parquet(tmp_paths.season_raw(2023))
    with pytest.raises(DataQualityError) as raised:
        etl.transform_season(2023, tmp_paths)
    assert str(raised.value).startswith("processed partition 2023-01 failed 1 data quality check")
    assert not list(tmp_paths.processed.glob("season=*/*.parquet"))
    assert not provenance.load(tmp_paths).processed


def test_a_corrupted_week_leaves_the_partition_it_had(raw_frame, tmp_paths):
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    written = etl.transform_week(2023, 1, tmp_paths)
    before = written.read_bytes()
    raw_frame.with_columns(pl.col("ydstogo").cast(pl.String)).write_parquet(
        tmp_paths.season_raw(2023)
    )
    with pytest.raises(DataQualityError, match="the nflverse 2023 file, week 1,"):
        etl.transform_week(2023, 1, tmp_paths)
    assert written.read_bytes() == before


def test_negative_timeouts_are_floored(raw_frame):
    # nflverse reads -1 after a third timeout in playoff overtime (2025 week 20).
    frame = raw_frame.with_columns(pl.lit(-1.0).alias("defteam_timeouts_remaining"))
    assert _processed(frame)["defteam_timeouts_remaining"].to_list() == [0] * raw_frame.height


# --- row counts ---------------------------------------------------------------------------------


def _week(plays_per_game: list[int]) -> pl.DataFrame:
    games = [f"2026_10_G{index:02d}_XX" for index in range(len(plays_per_game))]
    return pl.DataFrame(
        {"game_id": [g for g, n in zip(games, plays_per_game, strict=True) for _ in range(n)]}
    )


def test_a_normal_week_passes_the_volume_bands():
    assert volume.check_week(_week([170, 182, 165, 190]), "2026-10") == []


def test_a_truncated_game_fails():
    problems = volume.check_week(_week([170, 41, 180, 175, 178, 182, 190, 171]), "2026-10")
    assert problems == [
        "2026-10: 1 game(s) with fewer than 100 plays (2026_10_G01_XX has 41); "
        "a game published part-way, or a truncated file"
    ]


def test_duplicated_rows_fail():
    problems = volume.check_week(_week([350, 340]), "2026-10")
    assert any("more than 300 plays" in p for p in problems)
    assert any("plays per game on average" in p for p in problems)


def test_a_redelivered_week_may_move_a_little_but_not_a_lot():
    week = _week([175] * 14)
    assert volume.check_week(week, "2026-10", previous_rows=2_470) == []
    problems = volume.check_week(week, "2026-10", previous_rows=2_800)
    assert problems == [
        "2026-10: 2,450 plays, -12.5% against the 2,800 ingested before; a correction moves a "
        "week by a handful of plays, not 5%"
    ]


# --- the leakage contract ------------------------------------------------------------------------


def test_the_shipped_feature_lists_meet_the_declared_contract():
    assert leakage.declared_violations() == []


def test_an_nflfastr_output_as_a_feature_is_a_violation(monkeypatch):
    monkeypatch.setattr(features, "WP_FEATURES", (*features.WP_FEATURES, "wp"))
    assert leakage.declared_violations() == [
        "win probability: wp are nflfastR model outputs, never features"
    ]


def test_an_undeclared_feature_is_a_violation(monkeypatch):
    monkeypatch.setattr(features, "PLAYCALL_FEATURES", (*features.PLAYCALL_FEATURES, "air_yards"))
    (problem,) = leakage.declared_violations()
    assert problem.startswith("play call: air_yards not declared pre-snap")


def _modelling_frame(rows: int = 2_000) -> pl.DataFrame:
    """A frame shaped like features.load_plays, with independent noise for every feature."""
    import numpy as np

    rng = np.random.default_rng(0)
    columns = {
        name: rng.normal(size=rows) for name in {*features.WP_FEATURES, *features.PLAYCALL_FEATURES}
    }
    return pl.DataFrame(columns).with_columns(
        pl.Series("posteam_won", rng.integers(0, 2, rows)),
        pl.Series("is_pass_call", rng.integers(0, 2, rows)),
        pl.lit(True).alias("is_designed_play"),
        pl.lit(1.0).alias("down"),
    )


def test_features_without_signal_pass_the_measured_contract():
    report = leakage.LeakageReport()
    leakage.measured(_modelling_frame(), report)
    assert report.passed
    assert all(score < 0.6 for score in report.auc["win probability"].values())


def test_a_feature_that_carries_the_outcome_is_a_violation():
    # What `pass_play` does to the play call (AUC 1.0): known only after the snap.
    frame = _modelling_frame().with_columns(
        pl.col("is_pass_call").cast(pl.Float64).alias("shotgun")
    )
    report = leakage.LeakageReport()
    leakage.measured(frame, report)
    assert report.violations == [
        "play call: shotgun alone predicts is_pass_call with AUC 1.000 (> 0.95); "
        "it carries the outcome"
    ]


def test_auc_matches_its_definition():
    target = pl.Series([0, 0, 1, 1])
    assert leakage._auc(target, pl.Series([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert leakage._auc(target, pl.Series([0.9, 0.8, 0.1, 0.2])) == pytest.approx(1.0)


# --- freshness ---------------------------------------------------------------------------


@pytest.fixture()
def schedule(tmp_paths):
    def write(rows):
        pl.DataFrame(
            rows, schema=["season", "week", "gameday", "result"], orient="row"
        ).write_parquet(tmp_paths.schedule)
        return tmp_paths

    return write


SEASON = [
    (2026, 9, "2026-11-08", 3),
    (2026, 9, "2026-11-09", -7),
    (2026, 10, "2026-11-15", 10),
    (2026, 10, "2026-11-16", 1),
    (2026, 11, "2026-11-22", None),
]


def test_fresh_when_every_finished_week_is_ingested(schedule):
    report = freshness.check(schedule(SEASON), {"2026-09", "2026-10"}, date(2026, 11, 19))
    assert report.passed and report.in_season
    assert report.summary() == "2026: current through 2026-10"


def test_a_finished_week_is_due_two_days_after_its_last_game(schedule):
    paths = schedule(SEASON)
    assert freshness.check(paths, {"2026-09"}, date(2026, 11, 17)).passed
    late = freshness.check(paths, {"2026-09"}, date(2026, 11, 19))
    assert late.stale == [
        "2026-10 ended 2026-11-16 and is not ingested; is the weekly run failing?"
    ]


def test_results_missing_upstream_are_reported_as_nflverse_being_behind(schedule):
    report = freshness.check(schedule(SEASON), {"2026-09", "2026-10"}, date(2026, 11, 26))
    assert report.stale == [
        "2026-11: 1 game(s) played by 2026-11-24 still have no result in the nflverse "
        "schedule; nflverse is behind"
    ]


def test_nothing_is_due_out_of_season(schedule):
    report = freshness.check(schedule(SEASON), set(), date(2027, 6, 1))
    assert report.passed and not report.in_season


# --- alerts ------------------------------------------------------------------------------


def test_without_a_webhook_an_alert_is_a_log_line(monkeypatch, caplog):
    monkeypatch.delenv(alerts.WEBHOOK_ENV, raising=False)
    with caplog.at_level(logging.ERROR):
        assert alerts.send("data is behind", "2026-10 not ingested") is False
    assert "ALERT data is behind" in caplog.text


def test_a_webhook_gets_the_alert_and_its_url_is_never_logged(monkeypatch, caplog):
    sent = {}

    class Response:
        def raise_for_status(self):
            return None

    def post(url, json, timeout):
        sent.update(url=url, payload=json)
        return Response()

    secret = "https://hooks.example/T000/secret-token"
    monkeypatch.setenv(alerts.WEBHOOK_ENV, secret)
    monkeypatch.setattr(alerts.requests, "post", post)
    with caplog.at_level(logging.DEBUG):
        assert alerts.send("run failed", "raw_plays: down out of range") is True
    assert sent["url"] == secret
    assert sent["payload"]["text"] == sent["payload"]["content"]
    assert "raw_plays: down out of range" in sent["payload"]["text"]
    assert "secret-token" not in caplog.text
