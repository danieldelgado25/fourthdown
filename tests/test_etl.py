from __future__ import annotations

import polars as pl
import pytest

from fourthdown.data import etl, provenance, schema


def _transform(raw_frame: pl.DataFrame) -> pl.DataFrame:
    return etl.transform(raw_frame.lazy()).collect()


def test_transform_emits_the_contracted_columns(raw_frame):
    assert _transform(raw_frame).columns == list(schema.PROCESSED_COLUMNS)


def test_transform_renames_sql_hostile_columns(raw_frame):
    columns = _transform(raw_frame).columns
    assert "play_desc" in columns and "desc" not in columns
    assert "pass_play" in columns and "pass" not in columns


def test_situational_buckets(raw_frame):
    result = _transform(raw_frame).sort("play_id")
    assert result["distance_bucket"].to_list() == ["long", "short", "long", "very_long"]
    assert result["field_zone"].to_list() == [
        "own_territory",
        "red_zone",
        "midfield",
        "backed_up",
    ]
    assert result["score_state"].to_list() == [
        "tied",
        "trailing_big",
        "leading_one_score",
        "trailing_one_score",
    ]


def test_designed_play_excludes_kneels_and_special_teams(raw_frame):
    result = _transform(raw_frame).sort("play_id")
    assert result["is_designed_play"].to_list() == [True, True, False, False]
    # The play-call label exists only where a run/pass decision was actually made.
    assert result["is_pass_call"].to_list() == [True, False, None, None]


def test_game_script_flags(raw_frame):
    result = _transform(raw_frame).sort("play_id")
    assert result["is_neutral_script"].to_list() == [True, False, False, True]
    assert result["is_garbage_time"].to_list() == [False, True, True, False]


def test_posteam_won_follows_the_home_margin(raw_frame):
    # `result` is +7, so the home team (BBB) won and the away offense (AAA) lost.
    result = _transform(raw_frame).sort("play_id")
    assert result["posteam_won"].to_list() == [True, True, True, False]


def test_posteam_won_is_null_on_a_tie(raw_frame):
    tied = raw_frame.with_columns(pl.lit(0).cast(pl.Int32).alias("result"))
    assert _transform(tied)["posteam_won"].to_list() == [None] * raw_frame.height


def test_transform_fills_columns_a_season_does_not_publish(raw_frame):
    # Seasons before 2006 have no completion-probability columns; the plan must still run.
    older = raw_frame.drop("cpoe", "xpass", "pass_oe")
    result = _transform(older)
    assert result.columns == list(schema.PROCESSED_COLUMNS)
    assert result["cpoe"].null_count() == result.height


def _two_weeks(raw_frame: pl.DataFrame) -> pl.DataFrame:
    week_two = raw_frame.with_columns(
        week=pl.lit(2, dtype=raw_frame["week"].dtype), game_id=pl.lit("2023_02_BBB_AAA")
    )
    return pl.concat([raw_frame, week_two])


def test_transform_season_writes_one_partition_per_week(raw_frame, tmp_paths):
    _two_weeks(raw_frame).write_parquet(tmp_paths.season_raw(2023))
    written = etl.transform_season(2023, tmp_paths)
    assert written == [tmp_paths.week_processed(2023, 1), tmp_paths.week_processed(2023, 2)]
    assert [pl.read_parquet(path).height for path in written] == [raw_frame.height] * 2
    assert set(provenance.load(tmp_paths).processed) == {"2023-01", "2023-02"}


def test_transform_week_leaves_the_other_weeks_alone(raw_frame, tmp_paths):
    _two_weeks(raw_frame).write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    week_one = tmp_paths.week_processed(2023, 1)
    before = (week_one.stat().st_mtime_ns, provenance.load(tmp_paths).processed["2023-01"])
    etl.transform_week(2023, 2, tmp_paths)
    assert (week_one.stat().st_mtime_ns, provenance.load(tmp_paths).processed["2023-01"]) == before


def test_one_week_transformed_alone_matches_the_whole_season(raw_frame, tmp_paths):
    # Every derived column is row-local, so a week's partition does not depend on how
    # much of the season was transformed with it.
    _two_weeks(raw_frame).write_parquet(tmp_paths.season_raw(2023))
    etl.transform_season(2023, tmp_paths)
    whole = pl.read_parquet(tmp_paths.week_processed(2023, 2))
    etl.transform_week(2023, 2, tmp_paths)
    assert pl.read_parquet(tmp_paths.week_processed(2023, 2)).equals(whole)


def test_transform_week_refuses_a_week_not_in_the_file(raw_frame, tmp_paths):
    raw_frame.write_parquet(tmp_paths.season_raw(2023))
    with pytest.raises(ValueError, match="no plays for week 5"):
        etl.transform_week(2023, 5, tmp_paths)


def test_a_season_written_whole_is_split_into_weeks(raw_frame, tmp_paths):
    # Data roots from before weekly partitions hold season=2023/plays.parquet.
    _two_weeks(raw_frame).write_parquet(tmp_paths.season_raw(2023))
    legacy = tmp_paths.season_processed(2023) / "plays.parquet"
    legacy.parent.mkdir(parents=True)
    raw_frame.write_parquet(legacy)
    assert any("not in the manifest" in problem for problem in provenance.verify(tmp_paths))
    etl.transform_week(2023, 2, tmp_paths)
    assert not legacy.exists()
    assert sorted(path.name for path in legacy.parent.iterdir()) == [
        "week=01.parquet",
        "week=02.parquet",
    ]


def test_leakage_columns_are_kept_but_flagged():
    # They stay in the warehouse for analytics and baselines; phase 04 must not train on them.
    assert set(schema.PROCESSED_COLUMNS) >= schema.LEAKAGE_COLUMNS
    assert {"epa", "wp", "xpass", "pass_oe"} <= schema.LEAKAGE_COLUMNS
