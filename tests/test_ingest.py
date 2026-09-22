from __future__ import annotations

import pytest

from fourthdown.data import ingest


def test_parse_seasons_expands_ranges_and_deduplicates():
    assert ingest.parse_seasons("2009-2011,2011,2015") == [2009, 2010, 2011, 2015]


def test_parse_seasons_uses_latest_for_open_ended_range():
    assert ingest.parse_seasons("2022-", latest=2024) == [2022, 2023, 2024]


def test_parse_seasons_rejects_open_ended_range_without_latest():
    with pytest.raises(ValueError, match="needs a `latest` season"):
        ingest.parse_seasons("2022-")


def test_parse_seasons_rejects_backwards_range():
    with pytest.raises(ValueError, match="ends before it starts"):
        ingest.parse_seasons("2016-2009")


def test_parse_seasons_rejects_seasons_before_the_archive_starts():
    with pytest.raises(ValueError, match="nflverse starts at"):
        ingest.parse_seasons("1995")


def test_season_url_points_at_the_release_asset():
    assert ingest.season_url(2016).endswith("/pbp/play_by_play_2016.parquet")


def test_download_season_skips_an_existing_file(tmp_paths, monkeypatch):
    destination = tmp_paths.season_raw(2016)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"cached")

    def fail(*args, **kwargs):
        raise AssertionError("network call attempted for a cached season")

    monkeypatch.setattr(ingest.requests, "get", fail)
    assert ingest.download_season(2016, tmp_paths) == destination
    assert destination.read_bytes() == b"cached"
