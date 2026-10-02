from __future__ import annotations

import duckdb

from fourthdown.data import etl, ingest, provenance, warehouse


def _build(raw_frame, paths, season=2023):
    raw_frame.write_parquet(paths.season_raw(season))
    etl.transform_season(season, paths)
    warehouse.build(paths)
    return provenance.load(paths)


def test_partition_records_rows_and_the_raw_file_it_came_from(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    partition = manifest.processed[2023]
    assert partition.rows == raw_frame.height
    assert partition.raw_sha256 == provenance.sha256_file(tmp_paths.season_raw(2023))
    assert partition.sha256 == provenance.sha256_file(tmp_paths.season_processed(2023))
    assert partition.transform_sha256 == provenance.transform_sha256()


def test_every_view_is_recorded_with_what_it_reads(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    assert set(manifest.views) == set(warehouse.VIEW_NAMES)
    assert manifest.views["plays"].reads == (provenance.PROCESSED_SOURCE,)
    assert manifest.views["drives"].reads == ("plays",)


def test_data_version_is_stable_across_rebuilds_of_the_same_bytes(raw_frame, tmp_paths):
    first = _build(raw_frame, tmp_paths).data_version
    second = _build(raw_frame, tmp_paths).data_version
    assert first == second
    assert len(first) == provenance.DATA_VERSION_CHARS


def test_data_version_changes_with_the_data(raw_frame, tmp_paths):
    before = _build(raw_frame, tmp_paths).data_version
    after = _build(raw_frame.head(2), tmp_paths).data_version
    assert before != after


def test_manifest_round_trips_through_json(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    again = provenance.Manifest.from_json(manifest.to_json())
    assert again.processed == manifest.processed
    assert again.views == manifest.views
    assert again.data_version == manifest.data_version


def test_verify_passes_on_untouched_files_and_catches_drift(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    assert provenance.verify(tmp_paths) == []
    raw_frame.head(1).write_parquet(tmp_paths.season_raw(2023))
    problems = provenance.verify(tmp_paths)
    assert any("different raw file" in problem for problem in problems)


def test_verify_catches_an_edited_partition(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    with tmp_paths.season_processed(2023).open("ab") as handle:
        handle.write(b"tampered")
    assert any("changed since" in problem for problem in provenance.verify(tmp_paths))


def test_verify_reports_an_empty_manifest(tmp_paths):
    assert provenance.verify(tmp_paths)


def test_ingest_records_a_cached_season_without_retrieval_details(tmp_paths, monkeypatch):
    destination = tmp_paths.season_raw(2016)
    destination.write_bytes(b"cached")
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: None)
    ingest.download_season(2016, tmp_paths)
    record = provenance.load(tmp_paths).raw[2016]
    assert record.sha256 == provenance.sha256_file(destination)
    assert record.url == ingest.season_url(2016)
    assert record.retrieved_at is None


def test_ingest_records_headers_of_a_fresh_download(tmp_paths, monkeypatch):
    class Response:
        headers = {"ETag": '"abc"', "Last-Modified": "Tue, 01 Sep 2026 00:00:00 GMT"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, size):
            yield b"season bytes"

    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: Response())
    ingest.download_season(2017, tmp_paths)
    record = provenance.load(tmp_paths).raw[2017]
    assert record.etag == '"abc"'
    assert record.retrieved_at is not None
    assert record.bytes == len(b"season bytes")


def test_render_lists_views_partitions_and_sources(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    text = provenance.render(manifest)
    assert manifest.data_version in text
    for name in warehouse.VIEW_NAMES:
        assert f"| {name} |" in text


def test_views_built_from_the_manifest_still_query(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    with duckdb.connect(str(tmp_paths.database), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM plays").fetchone() == (raw_frame.height,)
