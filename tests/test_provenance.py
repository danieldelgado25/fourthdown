from __future__ import annotations

import shutil

import polars as pl
from typer.testing import CliRunner

from fourthdown.cli import app
from fourthdown.data import etl, ingest, modeling, provenance, warehouse


def _build(raw_frame, paths, season=2023):
    raw_frame.write_parquet(paths.season_raw(season))
    etl.transform_season(season, paths)
    warehouse.build(paths, run_tests=False)
    return provenance.load(paths)


def test_partition_records_rows_and_the_raw_file_it_came_from(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    partition = manifest.processed["2023-01"]
    assert (partition.season, partition.week) == (2023, 1)
    assert partition.rows == raw_frame.height
    assert partition.raw_sha256 == provenance.sha256_file(tmp_paths.season_raw(2023))
    assert partition.sha256 == provenance.sha256_file(tmp_paths.week_processed(2023, 1))
    assert partition.transform_sha256 == provenance.transform_sha256()


def test_every_dbt_model_is_recorded_with_what_it_reads(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    assert set(warehouse.VIEW_NAMES) <= set(manifest.models)
    assert {"fact_plays", "dim_game", "dim_team", "dim_player", "dim_season"} <= set(
        manifest.models
    )
    assert manifest.models["stg_plays"].reads == ("nflverse.processed_plays",)
    assert manifest.models["drives"].reads == ("dim_game", "fact_plays")
    assert manifest.models["fact_plays"].schema == "core"
    assert manifest.project_sha256 == modeling.project_sha256()


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
    assert again.models == manifest.models
    assert again.data_version == manifest.data_version


def test_verify_passes_on_untouched_files_and_catches_drift(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    assert provenance.verify(tmp_paths) == []
    raw_frame.head(1).write_parquet(tmp_paths.season_raw(2023))
    problems = provenance.verify(tmp_paths)
    assert any("different raw file" in problem for problem in problems)


def test_an_unwritable_report_does_not_fail_a_passed_verify(raw_frame, tmp_paths, tmp_path):
    _build(raw_frame, tmp_paths)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    args = ["lineage", "--data-dir", str(tmp_paths.root), "--output", str(blocker / "x.md")]
    verified = CliRunner().invoke(app, [*args, "--verify"])
    assert verified.exit_code == 0, verified.output
    assert "partitions match" in verified.output and "pass --output" in verified.output
    assert CliRunner().invoke(app, args).exit_code == 1


def test_verify_catches_an_edited_partition(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    with tmp_paths.week_processed(2023, 1).open("ab") as handle:
        handle.write(b"tampered")
    assert any("changed since" in problem for problem in provenance.verify(tmp_paths))


def test_verify_catches_a_model_whose_sql_changed(raw_frame, tmp_paths, tmp_path, monkeypatch):
    # A pipeline that only checks data files would keep serving the old tables.
    project = tmp_path / "dbt_project"
    shutil.copytree(modeling.project_dir(), project)
    monkeypatch.setenv(modeling.PROJECT_ENV, str(project))
    _build(raw_frame, tmp_paths)
    games = project / "models" / "semantic" / "games.sql"
    games.write_text(games.read_text() + "-- edited\n")
    assert provenance.verify(tmp_paths) == [
        "model games: models/semantic/games.sql changed since; rerun warehouse"
    ]
    (project / "macros" / "generic_tests_extra.sql").write_text("-- new macro\n")
    assert any(problem.startswith("dbt project:") for problem in provenance.verify(tmp_paths))


def test_verify_catches_a_players_file_that_arrived_or_changed(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    players = pl.DataFrame({"gsis_id": ["X"], "display_name": ["A B"], "position": ["QB"]})
    players.write_parquet(tmp_paths.players)
    assert provenance.verify(tmp_paths) == [
        "players: file arrived since the warehouse was built; rerun warehouse"
    ]
    warehouse.build(tmp_paths, run_tests=False)
    assert provenance.verify(tmp_paths) == []
    players.with_columns(display_name=pl.lit("C D")).write_parquet(tmp_paths.players)
    assert any(problem.startswith("players:") for problem in provenance.verify(tmp_paths))


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


def test_render_lists_models_partitions_and_sources(raw_frame, tmp_paths):
    manifest = _build(raw_frame, tmp_paths)
    text = provenance.render(manifest)
    assert manifest.data_version in text
    for name in warehouse.VIEW_NAMES:
        assert f"| {name} |" in text


def test_views_built_from_the_manifest_still_query(raw_frame, tmp_paths):
    _build(raw_frame, tmp_paths)
    with warehouse.connect(tmp_paths) as connection:
        assert connection.execute("SELECT count(*) FROM plays").fetchone() == (raw_frame.height,)
