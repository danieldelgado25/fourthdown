"""Content hashes and lineage for every table the pipeline produces.

Each step records what it wrote into one manifest at ``Paths.manifest``:

* ``raw``: the nflverse file for a season, its source URL, SHA-256, size, and the
  ETag and Last-Modified headers when it was downloaded. nflverse republishes the
  current season's file in place, so the hash is the only way to tell two downloads of
  "2024" apart.
* ``processed``: each weekly partition (keyed ``2026-10``, as the orchestrator keys it),
  its row count, its SHA-256, the hash of the raw season file it was derived from, and
  the hash of the transform code that derived it.
* ``models``: each dbt model and seed (``dbt_project/``) the warehouse was built from: its
  file's hash, schema, materialization, and what it reads. ``project_sha256`` covers the
  rest of the project that shapes the output (macros, the seed, ``dbt_project.yml``).
* ``players_sha256``: the nflverse player table the warehouse read, if any.
* ``built_from``: the partition hashes the live warehouse was built from, so the next
  build knows which weeks are new or changed (`fourthdown.data.warehouse.plan`).

``data_version`` folds the processed hashes and the dbt code into one short id.
A training run logs it, the model card carries it, and the serving API returns it with every
prediction, so a probability can be traced back to the exact bytes it was trained on.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fourthdown import __version__
from fourthdown.config import Paths, week_key
from fourthdown.data import modeling
from fourthdown.data.modeling import ModelRecord

CHUNK_SIZE = 1 << 20
DATA_VERSION_CHARS = 12
TRANSFORM_SOURCES = ("etl.py", "schema.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_sha256() -> str:
    """Hash of the code that turns a raw season into a processed partition."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in TRANSFORM_SOURCES:
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    """The checked-out commit, or None outside a git checkout (an installed wheel, a container)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RawRecord:
    season: int
    path: str
    url: str
    sha256: str
    bytes: int
    retrieved_at: str | None
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class PartitionRecord:
    season: int
    week: int
    path: str
    sha256: str
    rows: int
    raw_sha256: str
    transform_sha256: str
    written_at: str


@dataclass
class Manifest:
    raw: dict[int, RawRecord] = field(default_factory=dict)
    processed: dict[str, PartitionRecord] = field(default_factory=dict)
    models: dict[str, ModelRecord] = field(default_factory=dict)
    project_sha256: str | None = None
    players_sha256: str | None = None
    built_from: dict[str, str] = field(default_factory=dict)
    package_version: str = __version__
    git_commit: str | None = None
    updated_at: str | None = None

    @property
    def data_version(self) -> str:
        """One id for the processed bytes plus the dbt code that shapes them into tables."""
        digest = hashlib.sha256()
        for key in sorted(self.processed):
            digest.update(f"{key}:{self.processed[key].sha256}\n".encode())
        for name in sorted(self.models):
            digest.update(f"{name}:{self.models[name].sha256}\n".encode())
        if self.project_sha256:
            digest.update(f"project:{self.project_sha256}\n".encode())
        return digest.hexdigest()[:DATA_VERSION_CHARS]

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(sorted({record.season for record in self.processed.values()}))

    def to_json(self) -> str:
        payload = {
            "data_version": self.data_version,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "updated_at": self.updated_at,
            "raw": {str(season): asdict(record) for season, record in sorted(self.raw.items())},
            "processed": {key: asdict(record) for key, record in sorted(self.processed.items())},
            "models": {name: asdict(record) for name, record in self.models.items()},
            "project_sha256": self.project_sha256,
            "players_sha256": self.players_sha256,
            "built_from": dict(sorted(self.built_from.items())),
        }
        return json.dumps(payload, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        payload = json.loads(text)
        # A manifest from before dbt has "views" instead; it reads as having no models, so
        # `lineage --verify` asks for a rebuild.
        models = {
            name: ModelRecord(**{**record, "reads": tuple(record["reads"])})
            for name, record in payload.get("models", {}).items()
        }
        # Season-wide partitions from before weekly partitioning carry no week; they read
        # as unrecorded, so `lineage --verify` asks for the ETL to be rerun.
        processed = {
            key: PartitionRecord(**record)
            for key, record in payload.get("processed", {}).items()
            if "week" in record
        }
        return cls(
            raw={int(key): RawRecord(**record) for key, record in payload.get("raw", {}).items()},
            processed=processed,
            models=models,
            project_sha256=payload.get("project_sha256"),
            players_sha256=payload.get("players_sha256"),
            built_from=dict(payload.get("built_from", {})),
            package_version=str(payload.get("package_version", __version__)),
            git_commit=payload.get("git_commit"),
            updated_at=payload.get("updated_at"),
        )


def load(paths: Paths) -> Manifest:
    if not paths.manifest.exists():
        return Manifest()
    return Manifest.from_json(paths.manifest.read_text(encoding="utf-8"))


def save(paths: Paths, manifest: Manifest) -> None:
    manifest.git_commit = git_commit()
    manifest.updated_at = _now()
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = paths.manifest.with_suffix(".json.part")
    partial.write_text(manifest.to_json(), encoding="utf-8")
    partial.replace(paths.manifest)


def _relative(paths: Paths, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.root))
    except ValueError:
        return str(path)


def record_raw(
    paths: Paths,
    season: int,
    *,
    url: str,
    downloaded: bool,
    etag: str | None = None,
    last_modified: str | None = None,
) -> RawRecord:
    """Hash a season file. A file already on disk keeps its original retrieval details."""
    manifest = load(paths)
    path = paths.season_raw(season)
    digest = sha256_file(path)
    previous = manifest.raw.get(season)
    if not downloaded and previous is not None and previous.sha256 == digest:
        return previous
    record = RawRecord(
        season=season,
        path=_relative(paths, path),
        url=url,
        sha256=digest,
        bytes=path.stat().st_size,
        retrieved_at=_now() if downloaded else None,
        etag=etag,
        last_modified=last_modified,
    )
    manifest.raw[season] = record
    save(paths, manifest)
    return record


def _partition_record(paths: Paths, season: int, week: int, *, rows: int) -> PartitionRecord:
    path = paths.week_processed(season, week)
    return PartitionRecord(
        season=season,
        week=week,
        path=_relative(paths, path),
        sha256=sha256_file(path),
        rows=rows,
        raw_sha256=sha256_file(paths.season_raw(season)),
        transform_sha256=transform_sha256(),
        written_at=_now(),
    )


def record_week_partition(paths: Paths, season: int, week: int, *, rows: int) -> PartitionRecord:
    """Record one weekly partition the ETL just wrote."""
    manifest = load(paths)
    record = _partition_record(paths, season, week, rows=rows)
    manifest.processed[week_key(season, week)] = record
    save(paths, manifest)
    return record


def record_season_partitions(paths: Paths, season: int, rows: dict[int, int]) -> None:
    """Record every weekly partition of a season, replacing the season's old records."""
    manifest = load(paths)
    manifest.processed = {
        key: record for key, record in manifest.processed.items() if record.season != season
    }
    for week, count in sorted(rows.items()):
        manifest.processed[week_key(season, week)] = _partition_record(
            paths, season, week, rows=count
        )
    save(paths, manifest)


def record_models(paths: Paths, models: Sequence[ModelRecord], *, project_sha256: str) -> Manifest:
    """Record the dbt code a warehouse build ran; partitions no longer on disk drop out."""
    manifest = load(paths)
    manifest.models = {model.name: model for model in models}
    manifest.project_sha256 = project_sha256
    manifest.players_sha256 = sha256_file(paths.players) if paths.players.exists() else None
    manifest.processed = {
        key: record
        for key, record in sorted(manifest.processed.items())
        if paths.week_processed(record.season, record.week).exists()
    }
    manifest.built_from = {key: record.sha256 for key, record in manifest.processed.items()}
    save(paths, manifest)
    return manifest


def verify(paths: Paths) -> list[str]:
    """Every way the files on disk disagree with the manifest; empty when they match."""
    manifest = load(paths)
    problems: list[str] = []
    if not manifest.processed:
        problems.append(f"no processed partitions recorded in {paths.manifest}")
    for season, raw in sorted(manifest.raw.items()):
        path = paths.season_raw(season)
        if not path.exists():
            problems.append(f"raw {season}: {raw.path} is missing")
        elif sha256_file(path) != raw.sha256:
            problems.append(f"raw {season}: {raw.path} changed since it was recorded")
    current_transform = transform_sha256()
    raw_hashes: dict[int, str | None] = {}
    for key, partition in sorted(manifest.processed.items()):
        path = paths.week_processed(partition.season, partition.week)
        if not path.exists():
            problems.append(f"processed {key}: {partition.path} is missing")
            continue
        if sha256_file(path) != partition.sha256:
            problems.append(f"processed {key}: {partition.path} changed since it was recorded")
        if partition.season not in raw_hashes:
            raw_path = paths.season_raw(partition.season)
            raw_hashes[partition.season] = sha256_file(raw_path) if raw_path.exists() else None
        current_raw = raw_hashes[partition.season]
        if current_raw is not None and current_raw != partition.raw_sha256:
            problems.append(f"processed {key}: built from a different raw file; rerun etl")
        if partition.transform_sha256 != current_transform:
            problems.append(f"processed {key}: transform code changed since; rerun etl")
    recorded = {
        paths.week_processed(record.season, record.week) for record in manifest.processed.values()
    }
    for stray in sorted(set(paths.processed.glob("season=*/*.parquet")) - recorded):
        # dbt reads every file under processed/, so an unrecorded one is unaccounted data.
        problems.append(f"processed: {_relative(paths, stray)} is not in the manifest; rerun etl")
    if manifest.players_sha256 and (
        not paths.players.exists() or sha256_file(paths.players) != manifest.players_sha256
    ):
        problems.append("players: file changed since the warehouse read it; rerun warehouse")
    elif manifest.models and not manifest.players_sha256 and paths.players.exists():
        problems.append("players: file arrived since the warehouse was built; rerun warehouse")
    if not manifest.models:
        problems.append("no dbt models recorded; rerun warehouse")
    project = modeling.project_dir()
    for name, model in sorted(manifest.models.items()):
        source = project / model.path
        if not source.exists():
            problems.append(f"model {name}: {model.path} is missing; rerun warehouse")
        elif modeling.sha256_file(source) != model.sha256:
            problems.append(f"model {name}: {model.path} changed since; rerun warehouse")
    if manifest.models and manifest.project_sha256 != modeling.project_sha256():
        problems.append("dbt project: macros, seeds, or config changed since; rerun warehouse")
    return problems


def _short(digest: str | None) -> str:
    return "-" if digest is None else f"`{digest[:12]}`"


def render(manifest: Manifest) -> str:
    """The Markdown that lands in ``docs/lineage.md``."""
    lines = [
        "# Data lineage",
        "",
        f"- data version: `{manifest.data_version}`",
        f"- seasons: {len(manifest.seasons)}"
        + (f" ({manifest.seasons[0]}-{manifest.seasons[-1]})" if manifest.seasons else "")
        + f", {len(manifest.processed)} weekly partitions",
        f"- processed rows: {sum(p.rows for p in manifest.processed.values()):,}",
        f"- recorded at commit: {_short(manifest.git_commit)}",
        "",
        "Generated by `fourthdown lineage`. Hashes are SHA-256, shown to 12 characters;",
        "the full values are in `data/manifest.json`. `fourthdown lineage --verify` rehashes",
        "every file and fails if any of them changed.",
        "",
        "## dbt models",
        "",
        "Built by `dbt build` from `dbt_project/`; `fourthdown dbt docs generate` draws the",
        f"same graph. Macros, seed, and config hash: {_short(manifest.project_sha256)}.",
        "",
        "| model | schema | materialized | reads | file hash |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {model.name} | {model.schema} | {model.materialized} | {', '.join(model.reads)} "
        f"| {_short(model.sha256)} |"
        for model in manifest.models.values()
    ]
    lines += [
        "",
        "## Processed partitions (`plays`)",
        "",
        "One Parquet file per week. Per season here; every week's own hash is in the manifest.",
        "",
        "| season | weeks | rows | hash of the week hashes | from raw hash | transform hash |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for season in manifest.seasons:
        weeks = [p for _, p in sorted(manifest.processed.items()) if p.season == season]
        combined = hashlib.sha256("".join(p.sha256 for p in weeks).encode()).hexdigest()
        raw_hashes = {p.raw_sha256 for p in weeks}
        transforms = {p.transform_sha256 for p in weeks}
        lines.append(
            f"| {season} | {len(weeks)} ({weeks[0].week}-{weeks[-1].week}) "
            f"| {sum(p.rows for p in weeks):,} | {_short(combined)} "
            f"| {_short(raw_hashes.pop()) if len(raw_hashes) == 1 else 'mixed'} "
            f"| {_short(transforms.pop()) if len(transforms) == 1 else 'mixed'} |"
        )
    lines += [
        "",
        "## Raw sources",
        "",
        "| season | source | bytes | hash | retrieved | ETag |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.season} | [{Path(r.url).name}]({r.url}) | {r.bytes:,} | {_short(r.sha256)} "
        f"| {r.retrieved_at or 'before tracking'} | {r.etag or '-'} |"
        for r in manifest.raw.values()
    ]
    return "\n".join(lines) + "\n"
