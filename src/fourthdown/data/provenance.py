"""Content hashes and lineage for every table the pipeline produces.

Each step records what it wrote into one manifest at ``Paths.manifest``:

* ``raw``: the nflverse file for a season, its source URL, SHA-256, size, and the
  ETag and Last-Modified headers when it was downloaded. nflverse republishes the
  current season's file in place, so the hash is the only way to tell two downloads of
  "2024" apart.
* ``processed``: each season partition, its row count, its SHA-256, the hash of the raw
  file it was derived from, and the hash of the transform code that derived it.
* ``models``: each dbt model and seed (``dbt_project/``) the warehouse was built from: its
  file's hash, schema, materialization, and what it reads. ``project_sha256`` covers the
  rest of the project that shapes the output (macros, the seed, ``dbt_project.yml``).
* ``players_sha256``: the nflverse player table the warehouse read, if any.

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
from fourthdown.config import Paths
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
    path: str
    sha256: str
    rows: int
    raw_sha256: str
    transform_sha256: str
    written_at: str


@dataclass
class Manifest:
    raw: dict[int, RawRecord] = field(default_factory=dict)
    processed: dict[int, PartitionRecord] = field(default_factory=dict)
    models: dict[str, ModelRecord] = field(default_factory=dict)
    project_sha256: str | None = None
    players_sha256: str | None = None
    package_version: str = __version__
    git_commit: str | None = None
    updated_at: str | None = None

    @property
    def data_version(self) -> str:
        """One id for the processed bytes plus the dbt code that shapes them into tables."""
        digest = hashlib.sha256()
        for season in sorted(self.processed):
            digest.update(f"{season}:{self.processed[season].sha256}\n".encode())
        for name in sorted(self.models):
            digest.update(f"{name}:{self.models[name].sha256}\n".encode())
        if self.project_sha256:
            digest.update(f"project:{self.project_sha256}\n".encode())
        return digest.hexdigest()[:DATA_VERSION_CHARS]

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(sorted(self.processed))

    def to_json(self) -> str:
        payload = {
            "data_version": self.data_version,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "updated_at": self.updated_at,
            "raw": {str(season): asdict(record) for season, record in sorted(self.raw.items())},
            "processed": {
                str(season): asdict(record) for season, record in sorted(self.processed.items())
            },
            "models": {name: asdict(record) for name, record in self.models.items()},
            "project_sha256": self.project_sha256,
            "players_sha256": self.players_sha256,
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
        return cls(
            raw={int(key): RawRecord(**record) for key, record in payload.get("raw", {}).items()},
            processed={
                int(key): PartitionRecord(**record)
                for key, record in payload.get("processed", {}).items()
            },
            models=models,
            project_sha256=payload.get("project_sha256"),
            players_sha256=payload.get("players_sha256"),
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


def record_partition(paths: Paths, season: int, *, rows: int) -> PartitionRecord:
    manifest = load(paths)
    path = paths.season_processed(season)
    record = PartitionRecord(
        season=season,
        path=_relative(paths, path),
        sha256=sha256_file(path),
        rows=rows,
        raw_sha256=sha256_file(paths.season_raw(season)),
        transform_sha256=transform_sha256(),
        written_at=_now(),
    )
    manifest.processed[season] = record
    save(paths, manifest)
    return record


def record_models(paths: Paths, models: Sequence[ModelRecord], *, project_sha256: str) -> Manifest:
    """Record the dbt code a warehouse build ran; partitions no longer on disk drop out."""
    manifest = load(paths)
    manifest.models = {model.name: model for model in models}
    manifest.project_sha256 = project_sha256
    manifest.players_sha256 = sha256_file(paths.players) if paths.players.exists() else None
    on_disk = {season for season in manifest.processed if paths.season_processed(season).exists()}
    manifest.processed = {season: manifest.processed[season] for season in sorted(on_disk)}
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
    for season, partition in sorted(manifest.processed.items()):
        path = paths.season_processed(season)
        raw_path = paths.season_raw(season)
        if not path.exists():
            problems.append(f"processed {season}: {partition.path} is missing")
            continue
        if sha256_file(path) != partition.sha256:
            problems.append(f"processed {season}: {partition.path} changed since it was recorded")
        if raw_path.exists() and sha256_file(raw_path) != partition.raw_sha256:
            problems.append(f"processed {season}: built from a different raw file; rerun etl")
        if partition.transform_sha256 != current_transform:
            problems.append(f"processed {season}: transform code changed since; rerun etl")
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
        f"- seasons: {len(manifest.processed)}"
        + (f" ({manifest.seasons[0]}-{manifest.seasons[-1]})" if manifest.seasons else ""),
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
        "| season | rows | partition hash | from raw hash | transform hash |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {p.season} | {p.rows:,} | {_short(p.sha256)} | {_short(p.raw_sha256)} "
        f"| {_short(p.transform_sha256)} |"
        for p in manifest.processed.values()
    ]
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
