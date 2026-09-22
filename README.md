# fourthdown

A retrieval-augmented analytics assistant over 768,085 NFL plays (2009-2024, nflverse).

Ask it a number and it writes SQL. Ask it what happened and it retrieves narrative. Ask it
whether to go for it on fourth down and it calls a win-probability model. The LLM routes
and explains; it never does the arithmetic.

**Status: phase 00-01 complete** — ingestion, ETL, and the DuckDB semantic layer. The
retrieval, modelling, and serving phases are listed in
[docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
make install                 # venv + editable install with dev extras
make build                   # download 2009-2024, transform, build the warehouse (~10 min, ~1.5 GB)
make audit                   # regenerate docs/data_audit.md
```

Then query it directly:

```bash
duckdb data/warehouse/fourthdown.duckdb
```

```sql
-- Neutral-script pass rate by team, 2024: tendency with game script removed.
SELECT posteam, round(avg(is_pass_call::INT), 3) AS pass_rate, count(*) AS plays
FROM plays
WHERE season = 2024 AND is_designed_play AND is_neutral_script
GROUP BY posteam
ORDER BY pass_rate DESC
LIMIT 5;
```

A single season is enough to develop against: `make build SEASONS=2024`.

## Layout

```
src/fourthdown/
  cli.py              typer entrypoint: ingest / etl / warehouse / audit / build
  config.py           path resolution (override the data root with FOURTHDOWN_DATA_DIR)
  data/
    ingest.py         nflverse release downloader, incremental and resumable
    schema.py         the processed column contract, including the leakage list
    etl.py            lazy Polars pipeline -> season-partitioned Parquet
    warehouse.py      DuckDB views: plays, games, drives, team_game, player_game
    audit.py          validation checks + the generated audit report
docs/
  architecture.md     system design and phase status
  data_dictionary.md  grain, derived columns, leakage, data quirks
  data_audit.md       generated: per-season coverage and check results
```

## Data

nflverse publishes nflfastR play-by-play as one Parquet file per season, no
authentication, updated through the current season — which is why it is the primary
source rather than the static Kaggle 2009-2016 dump. Raw files are never modified; the ETL
projects ~90 of 372 columns, normalises types, and derives the situational features in
[docs/data_dictionary.md](docs/data_dictionary.md).

Every derived aggregate excludes kneels, spikes, aborted snaps, and special teams, so
`is_pass_call` is a real pre-snap decision and not an artifact of clock management. The
audit asserts that against published league rates on every build.

Data is gitignored. `make build` reproduces it.

## Development

```bash
make lint typecheck test     # ruff, mypy, pytest
```

Tests marked `slow` assert against the built warehouse (season play counts, the
neutral-vs-overall pass-rate gap, the rise in passing across the period) and skip when it
is absent.
