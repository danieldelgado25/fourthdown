# fourthdown

A retrieval-augmented analytics assistant over 768,085 NFL plays (2009-2024, nflverse).

Ask it a number and it writes SQL. Ask it what happened and it retrieves narrative. Ask it
whether to go for it on fourth down and it calls a win-probability model. The LLM routes
and explains; it never does the arithmetic.

**Status: phase 02 complete** — ingestion, ETL, the DuckDB semantic layer, and guarded
text-to-SQL against a local Ollama model. The narrative retrieval, modelling, and serving
phases are listed in [docs/architecture.md](docs/architecture.md).

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

## Asking questions

Text-to-SQL runs against a local [Ollama](https://ollama.com) model
(`FOURTHDOWN_LLM_MODEL`, default `qwen2.5-coder:7b`):

```bash
ollama pull qwen2.5-coder:7b
fourthdown schema-card                       # the prompt the model sees
fourthdown ask "Which team passed most on neutral downs in 2024?"
fourthdown eval                              # score the golden set -> docs/text_to_sql_eval.md
```

Every generated query is parsed with sqlglot before it reaches DuckDB: one read-only
SELECT, only the five semantic views, no filesystem functions, and a `LIMIT` is imposed.
Rejections and DuckDB errors are fed back to the model for a bounded number of repairs.
Accuracy is measured by executing the golden questions and comparing results with
handwritten reference SQL, not by matching query text.

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
  sql/guard.py        sqlglot validation: read-only, view whitelist, enforced LIMIT
  rag/
    schema_card.py    the prompt's view/column/semantics card, read from the live catalog
    text_to_sql.py    generate -> validate -> execute, with repair-on-error
  llm/client.py       Ollama behind a one-method protocol
  evaluation/
    golden.json       15 questions with reference SQL, including traps and one refusal
    harness.py        result-level scoring and the Markdown report
docs/
  architecture.md     system design and phase status
  data_dictionary.md  grain, derived columns, leakage, data quirks
  data_audit.md       generated: per-season coverage and check results
  text_to_sql_eval.md generated: golden-set score and per-question failures
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
