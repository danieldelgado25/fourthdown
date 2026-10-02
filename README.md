# fourthdown

A retrieval-augmented analytics assistant over 768,085 NFL plays (2009-2024, nflverse).

Ask it a number and it writes SQL. Ask it what happened and it retrieves narrative. Ask it
whether to go for it on fourth down and it calls a win-probability model. The LLM routes
and explains; it never does the arithmetic.

**Status: phase 05 complete** — ingestion, ETL, the DuckDB semantic layer, guarded
text-to-SQL, hybrid (dense + lexical) retrieval over generated narratives, the three
models (win probability, the fourth-down advisor, run/pass play call), and the serving
layer: a tool-routing orchestrator behind a Flask API and a React dashboard. Remaining
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
fourthdown eval                              # score the dev set -> docs/text_to_sql_eval.md
fourthdown eval --holdout -o docs/text_to_sql_holdout_eval.md   # score the held-out set
```

Every generated query is parsed with sqlglot before it reaches DuckDB: one read-only
SELECT, only the five semantic views, no filesystem functions, and a `LIMIT` is imposed.
Rejections and DuckDB errors are fed back to the model for a bounded number of repairs.
Accuracy is measured by executing the golden questions and comparing results with
handwritten reference SQL, not by matching query text. The prompt was tuned on a
15-question development set and checked on 15 held-out questions it never saw:
`qwen2.5-coder:7b` went from 7/15 to 14/15 correct on development and from 3/15 to 9/15
on held-out. Per-question results are in [docs/text_to_sql_eval.md](docs/text_to_sql_eval.md)
and [docs/text_to_sql_holdout_eval.md](docs/text_to_sql_holdout_eval.md).

## Narrative retrieval

Statistics come from SQL; "what happened" comes from retrieval. The warehouse is
rendered into deterministic English — one document per game, one per drive — which is
embedded with Ollama and stored in Postgres with pgvector:

```bash
docker compose up -d                         # pgvector/pgvector:pg16 on :5432
ollama pull nomic-embed-text
fourthdown index --seasons 2009- --playoff-drives    # ~8.6k documents
fourthdown recall "the comeback from 28-3"           # passages + provenance
fourthdown explain "how did the Seahawks lose Super Bowl XLIX?"
fourthdown eval-retrieval                    # -> docs/retrieval_eval.md
```

Every search runs two retrievers and fuses them by reciprocal rank: cosine similarity
over the embeddings (HNSW) for paraphrase, and Postgres full-text search (GIN over a
generated `tsvector`) for the names and numbers a dense model blurs. `--grain`,
`--season`, and `--team` filter before ranking, and each hit reports whether dense,
lexical, or both found it. `fourthdown explain` then answers from the retrieved passages
only, with `[n]` citations.

Over 8,501 indexed documents (4,345 games, 4,156 playoff drives) and 16 golden questions,
hybrid retrieval gets 0.94 hit@1 against 0.88 for dense alone and 0.81 for lexical alone
([docs/retrieval_eval.md](docs/retrieval_eval.md)).

Indexing is incremental and keyed on document ID, so re-running after a new week lands
rewrites only what changed. Drives outnumber games 23 to 1 and embedding is the slow
step, so `--playoff-drives` keeps the drive grain to the postseason; drop it to index all
99k. The index refuses to mix embedding spaces — a different model or dimensionality
requires `--reset`.

## Models

```bash
fourthdown train                             # fit all three -> docs/model_eval.md, data/models/
fourthdown advise --yardline 38 --togo 2 --minutes 4 --score-diff -3
```

Three models, all trained on 2009-2018, early-stopped on 2019-2020, and reported on
held-out 2021-2024 — season-disjoint, because splitting plays at random leaks the
outcome of a game into its own training set (every play of a win shares one label).

**Win probability** is a small PyTorch MLP on eleven pre-snap fields (field position,
down and distance, clock, score, timeouts, spread, home). It reaches 0.4553 log loss and
0.0066 expected calibration error on 166k held-out plays, against 0.4548 for nflfastR's
own `vegas_wp` — i.e. within noise of the reference implementation, from scratch.
nflfastR's fitted columns (`wp`, `vegas_wp`, `epa`, `xpass`, `cp`, ...) are enumerated in
`schema.LEAKAGE_COLUMNS` and `features.feature_matrix` raises if one reaches the design
matrix, so they can only ever be baselines.

**The fourth-down advisor** has no parameters of its own. It plays each option forward —
go, field goal, punt — into the game state it produces, asks the win-probability model
what that state is worth, and weights the branches by a conversion model (distance and
field position), a field-goal model (kick distance), and the empirical punt landing spot.
Because every option is a win-probability delta, the recommendation explains itself.
Replaying 4,000 held-out fourth downs: coaches went for it on 20.8% of them, the advisor
would on 42.3%, and the average actual decision left 0.40 win-probability points on the
field.

**Play call** is a gradient-boosted run/pass classifier: 74.0% accuracy and 0.815 AUC
against a 61.7% base rate, and ahead of nflfastR's `xpass` (71.1%, 0.790) on the same
plays. The analytics payoff is the per-team readability index it produces — how often the
model's call was the call, on early-down neutral-script plays only, so game script is not
mistaken for tendency. Full numbers in [docs/model_eval.md](docs/model_eval.md).

## Provenance, experiment tracking, and the model container

```bash
fourthdown lineage --verify      # rehash data/ against data/manifest.json -> docs/lineage.md
fourthdown train                 # refuses drifted data; logs the run to MLflow
make mlflow-ui                   # browse runs at http://127.0.0.1:5000
make wp-image && make wp-run     # the win-probability model on http://127.0.0.1:8080
```

Every build step hashes what it writes. The raw nflverse files, the processed partitions,
and the SQL of each view all go into `data/manifest.json`, together with the source URL
and the raw file each partition came from. They fold into one `data_version`. Every
training run logs that version, the season split, every hyperparameter, the metrics, and
the loss curve to MLflow, and writes a model card next to the weights. The container
serves those weights and refuses to start if their hash doesn't match the card. Every
prediction reports the data version and MLflow run that produced the model:

```bash
curl -s localhost:8080/v1/win-probability -H 'content-type: application/json' -d '{
  "yardline_100": 45, "down": 2, "ydstogo": 7,
  "game_seconds_remaining": 900, "score_differential": 3,
  "posteam_spread": -2.5, "posteam_is_home": true
}'
# {"win_probability": 0.7..., "model": {"data_version": "1e7d18214c2c",
#   "mlflow_run_id": "...", "artifact_sha256": "...", "git_commit": "..."}}
```

`{"states": [...]}` scores up to 1,000 situations at once. `GET /v1/model` returns the
card (features, split, held-out metrics), and `GET /health` is the container healthcheck.
Lineage tables are in [docs/lineage.md](docs/lineage.md).

## The assistant, the API, and the dashboard

One question, four tools. The router picks one and says why; the tool does the work and
returns its own evidence — a table and the SQL behind it, retrieved passages with
citations, or the win-probability value of every fourth-down option.

```bash
fourthdown chat "should they go for it on 4th and 2 from the 38, down 3 with 4 minutes left?"
make serve                                   # Flask on :8000
make web-install && make web                 # Vite dev server on :5173, proxying /api
make eval-routing                            # -> docs/routing_eval.md
```

Routing is LLM-first with a deterministic keyword router underneath: the model is asked
for exactly one tool name, and an unparseable, unknown, or unreachable answer falls back
to keywords rather than failing the request. On a 20-question golden set both routers
score 19/20 ([docs/routing_eval.md](docs/routing_eval.md)), which is the argument for
keeping the cheap one as the fallback.

Every dependency is optional at startup. Missing Postgres drops the narrative tool,
missing model artifacts drop the advisor, missing Ollama drops LLM routing — the API
reports what is loaded on `/api/health` and the dashboard greys out what it cannot do,
rather than erroring on first use.

| endpoint | purpose |
| --- | --- |
| `GET /api/health` | which of warehouse / llm / retrieval / models came up, and why not |
| `POST /api/ask` | route a question; returns the answer plus table, passages, SQL, route |
| `POST /api/advise` | fourth-down state in, ranked go / field goal / punt out |
| `GET /api/tendencies` | neutral-script pass rate and EPA by team for a season |

The services (DuckDB connection, pgvector pool, torch checkpoint) are opened once and
held read-only for the life of the process, which makes the app single-worker by design.

## Evaluation

```bash
make scorecard                               # every suite that can run here -> docs/scorecard.md
make scorecard-ci                            # what CI runs: offline suites, all required
```

Every harness reports to one scorecard, which is checked against the bounds in
[gates.json](src/fourthdown/evaluation/gates.json). Each bound records why it is there,
and some are ceilings: a win-probability AUC above 0.95 on held-out seasons points to a
leaked feature, not a better model. On every pull request, CI rebuilds the 2009-2024
warehouse and runs the SQL guard against 36 adversarial queries. It also scores keyword
routing, runs every golden reference query, and retrains all three models from scratch.
The scorecard is published to the job summary. Text-to-SQL and LLM routing run in
[eval-llm.yml](.github/workflows/eval-llm.yml) weekly, on demand, or on PRs labelled
`eval-llm`. Retrieval is scored locally against the full index. Design notes are in
[docs/architecture.md](docs/architecture.md#evaluation-in-ci-phase-06).

## Layout

```
src/fourthdown/
  cli.py              typer entrypoint: ingest / etl / warehouse / audit / build
  config.py           path resolution (override the data root with FOURTHDOWN_DATA_DIR)
  data/
    ingest.py         nflverse release downloader, incremental and resumable
    provenance.py     per-file SHA-256, table lineage, data_version, drift checks
    schema.py         the processed column contract, including the leakage list
    etl.py            lazy Polars pipeline -> season-partitioned Parquet
    warehouse.py      DuckDB views: plays, games, drives, team_game, player_game
    audit.py          validation checks + the generated audit report
  sql/guard.py        sqlglot validation: read-only, view whitelist, enforced LIMIT
  narrative/
    render.py         DuckDB rows -> deterministic game and drive documents
    teams.py          abbreviations -> names, so "Chiefs" matches `KC`
  retrieval/
    embed.py          Ollama embeddings behind a protocol, plus a hashing stub for tests
    store.py          pgvector schema, dense + lexical search, reciprocal-rank fusion
    index.py          warehouse -> narratives -> vectors -> Postgres, incremental
    recall.py         grounded answers with citations over retrieved passages
  rag/
    schema_card.py    the prompt's view/column/semantics card, read from the live catalog
    text_to_sql.py    generate -> validate -> execute, with repair-on-error
  llm/client.py       Ollama behind a one-method protocol
  models/
    features.py       leakage-safe design matrices and season-disjoint splits
    winprob.py        the PyTorch win-probability net, game state, training loop
    fourth_down.py    go / field goal / punt, valued through win probability
    playcall.py       run-pass GBM and the team predictability index
    metrics.py        log loss, Brier, AUC, calibration error, reliability bins
    train.py          fits all three, writes artifacts and docs/model_eval.md
    card.py           model card: artifact hash, data version, split, metrics, run id
    tracking.py       MLflow run logging (SQLite store under data/mlflow by default)
  serving/winprob.py  standalone win-probability API, the container's entrypoint
  evaluation/
    golden.json       15 questions with reference SQL, including traps and one refusal
    harness.py        result-level scoring and the Markdown report
    golden_retrieval.json  16 narrative questions with the games that answer them
    retrieval_harness.py   hit@1 / recall@k / MRR for hybrid vs dense vs lexical
    golden_routing.json    20 questions labelled with the tool that should answer them
    routing_harness.py     keyword vs LLM routing accuracy and misroutes
    golden_guard.json      adversarial, legitimate, and row-cap cases for the SQL guard
    suites.py              each harness as a scorecard suite, skipped when its service is absent
    scorecard.py           gates, pass/fail verdicts, Markdown and JSON output
    gates.json             the bound on every gated metric and the reason for it
  agent/
    tools.py          stats / narrative / advisor / tendency behind one Tool protocol
    parse.py          question -> season, team, and fourth-down game state
    router.py         LLM tool choice with a keyword router as the fallback
    assistant.py      route, run, and return structured evidence
    build.py          open what is available, report what is not
  api/app.py          Flask factory over one long-lived Services
web/
  src/api.ts          the typed wire format
  src/App.tsx         readiness bar and the three panels
  src/components/     ask, fourth-down advisor, team tendencies
docs/
  architecture.md     system design and phase status
  data_dictionary.md  grain, derived columns, leakage, data quirks
  data_audit.md       generated: per-season coverage and check results
  text_to_sql_eval.md generated: golden-set score and per-question failures
  retrieval_eval.md   generated: retrieval metrics per mode
  model_eval.md       generated: held-out model metrics, calibration, predictability
  routing_eval.md     generated: routing accuracy, keyword vs LLM
  scorecard.md        generated: every suite against its gates
  lineage.md          generated: data version, per-table sources and hashes
docker/
  winprob.Dockerfile  CPU-only image with the trained model and its card baked in
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
make web-test                # tsc, vite build, vitest
```

Tests marked `postgres` need `docker compose up -d` and skip without it; they run against
a separate `fourthdown_test` database so the development index survives.

Tests marked `slow` assert against the built warehouse (season play counts, the
neutral-vs-overall pass-rate gap, the rise in passing across the period) and skip when it
is absent.
