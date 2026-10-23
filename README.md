# fourthdown

A retrieval-augmented analytics assistant over 768,085 NFL plays (2009-2024, nflverse).

Ask it a number and it writes SQL. Ask it what happened and it retrieves narrative. Ask it
whether to go for it on fourth down and it calls a win-probability model. The LLM routes
and explains; it never does the arithmetic.

**Status: phase 07 complete** — ingestion, ETL, the DuckDB semantic layer, guarded
text-to-SQL, hybrid (dense + lexical) retrieval over generated narratives, the three
models (win probability, the fourth-down advisor, run/pass play call), a tool-routing
gateway with a React dashboard over four independently deployable services, gated
evaluation in CI, and Docker Compose and Helm deployments. Design notes and phase history
are in [docs/architecture.md](docs/architecture.md).

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
handwritten reference SQL, not by matching query text. `qwen2.5-coder:7b` currently
produces a runnable query for 14 of the 15 golden questions and the right answer for 7;
the failures are logged per question in
[docs/text_to_sql_eval.md](docs/text_to_sql_eval.md).

## Narrative retrieval

Statistics come from SQL; "what happened" comes from retrieval. The warehouse is
rendered into deterministic English — one document per game, one per drive — which is
embedded with Ollama and stored in Postgres with pgvector:

```bash
cp .env.example .env                         # pick a password; export FOURTHDOWN_PG_DSN to match
docker compose up -d                         # pgvector/pgvector:pg16 on 127.0.0.1:5432
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

## The assistant, the API, and the dashboard

One question, four tools. The router picks one and says why; the tool does the work and
returns its own evidence — a table and the SQL behind it, retrieved passages with
citations, or the win-probability value of every fourth-down option.

```bash
fourthdown chat "should they go for it on 4th and 2 from the 38, down 3 with 4 minutes left?"
make up                                      # every service in Docker; dashboard on :8000
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
rather than erroring on first use. Services retry what is missing every 30 seconds, so
pulling a model or finishing `fourthdown train` heals a running deployment.

| endpoint | purpose |
| --- | --- |
| `GET /api/health` | which of warehouse / llm / retrieval / models came up, and why not |
| `POST /api/ask` | route a question; returns the answer plus table, passages, SQL, route |
| `POST /api/advise` | fourth-down state in, ranked go / field goal / punt out |
| `GET /api/tendencies` | neutral-script pass rate and EPA by team for a season |

### Services

The app runs as four independently deployable services. Each owns one backing store, ships
as its own image with only its own dependencies, and can fail, restart, or scale without
the others:

| service | port | owns | image carries |
| --- | --- | --- | --- |
| gateway | 8000 | routing, the public `/api/*`, the dashboard | Flask, requests |
| warehouse | 8001 | `stats` (text-to-SQL), `tendency` | + DuckDB, sqlglot |
| retrieval | 8002 | `narrative` | + psycopg |
| models | 8003 | `advisor` | + torch, scikit-learn |

Tool services share one contract: `GET /health` lists the tools they have loaded (with
the descriptions the router reads) and `POST /tools/<name>` runs one. The gateway
discovers tools by polling `/health` and imports none of the services' code, so adding a
tool means deploying a service, not editing the gateway. A test fails the build if any
service starts importing another's dependencies.

`fourthdown chat` builds the same tools in-process, so the CLI needs no running services.

### Running it in Docker

Data is mounted, not baked in, so build and train on the host first:

```bash
cp .env.example .env                         # set POSTGRES_PASSWORD
make build train index                       # data/ on the host
make up                                      # postgres, ollama, and the four services
```

A one-shot `ollama-pull` container fetches the chat and embedding models into a volume;
the services pick them up when the download finishes. Without Docker, run each service
with `make serve-warehouse`, `serve-retrieval`, `serve-models`, and `serve-gateway`.

### Running it on Kubernetes

`deploy/helm/fourthdown` deploys the four services, with in-cluster Postgres and Ollama
(either can point at an external one instead: `postgres.enabled=false` with
`postgres.externalDsn`, `ollama.enabled=false` with `ollama.externalHost`). Readiness
probes use each service's `/health`, so a pod is only sent traffic once it has tools
loaded. Locally, on a single-node [kind](https://kind.sigs.k8s.io) cluster with the repo's
`data/` mounted on the node:

```bash
make kind-up                                 # cluster, images, helm install
make kind-forward                            # dashboard on http://localhost:8000
make kind-down
```

The warehouse stores its Parquet glob relative to the data root, so the same
`fourthdown.duckdb` opens on the host and at `/data` in a container. Warehouses built
before that change need one `fourthdown warehouse` to pick it up.

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
  evaluation/
    golden.json       15 questions with reference SQL, including traps and one refusal
    harness.py        result-level scoring and the Markdown report
    golden_retrieval.json  16 narrative questions with the games that answer them
    retrieval_harness.py   hit@1 / recall@k / MRR for hybrid vs dense vs lexical
    golden_routing.json    20 questions labelled with the tool that should answer them
    routing_harness.py     keyword vs LLM routing accuracy and misroutes
  agent/
    tools.py          the Tool protocol and the Evidence envelope, and their wire format
    parse.py          question -> season, team, and fourth-down game state
    router.py         LLM tool choice with a keyword router as the fallback
    assistant.py      route, run, and return structured evidence
  services/
    common.py         the tool-service contract: /health, /tools/<name>, self-healing
    gateway.py        tool discovery, routing, the public API, and the dashboard
    warehouse.py      stats and tendency tools over DuckDB
    retrieval.py      the narrative tool over pgvector
    models.py         the advisor tool and /advise over the trained models
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
make lint typecheck test     # ruff, mypy, pytest (keyword routing is gated in CI too)
make web-test                # tsc, vite build, vitest
```

Tests marked `postgres` need `docker compose up -d` and skip without it; they run against
a separate `fourthdown_test` database so the development index survives.

Tests marked `slow` assert against the built warehouse (season play counts, the
neutral-vs-overall pass-rate gap, the rise in passing across the period) and skip when it
is absent.

### Evaluation in CI

Every eval command takes a floor (`--min-accuracy`, `--min-correct`, `--min-hit-at-1`,
`--max-wp-log-loss`, `--min-play-call-accuracy`) and exits non-zero below it, after
writing its report. Keyword routing needs neither data nor an LLM, so it gates every push.
The rest -- text-to-SQL and LLM routing against Ollama, retrieval over a fresh index, and
a full retrain -- runs weekly and on demand in `.github/workflows/eval.yml`, with floors
at the published scores and the reports uploaded as an artifact.
