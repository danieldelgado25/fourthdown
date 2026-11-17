# Architecture

## Target system

```
React dashboard  ──>  Flask API  ──>  agent orchestrator
                                          │
                     ┌────────────────────┼────────────────────┬──────────────┐
                     ▼                    ▼                    ▼              ▼
             text-to-SQL              semantic search      model tools       LLM
             (DuckDB)                 (pgvector)           (PyTorch,      (Ollama, or a
                                                            scikit-learn)  hosted API)
                     ▲                    ▲                    ▲
                     └──── offline: Polars ETL, narrative generation, training ────┘
```

The design constraint behind all of it: **the LLM never sees the play table and never does
arithmetic.** Embedding 768k rows and retrieving by cosine similarity cannot answer "how
many", "most", or "average", which is most of what anyone asks of a play-by-play dataset.
So numeric questions go to SQL, narrative questions go to retrieval over generated prose,
and predictive questions go to a trained model. The model's job is routing, query
authoring, and explanation.

## Query routing

| question type | example | path |
| --- | --- | --- |
| aggregate | "EPA per dropback on third and long in 2014" | text-to-SQL over the DuckDB views |
| narrative | "what happened on the final drive of Super Bowl XLIX" | hybrid BM25 + dense retrieval over drive summaries |
| predictive | "4th and 2 on their own 45, down 4, three minutes left — go for it?" | win-probability model, evaluated for each decision |

## Data layer (built)

```
data/raw/pbp/play_by_play_<season>.parquet      nflverse release, untouched
data/raw/players.parquet                         nflverse player ids -> full names
data/processed/plays/season=<season>/plays.parquet   projected, typed, feature-derived
data/warehouse/fourthdown.duckdb                 six views over the partitions
```

One lazy Polars plan per season does projection, type normalisation, and feature
derivation; partitions are written independently so a new season is an incremental
`fourthdown build --seasons 2025` rather than a full rebuild.

The six views (`plays`, `games`, `drives`, `team_game`, `player_game`, `players`) are the entire
surface the query layer sees. This is a deliberate text-to-SQL decision: a model asked to
write SQL against `team_game` rarely invents a join, while a model handed 372 raw columns
invents one constantly. Anything the views cannot express is a signal to add a view, not
to widen the prompt.

## Query layer (built)

```
question ──> schema card + question ──> LLM ──> SQL ──> sqlglot guard ──> DuckDB (read-only)
                    ▲                                        │
                    └──────── error + the columns that exist ┘   (bounded repairs)
```

Three properties matter more than the prompt:

**The card is generated, not written.** Column names and types come from
`information_schema` at call time, so a view change cannot silently desynchronise from the
prompt. `plays` is trimmed to a curated subset rather than dumped whole; the card stays
under ~5k characters, which leaves room for the question and a repair round inside a small
local context window.

**Player names are resolved before the model sees the question.** The plays store
abbreviations like `J.Smith-Njigba`, and 197 of them are shared by several players.
`rag/names.py` matches the question against the full names from nflverse's player table
(hyphens, possessives, suffixes and one typo tolerated) and lists each match's
`player_id` with the prompt; a shared name lists every candidate with its teams and seasons.

**The guard is a parser, not a regex.** sqlglot parses the statement and rejects anything
that is not exactly one read-only query, references a relation outside the six views,
or calls a filesystem function (`read_parquet`, `glob`, ...). A `LIMIT` is inserted when
absent and tightened when it is too large. Execution then happens on a read-only
connection, so the guard is a second line of defence rather than the only one.

**Accuracy is measured on results.** The golden set pairs each question with handwritten
reference SQL and the harness compares executed rows, because there are many correct ways
to write the same query and none of them match a string. It includes the traps the
semantic layer exists to handle: historical team aliases (`SD` -> `LAC`), kneels that
have to be excluded, neutral-script versus raw tendency, and one question the data cannot
answer, which the model is expected to decline.

**Accuracy comes from semantics, not syntax.** The first 7B run produced runnable SQL for
14 of 15 golden questions but the right answer for only 7. The misses were domain and grain
mistakes: "third and long" filtered one distance bucket and dropped `down = 3`, a passer's
season EPA averaged per-game rates and put the play minimum in WHERE instead of HAVING,
the coldest game paired `min(temp)` with an unrelated `min(game_date)`, and a red-zone
question asked `drives` for a `field_zone` column that only `plays` has. The fixes target
those categories rather than the questions:

- the schema card states each aggregate view's grain, a football glossary (third and long,
  fourth-down attempts, designed runs, red zone, team codes), and rules for query shapes
  (ORDER BY ... LIMIT 1 for an extreme row, GROUP BY the flag for a comparison);
- six worked exemplars cover those shapes on different teams, seasons, and metrics, and a
  test fails if any exemplar duplicates a golden question or reference;
- repair prompts carry every failed attempt, the columns of each view the query touched,
  and, when DuckDB reports a missing column, which view owns it or that none does.

Diagnosing the failures also found a warehouse bug: `drives.scored_touchdown` was true
for pick-sixes and other return touchdowns, so the golden reference itself was counting
the defence's scores. It now follows the drive result.

**A season means the regular season.** Every aggregate view carries `season_type`, and
`games`, `team_game`, and `player_game` carry `playoff_round` (`wild_card` through
`super_bowl`, computed from the week because the last regular week moved from 17 to 18 in
2021). "Derrick Henry's rushing yards in 2020" is the box-score 2,027, not 2,067 with the
playoff game; "his yards in the 2020 playoffs" filters `'POST'`. The golden references
follow the same convention. Questions the live demo got wrong drove three more checks
that run only when a query returns nothing: a `player_name` filter that matches no one
(full names and nicknames, where nflverse stores `C.Ochocinco`) and a team code that is
not one of the 32 (`LAR` for `LA`) are fed back like an error, with the names or code
that do exist; a question naming only seasons outside the warehouse is refused before
the model is called. Player totals group by `player_id`, since names like `C.Johnson`
belong to several players.

**An empty result gets a second look.** The checks above catch two causes of an empty
result; anything else (a misread `playoff_round`, the wrong `role`, contradictory filters)
used to reach the user as a blank answer with no record. Now `fourthdown.sql.filters`
reruns the query as `count(*)` with every condition, then with each top-level `AND`
condition removed, and the model reviews those counts: either the answer is truly empty
(the reason is shown), or the query is wrong and it writes a correction. Design choices:

- The counts come first and need no model, because a model judging an empty result cold
  guesses. "Removing `playoff_round = 'afc_championship'` brings back rows" names the bug.
- The model only rewrites SQL. The correction passes the same guard and runs on DuckDB, so
  every number still comes from the warehouse, and the reply shows both queries.
- The prompt forbids dropping a filter the question states, and a correction that returns
  only NULLs or zeros does not replace the original. The golden set holds questions whose
  answer is nothing (`expect_empty`) to catch a reviewer that invents one.
- One review, no loop, and only on empty results, so an answered question costs nothing
  extra. When a correction also finds nothing, the reader gets a plain message; the
  model's diagnosis of that case proved unreliable and goes only to the log.
- Every empty result, reviewed or not, is a line in `data/feedback/misses.jsonl` with the
  question, both queries, the counts, and the verdict (`fourthdown misses`). The warehouse
  service mounts only that directory writable; a failed write never fails the answer.

**Every question is measured.** The gateway appends one line per `/api/ask` to
`data/feedback/queries.jsonl`: the tool, who chose it (`keyword`, `llm`, `caller`,
`fallback`), latency, whether SQL ran, the row count and whether the rows were empty, the
review verdict, any error, and whether a guest asked. It is the same best-effort JSONL
writer as the miss log, so the gateway gains a writable `feedback/` mount and still reads
no data. `fourthdown queries` reports nearest-rank p50/p95 latency, the error and empty
rates, and the route mix, with `--hours` for a recent window.

**A held-out set keeps the tuning honest.** Prompt changes made while looking at 15
questions can memorise them. `golden_holdout.json` holds 15 different questions, written
and scored before any tuning and never used to choose card text or exemplars. The
scorecard reports both, and the gates hold both.

| set | before | after |
| --- | --- | --- |
| development, correct | 7/15 | 14/15 |
| development, runnable | 14/15 | 15/15 |
| held-out, correct | 3/15 | 9/15 |
| held-out, runnable | 13/15 | 14/15 |

The held-out gain is smaller than the development gain, which is the expected shape: part
of the development improvement is fitting, and the held-out number is the one to quote.
Its remaining misses are listed in
[text_to_sql_holdout_eval.md](text_to_sql_holdout_eval.md).

**The examples teach more than the rules.** The development set grew to 55 questions so
every view is covered (team_game and players had none). Rerunning it found that the
held-out score had fallen to 1/15. The references had since adopted the regular-season
convention, and the schema card stated it, but five of the six season-scoped worked
examples left out `season_type = 'REG'`. The model copied the examples, not the rule, and
counted playoff games; four of the misses matched the reference exactly once the filter
was dropped. Adding the filter to those examples was the whole fix. A second fix moves
Super Bowl numerals out of the model's hands: `parse.super_bowls` reads "Super Bowl LX"
as season 2025 and the prompt states it, as it does for player ids, because the model
did the arithmetic wrong.

| run (qwen2.5-coder:7b) | development | held-out |
| --- | --- | --- |
| before (55-question set) | 28/54 | 1/15 |
| regular-season examples | 33/54 | 9/15 |
| + Super Bowl numerals | 35/54 | 9-11/15 |
| final scorecard, 55 questions | 36/55 | 9/15 |

Held-out prompts are identical with and without the numeral hint (no held-out question
names a Super Bowl), so its 9-to-11 is run-to-run spread from the repair attempts, which
sample at temperatures above zero. Treat two held-out questions as noise. These runs used
the same weights forced onto the CPU (`num_gpu 0`). Every number the model's answers
stated was in its result (`text_to_sql.grounded` = 1.0).

## Narrative retrieval (phase 03)

```
DuckDB views ──> generated narratives ──> Ollama embeddings ──> Postgres + pgvector
                                                                        │
            question ──> embedding ─┬─> dense (cosine, HNSW) ───┐       │
                                    └─> lexical (tsvector, GIN) ┴─> RRF ┴─> passages ──> LLM
```

**The corpus is generated, not scraped.** The warehouse stores `posteam = 'KC'` and
`fixed_drive_result = 'Touchdown'`; no embedding of that matches "what happened on the
Chiefs' last drive". So `narrative/render.py` writes deterministic English from the same
views the SQL layer queries -- one document per game, one per drive -- expanding team
codes to names and postseason weeks to "Super Bowl XLIX (49)". Deterministic matters:
re-indexing a season produces byte-identical text, so upserts are idempotent and the
embedding cost is paid once.

**Rows are not embedded.** 768k plays would be 768k vectors of mostly boilerplate, and
cosine similarity over them answers nothing a `GROUP BY` does not answer better. 4.3k
games plus 99k drives is the grain at which a question is actually asked.

**Both halves of retrieval, fused by rank.** Dense search generalises ("collapse" finds
a blown lead); lexical search does not miss a name or a number ("Super Bowl LVII",
"J.Burrow"). Their scores are incomparable -- a cosine distance and a `ts_rank_cd` share
no scale -- so they are combined by reciprocal rank fusion, `1/(60 + rank)` summed across
the lists a document appears in, which needs no tuning and no normalisation.
[docs/retrieval_eval.md](retrieval_eval.md) scores hybrid against each half alone on 16
golden questions: hybrid 0.94 hit@1 against 0.88 dense-only and 0.81 lexical-only, which
is the justification for the extra moving part.

One detail did all the work there. `websearch_to_tsquery` ANDs the query terms, so
"What happened in Super Bowl XLIX between the Patriots and the Seahawks?" required
*happened* and *between* to appear and matched nothing -- lexical scored 0.06 hit@1 and
hybrid was worse than dense alone. ORing the lexemes and letting `ts_rank_cd` reward
coverage instead took lexical to 0.81 and hybrid past both halves.

**Numbers still come from SQL.** The narratives contain numbers, but a retrieved passage
is evidence about *which* game, not the authority on a total. `fourthdown ask` stays the
path for aggregates; `fourthdown explain` answers from passages and cites them.

## Modelling (phase 04)

**Season-disjoint splits, not random ones.** Every play of a game shares one win/loss
label, so a random split puts the same outcome on both sides of the line and flatters the
model. Train is 2009-2018, validation 2019-2020 (early stopping only), test 2021-2024.
The test range is written open-ended and intersected with the seasons present, so a
freshly ingested season joins the held-out set rather than the training set.

**The leakage list is enforced in code.** nflfastR ships its own fitted outputs in the
same table: `wp`, `vegas_wp`, `epa`, `cp`, `cpoe`, `xpass`, `xyac_epa`. Using them as
features would be training on another model's answer key, so `schema.LEAKAGE_COLUMNS`
names them and `features.feature_matrix` raises if one appears in a design matrix. They
are kept for exactly one purpose: scoring them on the same held-out plays as a baseline.
That is what makes "0.4553 against vegas_wp's 0.4548" a claim rather than a coincidence.

**Calibration is the metric that matters.** A win-probability number is only useful if
60% means 60%, so the report carries expected calibration error and a ten-bucket
reliability table alongside log loss, Brier, and AUC. The fourth-down advisor consumes
probabilities, not rankings; an uncalibrated model with a good AUC would produce
confident nonsense.

**The advisor is arithmetic over the model, not a fourth model.** Each option is played
forward into the state it produces and evaluated with the win-probability net:

```
V(go)     = p_convert * WP(1st and 10, spot) + (1 - p_convert) * (1 - WP(their ball, spot))
V(kick)   = p_make * (1 - WP(their ball, 25, +3)) + (1 - p_make) * (1 - WP(their ball, spot))
V(punt)   = 1 - WP(their ball, empirical landing spot)
```

Possession flips mirror field position, negate the score differential and the spread, and
swap timeouts and home-field. Because the three numbers are on one scale, the
recommendation is a win-probability delta with its inputs shown, which is the form a
coach or an interviewer can argue with.

**Play call earns its keep through the tendency index.** Run/pass accuracy is table
stakes; the question worth asking is whether being readable costs anything. Readability
is measured per team-season on early-down neutral-script plays only, so a team that runs
out a 21-point lead is not scored as predictable, and it is correlated against the same
team-season's EPA per play.

## Serving (phase 05)

**Routing is a two-tier decision, not a prompt.** The LLM is asked for exactly one tool
name from the catalog of tools that actually loaded; a keyword router sits underneath it
and answers whenever the model is unreachable, returns prose, or names a tool that does
not exist. Both are measured against the same labelled question set
([routing_eval.md](routing_eval.md)), and they score the same 19/20 — which is the point:
the fallback is not a degraded mode, it is a cheap router the LLM has to beat.

**Tools return evidence, not prose.** Every tool produces the same envelope — an answer
line, an optional table, retrieved passages, and a detail map — so the API contract does
not change shape per tool and the dashboard renders provenance uniformly: the SQL that
produced a table, the passages behind a narrative claim, the win probability of each
fourth-down option. Nothing the LLM writes carries a number the tools did not compute.

**A situation is parsed, never guessed.** The advisor needs a game state, so the
question is parsed for distance, field position, clock, and margin; if distance or field
position is missing the tool says what it needs instead of inventing a plausible fourth
down. Numbers are the one thing the pipeline is not allowed to hallucinate.

**Every dependency is optional.** Services are opened once at startup, each failure is
recorded rather than raised, and the tool catalog is whatever came up. No Postgres means
no narrative tool; no artifacts means no advisor; no Ollama means keyword routing and no
SQL authoring. `/api/health` reports the state and the reason, and the dashboard disables
what it cannot reach — a fresh clone with nothing built still starts and explains itself.

Each service holds its own DuckDB connection, Postgres connection, or torch model for the
process lifetime and answers on `FOURTHDOWN_THREADS` waitress threads (default 4), so a
slow LLM-backed answer does not queue fast ones; capacity beyond that comes from replicas.

## Services (phase 07)

```
                     ┌─► warehouse :8001 ── DuckDB (read-only) ─────┐
browser ──► gateway ─┼─► retrieval :8002 ── Postgres + pgvector ────┼─► Ollama
            :8000  │ └─► models    :8003 ── torch artifacts         │
                   └────────────────────────────────────────────────┘ (routing)
```

**Split by backing store.** Each service owns exactly one store and the tools that read
it, so each has one reason to fail and one set of heavy dependencies. The gateway owns
nothing: it routes, forwards, and serves the dashboard, and its image has no DuckDB,
psycopg, or torch. `tests/test_service_isolation.py` imports each service in a fresh
interpreter and fails if it loads another service's dependencies.

**Discovery, not configuration.** Tool services publish their catalog on `/health` --
names and the descriptions the router reads -- and the gateway polls it (cached for 15
seconds). A service that is down contributes no tools, so the router never picks one that
cannot run; one that disappears mid-request yields a failed answer naming it, not a 500.

**Start in any order, heal in place.** A service that comes up without its store reports
what is missing and retries every 30 seconds. Postgres arriving late, a model finishing
its download, or `fourthdown train` writing artifacts all heal a running deployment
without a restart.

**One contract.** `/health` (200 with tools loaded, 503 without, so it is also the
readiness probe) and `POST /tools/<name>` returning the `Evidence` envelope. The
structured endpoints the dashboard uses, `/advise` and `/tendencies`, are proxied by the
gateway to whichever service owns `advisor` and `tendency`.

## Evaluation in CI (phase 06)

Every harness from phases 02-05 now feeds one scorecard (`fourthdown scorecard`) that
compares each metric with a bound in `src/fourthdown/evaluation/gates.json`. Each bound
records why it is there, so a failing build explains itself.

**Gates are bands, not just floors.** A held-out win-probability AUC above 0.95, or a
run/pass AUC above 0.92, is not a better model. It means a leaked feature, so the gate
fails it. The advisor's go-for-it rate is bounded on both sides as well, because an
advisor that always goes for it, or never does, can still agree with coaches often
enough to look fine.

**The suites split by what they need.**

| suite | needs | where it runs |
| --- | --- | --- |
| guard: 36 adversarial queries, legitimate queries, row caps | nothing | every PR |
| routing_keyword | nothing | every PR |
| references: every golden reference query runs on the warehouse | warehouse | every PR |
| grounding: every number in a stats answer is in its SQL result | warehouse | every PR |
| models: retrain all three models from scratch and score them | warehouse | every PR |
| text_to_sql, routing_llm | Ollama with `qwen2.5-coder:7b` | weekly, on demand, or with the `eval-llm` PR label |
| retrieval | the full Postgres index and `nomic-embed-text` | weekly and on demand, beside the LLM suites |

On every PR the models are retrained rather than loaded from checked-in artifacts. That
checks the pipeline still produces a good model, not that an old file is still on disk.
A suite that cannot run is skipped with a reason. CI passes `--require` for the suites it
expects, so a missing dependency fails the job instead of being reported as a pass.

Retrieval runs against the full index, every game and every postseason drive. Indexing
only the games the golden set asks about would make search much easier than against the
real corpus and report an inflated hit@1, so the job embeds all ~8.5k narratives first,
which is why it is weekly rather than per-PR.

**Numbers are checked, not trusted.** "Nothing the LLM writes carries a number the tools
did not compute" is a rule the grounding suite enforces. It takes the exact text a reader
gets from the stats tool, extracts every number, and requires each one to be a cell of
the result (at the precision printed, so `0.37` stands for `0.3679`), digits inside a
text cell (`07` in a game id supports "week 7"), the row count, or a number from the
question. Offline it replays every golden reference through the real formatting path, and
the gate is 100%: one invented number is a hallucination, not a rounding error. The
text_to_sql suite reports the same `grounded` metric over what the model wrote, including
the empty-result review's notes.

**The guard suite found real bypasses.** Before this phase the guard checked table names
against the allowed views, but it had no rule for table functions. A query could read a
file in one branch of a UNION as long as another branch read `games`. 11 of the 36
adversarial queries got through: `read_text`, `read_blob`, `sniff_csv`, `sqlite_scan`,
`query()`, `duckdb_tables()`, `range()`, `current_setting()`, and views qualified with
another catalog or schema. The guard now rejects every FROM source that is not a plain
view name in the warehouse schema.

## Provenance and the win-probability container

**Every table has a recorded origin.** Each pipeline step writes what it produced into
`data/manifest.json` (`src/fourthdown/data/provenance.py`):

| step | records |
| --- | --- |
| `ingest` | per season: source URL, SHA-256, bytes, and the ETag / Last-Modified of the download |
| `etl` | per partition: rows, SHA-256, the SHA-256 of the raw file it came from, and a hash of `etl.py` + `schema.py` |
| `warehouse` | per view: the SHA-256 of its SQL and what it reads (`plays` reads the partitions; the other four read `plays`) |

nflverse republishes the current season's file in place, so a season number does not
identify data; the hash does. `data_version` is a 12-character hash over the processed
partition hashes and the view SQL. `fourthdown lineage --verify` rehashes every file and
fails if a raw file changed, a partition was edited, a partition was built from a raw
file that has since been replaced, or the transform code changed after the ETL ran. The
human-readable version is [lineage.md](lineage.md).

**Training refuses drifted data and logs every run.** `fourthdown train` runs the same
verify first and exits if anything has drifted, so a model can't be trained on data
the manifest doesn't describe. Each run then goes to MLflow (`models/tracking.py`), by
default a SQLite store at `data/mlflow/mlflow.db`. The run records the data version as
a tag and a param, the season split, every hyperparameter, the headline metrics, the
per-epoch train and validation loss, and the artifacts, manifest, and report as
attachments. `FOURTHDOWN_MLFLOW_URI` points it at a shared server instead.

**The artifact carries its own provenance.** Training writes `data/models/model_card.json`
next to `winprob.pt`. It holds the artifact's SHA-256, the feature order, the data version,
the split, the metrics, the commit, and the MLflow run id. The serving container has no
MLflow and no warehouse, so the card is how it knows what it is serving.

**The container serves one model and proves which one.** `docker/winprob.Dockerfile`
bakes in the weights and the card and runs `fourthdown.serving.winprob` under gunicorn
with two preloaded workers. It installs only what inference imports (CPU torch, numpy,
polars, duckdb, scikit-learn, Flask), pinned, with no LLM, retrieval, or MLflow stack. At
startup it rehashes the weights and refuses to start if they don't match the card. Every
response then carries the data version, run id, and artifact hash, and the image labels
carry the same ids, so a probability can be traced from the response back to the run and
the source files.

**The demo VM turns itself off.** It bills by the hour while allocated, and shutting the
OS down does not deallocate it. The `idle-shutdown` service (`serving/idle.py`, in the
gateway image) checks once a minute: no question in the gateway's query log for an hour,
measured from the later of the last question and the service's own start, and no
`USER_PROCESS` entry in the host's `utmp` (mounted read-only, so an SSH session holds
the VM up). Then it gets a token for the VM's managed identity from the instance metadata
service and calls Resource Manager's `deallocate` on its own resource id, which it also
reads from the metadata service. No credential is stored on the machine, and the role
`deploy/azure/setup.sh` grants is scoped to that one VM. A failed call (no identity yet,
or off Azure) is logged and retried after another idle window, never every minute.

**Only gated models are served.** Training used to write straight into `data/models`, so
the last run served whatever it produced. Now each run trains into a candidate directory,
becomes a version of `fourthdown-win-probability` in the MLflow registry
(`models/registry.py`), and is tagged with the verdict of the same `models.*` gates the
scorecard applies; a missing gated metric counts as a failure. Promotion refuses a failed
version. A passing one gets the `champion` alias and its artifacts are exported from
MLflow into `data/models`: downloaded beside the directory, the weights rehashed against
the card, then swapped in file by file with the card last. The serving side is unchanged
and still has no MLflow; the card it loads now carries `registered_version`, and the model
container returns it with every prediction. Smoke runs (CI, the three-season cluster)
pass `--promote always`, which installs the model but never moves the alias.

## Containers and Kubernetes (phase 07b)

One Dockerfile with a target per service (`make images`), plus the model image. None of
the service images hold data, and each installs only its own pip extra, pinned by
`docker/constraints.txt`:

| image | contents | runs as |
| --- | --- | --- |
| `fourthdown-gateway` | routing, Flask, waitress, the built dashboard | the public API and the dashboard |
| `fourthdown-warehouse` | + DuckDB, sqlglot | stats, tendency |
| `fourthdown-retrieval` | + psycopg | narrative |
| `fourthdown-models` | + CPU torch, scikit-learn | advisor |
| `fourthdown-pipeline` | every extra, typer, MLflow | the pipeline Job, the index Job, `fourthdown wait` |
| `fourthdown-winprob` | one trained model and its card (above) | the standalone model service |

**Data is built where it is served.** `docker/pipeline.sh` runs `fourthdown build` and
`fourthdown train` against the volume mounted at `/data`, and skips both when the volume
already holds a build of the same configuration that passes `fourthdown lineage --verify`.
The provenance checks therefore run inside the cluster too. In Compose the volume is the
repo's `data/`, so a build on the host (`make build train`) serves just as well.

**Ordering is mostly unnecessary.** The services heal in place (above), so they start
alongside the pipeline Job and turn ready as the warehouse and models appear. Only the
index Job needs ordering: it runs `fourthdown wait` as an init container, polling for the
warehouse, Postgres, and the embedding model, and on timeout reports which check failed.

**Jobs are named by their inputs.** A Job's spec is immutable, so the pipeline, index,
and model-pull Jobs carry a hash of their inputs: seasons, training flags, and image for
the pipeline; seasons, embedding model, and pipeline for the index. Changing an input
creates a new Job, and an unchanged upgrade is a no-op. A service reopens only what is
missing, so the warehouse and models pods are annotated with the pipeline Job's name and
a rebuild rolls them.

**Sized for a laptop.** Requests total under one CPU, so the chart schedules on a 2-CPU
kind node beside the control plane, and limits leave room for the 7B model. The data
volume is ReadWriteOnce, which is enough when every pod runs on one node; a multi-node
cluster needs ReadWriteMany. While the warehouse service holds read-only DuckDB
connections, a pipeline Job cannot rewrite the warehouse, so a rebuild with different
seasons needs it scaled to zero first.

CI installs the chart on kind with three seasons, no Ollama, no retrieval service, and no
model image (`values-ci.yaml`). It then runs `helm test`, which checks health, fourth-down
advice, and keyword routing through the gateway, and reaches the dashboard through the
kind port mapping.


## Phase status

| phase | scope | status |
| --- | --- | --- |
| 00 | scope, repo, data audit | done |
| 01 | Polars ETL, Parquet, DuckDB semantic layer | done |
| 02 | schema card, text-to-SQL, sqlglot guardrails, golden set | done |
| 03 | narrative corpus, embeddings, hybrid retrieval | done |
| 04 | win probability, 4th-down advisor, play-call model | done |
| 05 | orchestrator, Flask API, React dashboard | done |
| 06 | evaluation harness in CI | done |
| 07a | provenance, MLflow tracking, containerized win-probability API | done |
| 07b | full stack in Docker Compose, Helm chart on a local kind cluster | done |
| 07c | gateway and tool services, provider-agnostic LLM, hybrid routing | done |
