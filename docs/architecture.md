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
data/processed/plays/season=<season>/plays.parquet   projected, typed, feature-derived
data/warehouse/fourthdown.duckdb                 five views over the partitions
```

One lazy Polars plan per season does projection, type normalisation, and feature
derivation; partitions are written independently so a new season is an incremental
`fourthdown build --seasons 2025` rather than a full rebuild.

The five views (`plays`, `games`, `drives`, `team_game`, `player_game`) are the entire
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

**The guard is a parser, not a regex.** sqlglot parses the statement and rejects anything
that is not exactly one read-only query, references a relation outside the five views,
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

The flip side of holding a DuckDB connection, a pgvector pool, and a torch checkpoint for
the process lifetime is that the app is single-worker by design. Multi-worker serving
(gunicorn with preload, or the model behind its own server) belongs with the container
work in phase 07.

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
| models: retrain all three models from scratch and score them | warehouse | every PR |
| text_to_sql, routing_llm | Ollama with `qwen2.5-coder:7b` | weekly, on demand, or with the `eval-llm` PR label |
| retrieval | Postgres index and `nomic-embed-text` | locally (`make scorecard`) |

On every PR the models are retrained rather than loaded from checked-in artifacts. That
checks the pipeline still produces a good model, not that an old file is still on disk.
A suite that cannot run is skipped with a reason. CI passes `--require` for the suites it
expects, so a missing dependency fails the job instead of being reported as a pass.

Retrieval is kept out of CI on purpose. Indexing only the games the golden set asks
about would make search much easier than against the full index. CI would then report
inflated hit@1 for a smaller corpus, and that number would be misleading.

**The guard suite found real bypasses.** Before this phase the guard checked table names
against the allowed views, but it had no rule for table functions. A query could read a
file in one branch of a UNION as long as another branch read `games`. 11 of the 36
adversarial queries got through: `read_text`, `read_blob`, `sniff_csv`, `sqlite_scan`,
`query()`, `duckdb_tables()`, `range()`, `current_setting()`, and views qualified with
another catalog or schema. The guard now rejects every FROM source that is not a plain
view name in the warehouse schema.

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
| 07 | Docker, then Helm on a local Kubernetes cluster | next |
