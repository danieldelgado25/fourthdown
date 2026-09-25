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
golden questions; that comparison is the justification for the extra moving part.

**Numbers still come from SQL.** The narratives contain numbers, but a retrieved passage
is evidence about *which* game, not the authority on a total. `fourthdown ask` stays the
path for aggregates; `fourthdown explain` answers from passages and cites them.

## Phase status

| phase | scope | status |
| --- | --- | --- |
| 00 | scope, repo, data audit | done |
| 01 | Polars ETL, Parquet, DuckDB semantic layer | done |
| 02 | schema card, text-to-SQL, sqlglot guardrails, golden set | done |
| 03 | narrative corpus, embeddings, hybrid retrieval | done |
| 04 | win probability, 4th-down advisor, play-call model | next |
| 05 | orchestrator, Flask API, React dashboard | planned |
| 06 | evaluation harness in CI | planned |
| 07 | Docker, then Helm on a local Kubernetes cluster | planned |
