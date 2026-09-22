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

## Phase status

| phase | scope | status |
| --- | --- | --- |
| 00 | scope, repo, data audit | done |
| 01 | Polars ETL, Parquet, DuckDB semantic layer | done |
| 02 | schema card, text-to-SQL, sqlglot guardrails, golden set | next |
| 03 | narrative corpus, embeddings, hybrid retrieval | planned |
| 04 | win probability, 4th-down advisor, play-call model | planned |
| 05 | orchestrator, Flask API, React dashboard | planned |
| 06 | evaluation harness in CI | planned |
| 07 | Docker, then Helm on a local Kubernetes cluster | planned |
