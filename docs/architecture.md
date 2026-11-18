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
arithmetic.** Embedding 817k rows and retrieving by cosine similarity cannot answer "how
many", "most", or "average", which is most of what anyone asks of a play-by-play dataset.
So numeric questions go to SQL, narrative questions go to retrieval over generated prose,
and predictive questions go to a trained model. The model's job is routing, query
authoring, and explanation.

## Query routing

| question type | example | path |
| --- | --- | --- |
| aggregate | "EPA per dropback on third and long in 2014" | text-to-SQL over the DuckDB semantic layer |
| narrative | "what happened on the final drive of Super Bowl XLIX" | hybrid BM25 + dense retrieval over drive summaries |
| predictive | "4th and 2 on their own 45, down 4, three minutes left — go for it?" | win-probability model, evaluated for each decision |

## Data layer (built)

```
data/raw/pbp/play_by_play_<season>.parquet      nflverse release, untouched
data/raw/players.parquet                         nflverse player ids -> full names
data/raw/schedules.parquet                       nflverse schedule: which weeks are over
data/processed/plays/season=<season>/week=<ww>.parquet   projected, typed, feature-derived
data/warehouse/fourthdown.duckdb                 built by dbt from dbt_project/
```

One lazy Polars plan does projection, type normalisation, and feature derivation. Output
is one file per NFL week, so a new week is one new file and every other partition keeps
its bytes and its hash. Every derived column is computed from its own row, so a week
transformed alone matches the same week transformed with its season; a test holds that.

### The warehouse is a dbt project (phase 08)

`dbt_project/` turns the processed partitions into the warehouse in four layers, one DuckDB
schema each. `fourthdown warehouse` runs it (`dbt build`, in-process, through
`src/fourthdown/data/modeling.py`):

```
nflverse.processed_plays -> staging.stg_plays --+--> core.dim_season --> core.dim_game --+
nflverse.players -------> staging.stg_players --+                                        |
seed teams.csv ---------------------------------+--> core.dim_team                       |
                                                +--> core.dim_player                     |
                                                +--> core.fact_plays --------------------+
                                                                                         v
                              marts.mart_team_season_tendencies    main.plays, games, drives,
                              (the dashboard's tendency panel)     team_game, player_game, players
                                                                   (text-to-SQL's semantic layer)
```

| layer | schema | materialized | what it is |
| --- | --- | --- | --- |
| staging | `staging` | view | the Parquet as the ETL wrote it, plus a `play_key`; paths relative to the data root |
| core | `core` | table | the star schema: `fact_plays` (one row per play) and `dim_game`, `dim_team`, `dim_player`, `dim_season` |
| marts | `marts` | table | `mart_team_season_tendencies`: one row per team-season, built for one consumer |
| semantic | `main` | table (`plays`, `players`: view) | the six names the text-to-SQL prompt is written against, rebuilt from the star schema |

**Why a star schema under an unchanged semantic layer.** `fact_plays` keeps a play's
situation, outcome, and model outputs, plus its keys. The 19 attributes nflverse repeats
on every play of a game (teams, score, lines, weather, venue, coaches) live once on
`dim_game`. The split is provably lossless: a data test fails if any game carries two
values for one of them, and the slow suite checks `plays` (the fact joined back to the
game) against the processed Parquet with `EXCEPT ALL`. Playoff rounds now come from one
rule on `dim_season` (17 regular-season weeks through 2020, 18 since) instead of a CASE
repeated in three views, and a test holds that rule to the schedule actually played.

The LLM still sees only the six semantic tables. The text-to-SQL prompt, the golden sets,
and their scores are written against those names, and a model handed a star schema
invents joins the way a model handed 372 columns does. The SQL guard already admits only
unqualified names from that list in the `main` schema, so generated SQL cannot reach
`core` or `marts`. Moving the model onto the star schema would be its own experiment,
scored against the holdout set like any other prompt change. The remodel reproduced all
six relations row for row on 2009-2025; one `team_game` average differs in its 17th
significant digit, from summation order.

**`core` and `marts` are written to be portable.** They use `count_if`, `CASE`,
`* EXCLUDE`, and `mode()` rather than DuckDB's `FILTER` and `::` casts, so a second
target (Snowflake) can build them unchanged. The semantic models keep DuckDB SQL, because
they exist for a DuckDB-only consumer.

**Data tests gate the build.** The checks that used to be Python in `audit.py` are now 64
dbt tests:

- `unique` and `not_null` on every grain;
- `relationships` from each fact key to its dimension. Players credited in the
  play-by-play but missing from nflverse's player table still get a `dim_player` row, so
  the passer, rusher, and receiver keys always resolve;
- `accepted_values` on every label column;
- custom generic tests in `dbt_project/macros/tests/` for what the built-ins cannot say:
  value ranges, a league mean inside a band (pass rate 0.52-0.65, EPA per play within
  0.05 of zero), and a null-rate ceiling on the win label. They are macros rather than
  `dbt_utils` so that a build never runs `dbt deps` or needs the network.

`dbt build` runs each model's tests before building what depends on it. `warehouse.build`
builds into a side directory and swaps the file in only when everything passed, so a bad
upstream release fails the pipeline while the live warehouse keeps serving.
`fourthdown audit` renders the last build's results beside the coverage tables in
[data_audit.md](data_audit.md).

**One profile for the laptop, the container, and the tests.** The profile reads its paths
from the environment (`FOURTHDOWN_DATA_DIR`, `FOURTHDOWN_DBT_DATABASE`). dbt's target and
log directories go under the data root, because the project directory is read-only in the
container. `fourthdown dbt <command>` sets the same environment around any dbt command
(`fourthdown dbt docs generate`, `fourthdown dbt test --select fact_plays`). CI uploads the
generated docs site, whose lineage graph runs from the Parquet sources to the semantic
tables.

### Dagster runs the pipeline as an asset graph (phase 09)

`src/fourthdown/orchestration/` replaces `docker/pipeline.sh`, a script that rebuilt
everything unless a stamp file said the data was current. The pipeline is now one Dagster
asset graph:

```
raw_plays[week] -> processed_plays[week] -> dbt: stg_* -> fact_plays, dim_* -> mart_*, semantic
raw_players ---------------------------------^                                    |
                                                         narrative_index <--------+
                                                         trained_models -> model_scorecard
```

| asset | partitioned | what it does |
| --- | --- | --- |
| `raw_plays` | by week | conditional download of the season file; checks the week against the schedule |
| `processed_plays` | by week | the Polars transform of that week, `season=S/week=WW.parquet` |
| `raw_players` | no | nflverse's player table, refreshed so new players get full names |
| dbt models (21) | no | `dbt build` via dagster-dbt; the 64 dbt tests are asset checks |
| `narrative_index` | no | embeds the narratives not yet in pgvector |
| `trained_models` | no | train, register, serve through the gates; skipped when nothing changed |
| `model_scorecard` | no | offline suites plus the training run's metrics; a `gates_pass` check |

**Partitions are weeks that have been played.** Keys are `<season>-<week>`, the same key
the manifest and the file layout use. The set is dynamic rather than a static season x
week grid because seasons differ in length (21 weeks through 2020, 22 since) and a week
exists only once it is over. The nflverse schedule decides that. A week is complete when
every game listed for it has a result, and `raw_plays` also requires every one of those
games in the play-by-play. A week that is not ready fails with the reason, such as
"not every game of the week has a final result yet". Every season from 2009 to 2025 was
checked against this rule with no exceptions: cancelled games are not in the schedule.

**The ingest assets are partitioned; everything downstream reads all weeks.** A model
trains on every season, and the semantic tables span them. So dbt, the index, training,
and the scorecard are unpartitioned assets that depend on all of `processed_plays`. The
ingest assets take a single-run backfill policy, so a 362-week backfill is one run with one
download per season, not 362 runs.

**A new week does not rebuild what it did not change:**

- *Download.* The season file is requested with `If-None-Match` and the recorded ETag.
  An unchanged file costs one request.
- *Transform.* Only the week's file is written. The other weeks keep their bytes, their
  manifest records, and so their place in `data_version`.
- *dbt.* `warehouse.plan` compares the partition hashes the live warehouse was built
  from (`built_from` in the manifest) with those on disk. When weeks were only added or
  changed, dbt runs on a copy of the live database, and `fact_plays` (incremental,
  `delete+insert` keyed by `game_id`) rebuilds just those weeks' games. Overwriting by
  game, not by play, also drops a play nflverse deleted in a correction. The dimensions,
  mart, and semantic tables are small enough to rebuild each time. Changed dbt code, a
  removed partition, or no live warehouse means a full build. A test holds the
  incremental result equal to a full rebuild.
- *Index.* Only documents whose id is not in pgvector are embedded.
- *Training.* Retraining is inherent: the model fits on every season. It is skipped when
  the last run, served or not, used this data version and these settings.

Measured on the real data: the first run on 2009-2026 week 9 took 3.5 minutes. The next
run added 2026 week 10 alone and took 2m17s. Ingest took 1.5 seconds, the transform 0.1
seconds, and the incremental dbt build 6 seconds with all 64 tests (`fact_plays` 2.0
seconds instead of 6.2). Training took 1m36s and the scorecard 6 seconds. A rerun with
nothing new took 30 seconds and did not train.

**Schedule and sensor.** `weekly_ingest` runs on Tuesdays at 10:00 New York time,
September through February, after Monday night's game. It registers the current season's
newly completed weeks and ingests them, plus the latest week again, since nflverse
corrects stats in the days after a game. `new_data_sensor` compares the latest
`processed_plays` materialization with the latest `fact_plays` materialization. When data
is newer than the warehouse, it runs `publish` once, however many weeks landed. Its
`run_key` is that materialization's storage id, so a failed publish is not retried on the
same data. A retrain is promoted only through the registry gates, so a bad week cannot
replace the served model.

**dbt inside Dagster.** dagster-dbt turns each model into an asset and each test into an
asset check, with the dbt sources mapped onto the ingest assets. The run still builds on
a staging copy and swaps it in only when everything passed. Dagster's events are held
until the swap, so a materialization in Dagster always describes the live warehouse. A
failing test reports its check as failed, and the run fails while the live warehouse
keeps serving. Selecting only part of the dbt graph is refused, because the warehouse is
swapped in whole.

**One process in Kubernetes, a daemon in Compose.** `fourthdown pipeline` runs the graph
once, in-process: register the completed weeks, ingest the ones not processed yet, then
`publish`. The Helm Job runs it, and so does the compose `pipeline` service. The Helm Job
waits for Postgres and the embedding model first when the index is on. For ongoing weekly
updates, the compose `dagster` profile adds the webserver (UI on port 3000) and the
daemon, which runs the schedule and sensor. Both share one instance on the data volume
(`docker/dagster.yaml`). The run queue allows one run at a time, because every run writes
the same DuckDB file, Parquet, and MLflow registry. The cluster has no daemon. A weekly
cluster refresh is a new pipeline Job, and that is a no-op when nothing landed.

### Data quality gates every stage (phase 10)

`src/fourthdown/quality/` holds what each stage must be true of before the next one may
use it. Every check reports through `DataQualityError`, one line per failure, in words,
with example values. That way a failed run's message is the diagnosis.

**Contracts (Pandera) at the two boundaries.** The raw contract is structural, because
the raw file belongs to someone else. It checks that the columns the ETL cannot do without
are present (`schema.REQUIRED_SOURCE_COLUMNS`), that the columns read as numbers are
numbers, that play keys are present and unique, and that the file holds the season it is
named for. This matters because the ETL used to fill a missing column with nulls and cast
text to numbers without complaint: a renamed `down` would have become a season of null
downs and empty buckets. The processed contract covers the table everything downstream
trusts: exactly the 117 contracted columns in order, each with its type; value ranges set
from profiling every play since 2009 (down 1-4, quarter 1-6, timeouts 0-3, probabilities
0-1, and so on); labels from known sets; and only the partition's own season and week.
The ETL now casts every column to its contract type, so its output no longer depends on the
input's types; on real data the casts are a no-op, with byte-identical partitions. Each
week passes before any is written, so a bad file writes nothing.

**Row counts are their own check,** because a file can meet every column contract and
still be wrong. From 2009 to 2025 every game has 135-245 rows and every week averages
161-207 per game. A game under 100 or over 300 rows, a week averaging outside 145-225, or
a re-delivered week more than 5% off its last ingest fails. The last one is drift:
nflverse republishes weeks for days after them, and a correction moves a handful of plays,
not a twentieth of the week.

**A rejected download never replaces a good file.** A new season file lands beside the
old one and must pass the raw contract and the row counts first. A file that fails goes to
`data/raw/quarantine/` with a timestamp, and the previous file and its manifest record stay.
The next run asks nflverse again with the previous ETag. Completeness ("the week is not
over") never quarantines a file, since that is not the file's fault.

**The leakage contract** is enforced before any training: by `release.train_and_release`
(so `fourthdown train` too), and in Dagster as a blocking check on `plays`, so a broken
contract skips `trained_models`. It has two layers, because neither catches what the other
does:

| layer | rule | what only it catches |
| --- | --- | --- |
| declared | every feature is in `schema.PRE_SNAP_COLUMNS`, none in `LEAKAGE_COLUMNS` | `wp` (AUC 0.84 alone, the same as `score_differential`'s 0.83; no data test could single it out) |
| measured | no feature alone predicts its target with AUC above 0.95 | a post-snap outcome declared as pre-snap: `pass_play` predicts the play call with AUC 1.0 |

The real features top out at 0.83 (`score_differential` for the winner, `shotgun` 0.71 for
the call). The AUCs are reported with every check, so a feature creeping toward the line
is visible before it crosses it.

**Freshness is measured against the schedule, not a clock.** "No update in seven days" is
wrong in a bye week and in February. "A week ended two days ago and isn't in the data" is
right all year. The check also notices nflverse falling behind: games two days past with
no result. It runs daily at noon New York time, September to February, and passes out of
season.

**Alerts.** The freshness check alerts when it fails, and a run-failure sensor sends
every failed run with each failing step's message. Alerts always go to the log and the
Dagster UI, and to a Slack or Discord webhook when `FOURTHDOWN_ALERT_WEBHOOK` is set. The
URL is a credential and is never logged.

**What it found.** The processed contract's first pass over 2009-2025 failed on one value
in 816,856 plays. nflverse's timeout counter reads -1 after a third timeout in playoff
overtime (2025 week 20, BUF at DEN). The ETL now floors it at 0, which changed that one
partition. When nflverse's real 2026 file was republished with one week-10 game cut to 40
plays, the run failed on both row-count checks, quarantined the download, and left the
previous file, the week's partition, and the live warehouse as they were.

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

**Ask a question is database access, so the whole warehouse has to be reachable.** Three
changes follow from treating it that way rather than as a demo of the questions it was
tuned on:

- *Totals come from the box score.* "Most rushing yards in 2025" summed designed runs in
  `plays` (BAL, 2,372) where the box score says BUF, 2,714. The card now says yards,
  touchdowns, and catches are `player_game` sums by role, grouped by team for a team, and
  an exemplar shows a team passing-yards leader.
- *Facts go to the warehouse.* "Super Bowl" alone was a narrative cue, so "who won Super
  Bowl LX" was answered from generated summaries. Who won, winners, champions, and
  leaders are now stats cues; how a game unfolded stays on narrative.
- *Every `plays` column is reachable without a bigger card.* The card names 37 of the 117
  columns. Each of the others has cues (its own name read as words, plus phrases such as
  "penalties" or "pass rate over expected"), and the up to 12 a question matches are
  listed after the card with notes on their values (`schema_card.relevant`). The prompt
  grows only for questions that need it. Touchdown and conversion flags have no cues:
  `player_game` and `team_game` hold those totals, and listing the `plays` flags beside
  them pulled the model off the right view.

Thirteen development questions cover them (penalties, air yards, field-goal distance,
sacks, interceptions, pass rate over expected, two-point tries, scrambles, overtime,
yards after the catch, lost fumbles, a box-score team total, a Super Bowl winner).

| run (qwen2.5-coder:7b, CPU) | development (68) | held-out |
| --- | --- | --- |
| before | 39/68 | 9/15 |
| box score + routing + per-question columns | 44/68 | 9/15 |
| without touchdown and conversion cues | 46/68 | 9/15 |
| + `team` rule, repair hints, single-row check | 46/68 | 9/15 |
| + two-point and kick check | 47/68 | 10/15 |

The gain is seven questions (the box-score leader, both sack questions, penalties,
field-goal distance, two-point tries, overtime). The first run also lost four; two were
the touchdown and conversion cues described above, and both pass once those cues are
gone. Three losses remain against the baseline. Two are queries that group `drives` or
`plays` by `team`, a column only the box-score views have, which the new box-score rule
("GROUP BY team for a team") probably invites; one sums `total_points` over a season
instead of taking the top game.

Those three were fixed without more prompt rules where code could decide. The box-score
rule now says only `player_game` and `team_game` have `team`, and a repair hint gives the
view's own spelling ("drives has no team; use posteam or defteam") rather than only the
views that own the column, which had sent the model to `team_game` without
`scored_touchdown`. Two query shapes that always mean a mistake go back for repair before
they run: an aggregate without GROUP BY under ORDER BY (a season's `sum(total_points)`
for "the highest-scoring game"), and a two-point try, punt, field goal, or extra point
counted under `is_designed_play`, which no such play is. All three questions pass in the
rerun, and the development total stays 46/68: three others flipped, two of which
(`coldest-game`, `most-pass-happy-neutral`) have flipped between runs before. The third,
`two-point-attempts`, was the `is_designed_play` shape; with that check in, a full rerun
scores 47/68 and 10/15. Against the 39/68 baseline that is ten questions gained and one
lost (`player-completions`, which the model declined to answer).

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

**Rows are not embedded.** 817k plays would be 817k vectors of mostly boilerplate, and
cosine similarity over them answers nothing a `GROUP BY` does not answer better. 4.6k
games plus 105k drives is the grain at which a question is actually asked.

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
([routing_eval.md](routing_eval.md)), and they scored the same 19/20 — which is the point:
the fallback is not a degraded mode, it is a cheap router the LLM has to beat. On the
29-question set, after facts about a Super Bowl moved to stats, the keyword router scores
29/29 and the LLM and hybrid routers 28/29: the model sends "which team had the highest EPA
per play" to the tendency tool.

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
| `etl` | per weekly partition (`2026-10`): rows, SHA-256, the SHA-256 of the raw file it came from, and a hash of `etl.py` + `schema.py` |
| `warehouse` | per dbt model and seed: the SHA-256 of its file, its schema and materialization, and what it reads; plus one hash over the macros, seed, and `dbt_project.yml`; and the partition hashes it was built from, which the next build diffs |

nflverse republishes the current season's file in place, so a season number does not
identify data; the hash does. `data_version` is a 12-character hash over the processed
partition hashes and the dbt code. `fourthdown lineage --verify` rehashes every file and
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
| `fourthdown-pipeline` | every extra, typer, MLflow, dbt, Dagster | the pipeline Job, the Dagster UI and daemon, `fourthdown wait` |
| `fourthdown-winprob` | one trained model and its card (above) | the standalone model service |

**Data is built where it is served.** The pipeline Job runs `fourthdown pipeline`, the
Dagster asset graph, against the volume mounted at `/data`. Weeks already processed are
not ingested again, dbt has nothing to rebuild, and training is skipped while the data
version and training flags are unchanged. Training refuses data that fails
`fourthdown lineage --verify`, so the provenance checks run inside the cluster too. In Compose the volume is the
repo's `data/`, so a build on the host (`make build train`) serves just as well.

**Ordering is mostly unnecessary.** The services heal in place (above), so they start
alongside the pipeline Job and turn ready as the warehouse and models appear. Only the
narrative index needs ordering. When it is enabled, the pipeline Job runs `fourthdown
wait` as an init container, polling for Postgres and the embedding model, and on timeout
reports which check failed.

**Jobs are named by their inputs.** A Job's spec is immutable, so the pipeline and
model-pull Jobs carry a hash of their inputs: for the pipeline, the seasons, training
flags, index seasons, embedding model, and image. Changing an input
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
| 08 | dbt: star schema, tendency mart, data tests that gate the build, dbt docs in CI | done |
| 09 | Dagster asset graph partitioned by NFL week, incremental dbt, weekly schedule, new-data sensor | done |
| 10 | Pandera contracts at ingest, row-count drift, quarantine, a leakage contract that blocks training, freshness alerts | done |
