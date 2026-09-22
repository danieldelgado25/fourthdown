# Data dictionary

The processed play table is a projection of the nflverse play-by-play release (372
columns per season) down to the columns this project actually queries, plus twelve
derived columns. `src/fourthdown/data/schema.py` is the authoritative list; this document
explains the parts that are not self-describing.

## Grain and keys

| concept | key | notes |
| --- | --- | --- |
| play | `(game_id, play_id)` | Includes penalties (`play_type = 'no_play'`), timeouts, and administrative rows. |
| drive | `drive_id` = `game_id` + `-` + `fixed_drive` | `fixed_drive` is nflverse's repaired drive counter; the raw `drive` column is unreliable. |
| game | `game_id` | e.g. `2023_01_ARI_WAS`. |
| team-game | `(game_id, posteam)` | Offensive plays only. |

## Renamed source columns

| source | processed | reason |
| --- | --- | --- |
| `desc` | `play_desc` | `DESC` is a SQL keyword and breaks generated queries. |
| `pass` | `pass_play` | Reads as a verb in SQL. Note it is a *dropback* flag: sacks and scrambles count as passes. |
| `rush` | `rush_play` | Symmetry with `pass_play`. |
| `special` | `special_teams` | `special` alone is ambiguous. |

## Derived columns

| column | definition |
| --- | --- |
| `posteam_is_home` | `posteam = home_team`. |
| `posteam_won` | Game outcome from the offense's perspective; `NULL` on ties (0.34% of plays). Derived from `result`, the final home margin. |
| `distance_bucket` | `short` ≤ 2, `medium` 3-6, `long` 7-10, `very_long` > 10 yards to go; `NULL` when there is no down. |
| `field_zone` | By `yardline_100` (yards to the opponent's goal line): `red_zone` ≤ 20, `opponent_territory` ≤ 40, `midfield` ≤ 60, `own_territory` ≤ 80, `backed_up` > 80. |
| `score_state` | Seven buckets from `trailing_big` (≤ -17) to `leading_big` (> +16), with `tied` in the middle. |
| `is_early_down` | Down is 1 or 2. |
| `is_two_minute` | `half_seconds_remaining` ≤ 120. |
| `is_neutral_script` | Win probability in [0.2, 0.8], first three quarters, outside the two-minute warning. The standard filter for measuring tendency without game-script contamination. |
| `is_garbage_time` | Fourth quarter with win probability outside [0.05, 0.95]. |
| `is_designed_play` | A genuine pre-snap run/pass decision: `pass_play` or `rush_play`, a down exists, and the play is not a kneel, spike, aborted snap, deleted row, or special teams. |
| `is_pass_call` | The play-call label — `pass_play` restricted to designed plays, `NULL` elsewhere. Every play-call model trains on this, never on `play_type`. |
| `drive_id` | See keys above. |

## Columns that must not be used as features

`schema.LEAKAGE_COLUMNS` lists nflfastR's own model outputs: `ep`, `epa`, `qb_epa`, `wp`,
`def_wp`, `wpa`, `vegas_wp`, `vegas_wpa`, `cp`, `cpoe`, `success`, `xpass`, `pass_oe`,
`xyac_epa`, `xyac_mean_yardage`.

They are kept in the warehouse because they are the right tool for analytics questions
and the honest baseline to compare against — `xpass` in particular is a fitted
pass-probability model, which is exactly what our play-call model predicts. Training on
them produces a model that scores well and has learned nothing: `wp` is an outcome-aware
estimate, and `epa` is computed after the play resolves. The split is enforced by
convention now and by the feature-builder in phase 04.

## Known data quirks

- **Null `play_type` (2.9% of plays).** Timeouts, quarter ends, and administrative rows.
  They carry no down and are excluded from every designed-play aggregate.
- **`no_play` (9.3%).** Plays wiped out by a penalty. The `desc` field still describes what
  happened, which matters for the narrative corpus but not for play-call modelling.
- **EPA/WP nulls (~1.1% / ~0.6%).** Mostly the same administrative rows, plus end-of-half
  situations where the model has no defined state.
- **Season length changes.** 256 regular-season games per year through 2020, 272 from
  2021 (17-game season). Per-season totals are not comparable without normalising.
- **Relocations are back-filled.** nflverse normalises team abbreviations to their current
  form across all history: there is no `SD`, `STL`, or `OAK` anywhere in 2009-2024, only
  `LAC`, `LA`, and `LV` (32 distinct values total). Convenient for joins, but it means a
  question about "the San Diego Chargers in 2013" must be mapped to `LAC`, and any answer
  should say so rather than silently renaming history. The alias table belongs in the
  text-to-SQL prompt.
