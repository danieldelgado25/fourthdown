# Data audit

Generated 2026-11-18 13:33 UTC by `fourthdown audit`. Source: nflverse play-by-play releases.

## Data tests

64 of 64 dbt data tests passed on the last warehouse build (`dbt build` in `fourthdown warehouse`), failures first.

| test | result | detail |
| --- | --- | --- |
| accepted_range fact_plays.qtr | pass | 0 failing rows |
| accepted_range fact_plays.yardline_100 | pass | 0 failing rows |
| accepted_range mart_team_season_tendencies.fourth_down_go_rate | pass | 0 failing rows |
| accepted_range mart_team_season_tendencies.neutral_pass_rate | pass | 0 failing rows |
| accepted_range mart_team_season_tendencies.pass_rate | pass | 0 failing rows |
| accepted_values dim_game.playoff_round | pass | 0 failing rows |
| accepted_values dim_game.season_type | pass | 0 failing rows |
| accepted_values fact_plays.distance_bucket | pass | 0 failing rows |
| accepted_values fact_plays.down | pass | 0 failing rows |
| accepted_values fact_plays.field_zone | pass | 0 failing rows |
| accepted_values fact_plays.play_type | pass | 0 failing rows |
| accepted_values fact_plays.posteam_type | pass | 0 failing rows |
| accepted_values fact_plays.score_state | pass | 0 failing rows |
| accepted_values teams.conference | pass | 0 failing rows |
| every season has 32 teams | pass | 0 failing rows |
| expression_is_true dim_game (season_type = 'REG' or playoff_round is not null) | pass | 0 failing rows |
| expression_is_true fact_plays (is_pass_call is null or is_designed_play) | pass | 0 failing rows |
| finished seasons have full regular seasons | pass | 0 failing rows |
| game attributes are constant within a game | pass | 0 failing rows |
| mean_between fact_plays (cast(is_pass_call as integer)) | pass | 0 failing rows |
| mean_between fact_plays (epa) | pass | 0 failing rows |
| not_null dim_game.away_team | pass | 0 failing rows |
| not_null dim_game.game_id | pass | 0 failing rows |
| not_null dim_game.home_team | pass | 0 failing rows |
| not_null dim_game.season | pass | 0 failing rows |
| not_null dim_game.season_type | pass | 0 failing rows |
| not_null dim_game.week | pass | 0 failing rows |
| not_null dim_player.player_id | pass | 0 failing rows |
| not_null dim_season.season | pass | 0 failing rows |
| not_null dim_team.team | pass | 0 failing rows |
| not_null fact_plays.game_id | pass | 0 failing rows |
| not_null fact_plays.play_id | pass | 0 failing rows |
| not_null fact_plays.play_key | pass | 0 failing rows |
| not_null fact_plays.season | pass | 0 failing rows |
| not_null mart_team_season_tendencies.season | pass | 0 failing rows |
| not_null mart_team_season_tendencies.team | pass | 0 failing rows |
| not_null stg_players.player_id | pass | 0 failing rows |
| not_null stg_plays.game_id | pass | 0 failing rows |
| not_null stg_plays.play_id | pass | 0 failing rows |
| not_null stg_plays.play_key | pass | 0 failing rows |
| not_null stg_plays.season | pass | 0 failing rows |
| not_null teams.team | pass | 0 failing rows |
| null_rate_below fact_plays.posteam_won | pass | 0 failing rows |
| regular season weeks match the schedule | pass | 0 failing rows |
| relationships dim_game.away_team | pass | 0 failing rows |
| relationships dim_game.home_team | pass | 0 failing rows |
| relationships dim_game.season | pass | 0 failing rows |
| relationships fact_plays.defteam | pass | 0 failing rows |
| relationships fact_plays.game_id | pass | 0 failing rows |
| relationships fact_plays.passer_player_id | pass | 0 failing rows |
| relationships fact_plays.posteam | pass | 0 failing rows |
| relationships fact_plays.receiver_player_id | pass | 0 failing rows |
| relationships fact_plays.rusher_player_id | pass | 0 failing rows |
| relationships fact_plays.season | pass | 0 failing rows |
| relationships mart_team_season_tendencies.team | pass | 0 failing rows |
| unique dim_game.game_id | pass | 0 failing rows |
| unique dim_player.player_id | pass | 0 failing rows |
| unique dim_season.season | pass | 0 failing rows |
| unique dim_team.team | pass | 0 failing rows |
| unique fact_plays.play_key | pass | 0 failing rows |
| unique stg_players.player_id | pass | 0 failing rows |
| unique stg_plays.play_key | pass | 0 failing rows |
| unique teams.team | pass | 0 failing rows |
| unique_combination_of_columns mart_team_season_tendencies | pass | 0 failing rows |

## Coverage by season

| season | games | plays | designed_plays | pass_rate | neutral_pass_rate | epa_per_play | epa_null_rate | wp_null_rate | desc_null_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2009 | 267 | 46519 | 34202 | 0.5917 | 0.5581 | -0.0058 | 0.0117 | 0.0059 | 0.0 |
| 2010 | 267 | 46892 | 34387 | 0.6019 | 0.5689 | -0.0047 | 0.0116 | 0.0058 | 0.0 |
| 2011 | 267 | 47448 | 34783 | 0.6069 | 0.5691 | 0.0028 | 0.0114 | 0.0058 | 0.0 |
| 2012 | 267 | 47834 | 35229 | 0.6094 | 0.579 | 0.0168 | 0.0114 | 0.0057 | 0.0 |
| 2013 | 267 | 48158 | 35436 | 0.6192 | 0.5862 | 0.0108 | 0.0113 | 0.0057 | 0.0 |
| 2014 | 267 | 47629 | 35134 | 0.6204 | 0.5871 | 0.0014 | 0.0114 | 0.0057 | 0.0 |
| 2015 | 267 | 48122 | 35457 | 0.629 | 0.5922 | -0.002 | 0.0113 | 0.0057 | 0.0 |
| 2016 | 267 | 47651 | 35190 | 0.6289 | 0.5915 | 0.0173 | 0.0114 | 0.0058 | 0.0 |
| 2017 | 267 | 47245 | 34833 | 0.6159 | 0.5831 | -0.0136 | 0.0114 | 0.0058 | 0.0 |
| 2018 | 267 | 47109 | 34632 | 0.6273 | 0.5903 | 0.0208 | 0.0115 | 0.0058 | 0.0 |
| 2019 | 267 | 47260 | 34952 | 0.6276 | 0.5963 | 0.0099 | 0.0115 | 0.0058 | 0.0 |
| 2020 | 269 | 47705 | 35364 | 0.6238 | 0.5913 | 0.0497 | 0.0113 | 0.0056 | 0.0 |
| 2021 | 285 | 49922 | 37038 | 0.6222 | 0.5914 | 0.0175 | 0.0114 | 0.0057 | 0.0 |
| 2022 | 284 | 49434 | 36616 | 0.6105 | 0.5779 | 0.0076 | 0.0115 | 0.0057 | 0.0 |
| 2023 | 285 | 49665 | 36810 | 0.6217 | 0.5973 | -0.0107 | 0.0115 | 0.0057 | 0.0 |
| 2024 | 285 | 49492 | 36454 | 0.612 | 0.5798 | 0.0233 | 0.0115 | 0.0058 | 0.0 |
| 2025 | 285 | 48771 | 35928 | 0.6113 | 0.5796 | 0.0185 | 0.0117 | 0.0058 | 0.0 |

`pass_rate` counts designed run/pass plays only (kneels, spikes, aborted snaps, and
special teams excluded). `neutral_pass_rate` further restricts to neutral game script:
win probability between 0.2 and 0.8, first three quarters, outside the two-minute
warning. The gap between the two is the game-script effect that a naive tendency stat
mistakes for coaching philosophy.

## Play type distribution

| play_type | plays | share |
| --- | --- | --- |
| pass | 339351 | 0.4154 |
| run | 241637 | 0.2958 |
| no_play | 76193 | 0.0933 |
| kickoff | 47167 | 0.0577 |
| punt | 40421 | 0.0495 |
| (null) | 24090 | 0.0295 |
| extra_point | 21797 | 0.0267 |
| field_goal | 17895 | 0.0219 |
| qb_kneel | 7047 | 0.0086 |
| qb_spike | 1258 | 0.0015 |
