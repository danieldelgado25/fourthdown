# Data audit

Generated 2026-09-22 18:47 UTC by `fourthdown audit`. Source: nflverse play-by-play releases.

## Checks

| check | result | detail |
| --- | --- | --- |
| unique play keys | pass | 0 duplicated (game_id, play_id) |
| 32 teams per season | pass | max distinct posteam = 32 |
| full regular seasons | pass | fewest regular season games in a season = 256 |
| league pass rate in [0.52, 0.65] | pass | pass rate = 0.6169 |
| EPA per play near zero | pass | mean EPA = 0.0089 |
| downs within 1-4 | pass | 0 out-of-range downs |
| yardline_100 within 0-100 | pass | 0 out-of-range yardlines |
| play-call label only on designed plays | pass | 0 labelled non-designed plays |
| win label present (ties excepted) | pass | null rate = 0.0034 |

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

`pass_rate` counts designed run/pass plays only (kneels, spikes, aborted snaps, and
special teams excluded). `neutral_pass_rate` further restricts to neutral game script:
win probability between 0.2 and 0.8, first three quarters, outside the two-minute
warning. The gap between the two is the game-script effect that a naive tendency stat
mistakes for coaching philosophy.

## Play type distribution

| play_type | plays | share |
| --- | --- | --- |
| pass | 319614 | 0.4161 |
| run | 226742 | 0.2952 |
| no_play | 71470 | 0.093 |
| kickoff | 44240 | 0.0576 |
| punt | 38379 | 0.05 |
| (null) | 22645 | 0.0295 |
| extra_point | 20467 | 0.0266 |
| field_goal | 16755 | 0.0218 |
| qb_kneel | 6595 | 0.0086 |
| qb_spike | 1178 | 0.0015 |
