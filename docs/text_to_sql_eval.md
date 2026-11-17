# Text-to-SQL evaluation

- questions: 55
- produced a runnable query: 54/55
- correct result: 36/55
- repair attempts used: 12

| question | correct | detail |
| --- | --- | --- |
| team-season-pass-rate | yes | matches reference |
| most-pass-happy-neutral | no | Binder Error: Referenced column "is_pass_call" not found in FROM clause! |
| third-and-long-epa | yes | matches reference |
| red-zone-td-rate | yes | matches reference |
| coldest-game | no | expected [('2015_18_SEA_MIN', -6)], got [('2016_15_TEN_KC', 1)] |
| qb-epa-leader | no | expected [('P.Mahomes',)], got [('M.Stafford', 0.21926968590380297)] |
| fourth-down-attempts-trend | no | expected [(2019, 618), (2020, 691), (2021, 820)], got [(2021, 855), (2019, 646), (2022, 786)] |
| shotgun-share | yes | matches reference |
| home-win-rate | yes | matches reference |
| team-epa-ranking | yes | matches reference |
| chargers-relocation | no | expected [(1103,)], got [(1210,)] |
| kneel-exclusion | no | expected [(458,)], got [(0,)] |
| two-minute-pass-rate | no | expected [(False, 0.5945784180412651), (True, 0.7897321691935579)], got [(0.6160540059563352, 0.043721998373022)] |
| biggest-blowout | yes | matches reference |
| receiver-receptions-full-name | yes | matches reference |
| receiver-targets | yes | matches reference |
| week-opponent | yes | matches reference |
| matchup-total-points | no | expected [(105,)], got [(145,)] |
| unanswerable-before-coverage | yes | declined |
| playoff-player-total | yes | matches reference |
| playoff-round-winner | yes | matches reference |
| super-bowl-numeral | yes | matches reference |
| playoff-touchdown-passes | yes | matches reference |
| playoff-wins-span | no | expected [(15,)], got [(45,)] |
| playoff-drives | yes | matches reference |
| regular-season-wins | no | expected [(14,)], got [(8,)] |
| regular-season-default-player | yes | matches reference |
| unanswerable-salary | yes | declined |
| empty-lions-super-bowl | no | invented an answer: [('2023_19_LA_DET',), ('2014_18_DET_DAL',), ('2011_18_DET_NO',)] |
| empty-brady-2023 | yes | found nothing, correctly |
| empty-browns-2015-playoffs | yes | found nothing, correctly |
| super-bowl-lx-winner | yes | matches reference |
| team-rushing-leader-2025 | yes | matches reference |
| team-third-down-leader | yes | matches reference |
| team-fourth-down-conversions | yes | matches reference |
| team-games-above-epa | no | declined to answer |
| team-most-interceptions | yes | matches reference |
| team-fumbles-lost | yes | matches reference |
| team-best-game-epa | no | expected [('TEN', 'DET')], got [('TEN', 'DET', 0.49913048292054)] |
| drive-count | yes | matches reference |
| drive-punt-share | yes | matches reference |
| drive-td-share-leader | yes | matches reference |
| drive-turnover-on-downs | no | expected [(349,)], got [(636,)] |
| games-outdoors | no | expected [(177,)], got [(0,)] |
| games-highest-scoring | yes | matches reference |
| games-home-losses | yes | matches reference |
| player-rushing-leader | yes | matches reference |
| player-receiving-tds | yes | matches reference |
| player-single-game-passing | no | expected [('J.Burrow',)], got [('P.Mahomes', 5250)] |
| player-100-yard-games | no | expected [(10,)], got [(1,), (1,), (1,)] |
| player-completions | yes | matches reference |
| plays-sacks | no | declined to answer |
| plays-air-yards | no | expected [(8.144904458598726,)], got [(6.502086230876217,)] |
| plays-punts | yes | matches reference |
| players-position | yes | matches reference |
