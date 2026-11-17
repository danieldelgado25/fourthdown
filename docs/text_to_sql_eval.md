# Text-to-SQL evaluation

- questions: 68
- produced a runnable query: 66/68
- correct result: 46/68
- repair attempts used: 17

| question | correct | detail |
| --- | --- | --- |
| team-season-pass-rate | yes | matches reference |
| most-pass-happy-neutral | yes | matches reference |
| third-and-long-epa | yes | matches reference |
| red-zone-td-rate | yes | matches reference |
| coldest-game | yes | matches reference |
| qb-epa-leader | no | expected [('P.Mahomes',)], got [('M.Stafford', 0.21926968590380297)] |
| fourth-down-attempts-trend | no | expected [(2019, 618), (2020, 691), (2021, 820)], got [(2019, 646), (2020, 726), (2021, 855)] |
| shotgun-share | yes | matches reference |
| home-win-rate | yes | matches reference |
| team-epa-ranking | yes | matches reference |
| chargers-relocation | no | expected [(1103,)], got [(1210,)] |
| kneel-exclusion | no | expected [(458,)], got [(0,)] |
| two-minute-pass-rate | no | expected [(False, 0.5945784180412651), (True, 0.7897321691935579)], got [(False, 0.5940360472002925), (True, 0.7895628600474416)] |
| biggest-blowout | yes | matches reference |
| receiver-receptions-full-name | yes | matches reference |
| receiver-targets | yes | matches reference |
| week-opponent | yes | matches reference |
| matchup-total-points | no | expected [(105,)], got [(823,)] |
| unanswerable-before-coverage | yes | declined |
| playoff-player-total | yes | matches reference |
| playoff-round-winner | yes | matches reference |
| super-bowl-numeral | yes | matches reference |
| playoff-touchdown-passes | yes | matches reference |
| playoff-wins-span | no | expected [(15,)], got [(97,)] |
| playoff-drives | yes | matches reference |
| regular-season-wins | no | expected [(14,)], got [(8,)] |
| regular-season-default-player | yes | matches reference |
| unanswerable-salary | yes | declined |
| empty-lions-super-bowl | no | invented an answer: [('2023_19_LA_DET',), ('2014_18_DET_DAL',), ('2023_20_TB_DET',)] |
| empty-brady-2023 | yes | found nothing, correctly |
| empty-browns-2015-playoffs | yes | found nothing, correctly |
| super-bowl-lx-winner | yes | matches reference |
| team-rushing-leader-2025 | yes | matches reference |
| team-third-down-leader | yes | matches reference |
| team-fourth-down-conversions | yes | matches reference |
| team-games-above-epa | no | expected [(3,)], got [(76,), (54,), (64,)] |
| team-most-interceptions | no | Binder Error: Referenced column "team" not found in FROM clause! |
| team-fumbles-lost | yes | matches reference |
| team-best-game-epa | no | expected [('TEN', 'DET')], got [('TEN', 'DET', 2020, 'REG', 15, 0.49913048292054)] |
| drive-count | yes | matches reference |
| drive-punt-share | yes | matches reference |
| drive-td-share-leader | no | Binder Error: Referenced column "team" not found in FROM clause! |
| drive-turnover-on-downs | no | expected [(349,)], got [(636,)] |
| games-outdoors | yes | matches reference |
| games-highest-scoring | no | expected [(101,)], got [(11680,)] |
| games-home-losses | yes | matches reference |
| player-rushing-leader | yes | matches reference |
| player-receiving-tds | yes | matches reference |
| player-single-game-passing | no | expected [('J.Burrow',)], got [('P.Mahomes', 5250)] |
| player-100-yard-games | no | expected [(10,)], got [(1,), (1,), (1,)] |
| player-completions | yes | matches reference |
| plays-sacks | yes | matches reference |
| plays-air-yards | no | expected [(8.144904458598726,)], got [(8.205128205128204,)] |
| plays-punts | yes | matches reference |
| players-position | yes | matches reference |
| penalties-on-offense | yes | matches reference |
| air-yards-team-leader | no | declined to answer |
| longest-field-goal | yes | matches reference |
| team-sacks-defense | yes | matches reference |
| most-interceptions-thrown-team | yes | matches reference |
| pass-rate-over-expected-leader | no | expected [('KC',)], got [('DAL', 74.1021357602709)] |
| two-point-attempts | yes | matches reference |
| qb-scrambles | no | expected [(66,)], got [(0,)] |
| overtime-games | yes | matches reference |
| yards-after-catch-leader | no | expected [('A.Kamara',)], got [('S.Diggs', 1535)] |
| fumbles-lost-team | yes | matches reference |
| team-rushing-yards-box-score | yes | matches reference |
| super-bowl-lvii-winner | yes | matches reference |
