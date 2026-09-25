# Text-to-SQL evaluation

- questions: 15
- produced a runnable query: 14/15
- correct result: 7/15
- repair attempts used: 8

| question | correct | detail |
| --- | --- | --- |
| team-season-pass-rate | no | expected [(0.6780028943560058,)], got [(0.6777186478449986,)] |
| most-pass-happy-neutral | yes | matches reference |
| third-and-long-epa | no | expected [(-0.0374167889074764,)], got [(-0.017718022624499202,)] |
| red-zone-td-rate | no | Binder Error: Referenced column "field_zone" not found in FROM clause! |
| coldest-game | no | expected [('2015_18_SEA_MIN', -6)], got [(-6, datetime.date(2009, 9, 10))] |
| qb-epa-leader | no | expected [('A.Rodgers',)], got [] |
| fourth-down-attempts-trend | no | expected [(2019, 646), (2020, 726), (2021, 855)], got [(2023, 4490), (2020, 3832), (2022, 4300)] |
| shotgun-share | yes | matches reference |
| home-win-rate | yes | matches reference |
| team-epa-ranking | yes | matches reference |
| chargers-relocation | yes | matches reference |
| kneel-exclusion | no | expected [(506,)], got [(14735,)] |
| two-minute-pass-rate | no | expected [(False, 0.5948739666217329), (True, 0.7909684525214381)], got [(0.7909684525214381, None)] |
| biggest-blowout | yes | matches reference |
| unanswerable-salary | yes | declined |
