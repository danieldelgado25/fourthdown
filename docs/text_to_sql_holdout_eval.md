# Text-to-SQL evaluation (held-out set)

- questions: 15
- produced a runnable query: 14/15
- correct result: 9/15
- repair attempts used: 2

| question | correct | detail |
| --- | --- | --- |
| bills-neutral-pass-rate | no | expected [(0.6791744840525328,)], got [(0.8947368421052632,)] |
| highest-run-rate | no | Binder Error: Referenced column "is_designed_play" not found in FROM clause! |
| fourth-down-success-rate | yes | matches reference |
| backed-up-drives | yes | matches reference |
| highest-scoring-game | yes | matches reference |
| receiving-td-leader | yes | matches reference |
| rusher-epa-leader | yes | matches reference |
| shotgun-vs-under-center | yes | matches reference |
| raiders-home-wins | no | declined to answer |
| titans-first-down-runs | no | expected [(312,)], got [(7435,)] |
| red-zone-epa | yes | matches reference |
| fourth-down-conversion-leader | yes | matches reference |
| punt-share | yes | matches reference |
| early-vs-late-down-pass-rate | no | expected [(False, 0.8059323982524719), (True, 0.5647054638588503)], got [(1, 0.5225134830051423), (2, 0.6200065757027783), (3, 0.8201098760700141)] |
| unanswerable-attendance | no | answered a question it cannot answer |
