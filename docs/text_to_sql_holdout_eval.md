# Text-to-SQL evaluation (held-out set)

- questions: 15
- produced a runnable query: 15/15
- correct result: 9/15
- repair attempts used: 4

| question | correct | detail |
| --- | --- | --- |
| bills-neutral-pass-rate | no | expected [(0.6877551020408164,)], got [(0.8823529411764706,)] |
| highest-run-rate | no | declined to answer |
| fourth-down-success-rate | yes | matches reference |
| backed-up-drives | no | expected [(452,)], got [(31,)] |
| highest-scoring-game | yes | matches reference |
| receiving-td-leader | yes | matches reference |
| rusher-epa-leader | no | expected [('J.Hurts',)], got [('K.Williams', 0.10435196125965134)] |
| shotgun-vs-under-center | yes | matches reference |
| raiders-home-wins | no | declined to answer |
| titans-first-down-runs | yes | matches reference |
| red-zone-epa | yes | matches reference |
| fourth-down-conversion-leader | yes | matches reference |
| punt-share | yes | matches reference |
| early-vs-late-down-pass-rate | no | expected [(False, 0.8055021624219125), (True, 0.5630528083597686)], got [(1, 0.5212898979927608), (2, 0.6177586206896551), (3, 0.8195649272654477)] |
| unanswerable-attendance | yes | declined |
