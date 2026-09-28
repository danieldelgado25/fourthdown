# Model evaluation

Every number below is measured on held-out seasons (2021-2024); models were fit on 2009-2018 and early-stopped on 2019-2020.

## Win probability

Splits: train 399,601 rows (2009-2018), valid 79,345 rows (2019-2020), test 166,371 rows (2021-2024).
Best epoch 8 of 14 (validation log loss 0.4269).

| model | rows | log loss | Brier | AUC | accuracy | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| fourthdown MLP | 166,371 | 0.4553 | 0.1509 | 0.8634 | 0.777 | 0.0066 |
| nflfastR vegas_wp | 166,371 | 0.4548 | 0.1507 | 0.8637 | 0.776 | 0.0063 |
| nflfastR wp | 166,371 | 0.4890 | 0.1654 | 0.8342 | 0.741 | 0.0110 |

The baselines are nflfastR's own fitted win-probability columns, scored on the same plays. They are never features -- `features.feature_matrix` raises if one reaches the design matrix -- so this is a comparison, not a leak.

### Calibration

| bucket | plays | predicted | observed |
| --- | --- | --- | --- |
| 0.0-0.1 | 23,696 | 0.038 | 0.039 |
| 0.1-0.2 | 14,092 | 0.149 | 0.167 |
| 0.2-0.3 | 13,781 | 0.250 | 0.246 |
| 0.3-0.4 | 14,312 | 0.350 | 0.335 |
| 0.4-0.5 | 15,745 | 0.449 | 0.445 |
| 0.5-0.6 | 14,523 | 0.551 | 0.562 |
| 0.6-0.7 | 15,826 | 0.650 | 0.659 |
| 0.7-0.8 | 14,658 | 0.750 | 0.752 |
| 0.8-0.9 | 15,343 | 0.851 | 0.845 |
| 0.9-1.0 | 24,395 | 0.961 | 0.963 |

## Fourth-down advisor

- fourth downs audited: 4,000
- coach and advisor agree: 66.6%
- coaches went for it: 20.8%; advisor would: 42.3%
- mean win probability surrendered by the actual choice: 0.40 points

The advisor has no parameters of its own: it evaluates the win-probability model at the states each option leads to, weighted by a conversion model, a field-goal model, and the empirical punt landing spot.

## Play call (run or pass)

Splits: train 349,283 rows (2009-2018), valid 70,316 rows (2019-2020), test 146,918 rows (2021-2024).

| model | rows | log loss | Brier | AUC | accuracy | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| fourthdown GBM | 146,918 | 0.5077 | 0.1707 | 0.8153 | 0.740 | 0.0369 |
| base rate | 146,918 | 0.6657 | 0.2364 | 0.5000 | 0.617 | 0.0014 |
| nflfastR xpass | 146,918 | 0.5254 | 0.1788 | 0.7902 | 0.711 | 0.0108 |

### Predictability

How often the model's call was the call, on early-down neutral-script plays, by team-season. 0.5 is a coin flip; `confidence` is how sure the model was.

Most predictable:

| season | team | plays | pass rate | predictability | confidence | EPA/play |
| --- | --- | --- | --- | --- | --- | --- |
| 2021 | ATL | 400 | 53.0% | 0.750 | 0.656 | +0.014 |
| 2021 | CIN | 467 | 55.2% | 0.747 | 0.662 | +0.050 |
| 2022 | LV | 427 | 49.6% | 0.740 | 0.655 | +0.012 |
| 2021 | LA | 500 | 57.4% | 0.740 | 0.668 | -0.019 |
| 2023 | LA | 460 | 56.1% | 0.735 | 0.666 | -0.022 |

Least predictable:

| season | team | plays | pass rate | predictability | confidence | EPA/play |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | BAL | 463 | 51.8% | 0.525 | 0.664 | +0.046 |
| 2024 | PHI | 472 | 47.5% | 0.525 | 0.667 | +0.096 |
| 2024 | GB | 421 | 42.3% | 0.527 | 0.664 | +0.107 |
| 2022 | ATL | 396 | 42.7% | 0.530 | 0.656 | -0.007 |
| 2024 | DET | 412 | 48.3% | 0.551 | 0.658 | +0.129 |

Correlation between predictability and EPA per play: +0.072 across 128 team-seasons.
