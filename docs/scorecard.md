# Evaluation scorecard

**PASS**: 21 gates passed, 0 failed, 0 skipped.

## Suites

| suite | status | time |
| --- | --- | --- |
| guard | ran | 0s |
| routing_keyword | ran | 0s |
| references | ran | 0s |
| models | ran | 123s |
| text_to_sql | ran | 578s |
| routing_llm | ran | 60s |
| retrieval | ran | 5s |

## Gates

| metric | value | bound | status | why |
| --- | --- | --- | --- | --- |
| guard.attacks_blocked | 1 | >= 1 | pass | every adversarial query in golden_guard.json must be rejected before it reaches DuckDB |
| guard.legitimate_allowed | 1 | >= 1 | pass | a guard that rejects real analytics questions pushes the repair loop into worse SQL |
| guard.row_caps_enforced | 1 | >= 1 | pass | no generated query may return more than the row cap |
| routing_keyword.accuracy | 0.95 | >= 0.9 | pass | the keyword router is the fallback when Ollama is down; measured 19/20 |
| routing_keyword.worst_tool_recall | 0.8 | >= 0.8 | pass | a router that collapses one tool onto another looks fine on accuracy alone |
| references.runnable | 1 | >= 1 | pass | every golden reference query must pass the guard and execute; a schema change breaks them first |
| models.wp_auc | 0.8634 | 0.83 to 0.95 | pass | measured 0.863 on 2021-2024; above 0.95 on held-out seasons means a leaked feature |
| models.wp_log_loss | 0.4553 | <= 0.48 | pass | measured 0.455; the full-split model matches nflfastR's vegas_wp |
| models.wp_ece | 0.0066 | <= 0.02 | pass | the fourth-down advisor compares probabilities directly, so they must be calibrated |
| models.wp_log_loss_gap_vs_vegas | 0.0005 | <= 0.015 | pass | stay within reach of nflfastR's own model scored on the same held-out plays |
| models.playcall_auc | 0.8153 | 0.78 to 0.92 | pass | measured 0.815; run/pass from pre-snap state tops out well below 0.92 without leakage |
| models.playcall_lift_over_majority | 0.123 | >= 0.08 | pass | the play-call model must beat always guessing pass by a real margin; measured +0.12 |
| models.advisor_agreement | 0.666 | >= 0.55 | pass | measured 0.67; far below that the advisor is disagreeing with coaches on obvious punts |
| models.advisor_go_rate | 0.4233 | 0.25 to 0.6 | pass | measured 0.42; outside this band the advisor has collapsed to always or never going for it |
| text_to_sql.runnable | 0.9333 | >= 0.85 | pass | measured 14/15 with qwen2.5-coder:7b; guarded SQL that fails to run is a wasted answer |
| text_to_sql.correct | 0.4667 | >= 0.4 | pass | execution accuracy on the golden set; measured 7/15 with qwen2.5-coder:7b |
| routing_llm.accuracy | 0.95 | >= 0.9 | pass | the LLM router must at least match the keyword fallback; measured 19/20 |
| routing_llm.worst_tool_recall | 0.8333 | >= 0.8 | pass | per-tool recall, as for the keyword router |
| retrieval.hit_at_1 | 0.9375 | >= 0.85 | pass | measured 0.94 hybrid hit@1 on the retrieval golden set |
| retrieval.mrr | 0.9688 | >= 0.9 | pass | measured 0.969 hybrid MRR |
| retrieval.hybrid_mrr_gain | 0.0312 | >= 0 | pass | hybrid must not lose to dense or lexical alone, or the fusion is not earning its cost |

## Reported, not gated

| metric | value |
| --- | --- |
| guard.attack_cases | 36 |
| references.nonempty | 1 |
| models.wp_brier | 0.1509 |
| models.playcall_accuracy | 0.7396 |
| models.coach_go_rate | 0.2082 |
| models.playcall_lift_over_xpass | 0.0289 |
| text_to_sql.repairs_per_question | 0.5333 |
| retrieval.recall_at_k | 1 |
| retrieval.dense_mrr | 0.9375 |
| retrieval.lexical_mrr | 0.8958 |

## Findings

- **routing_keyword**: `How did the Colts blow their lead against the Vikings in 2022?` expected narrative, got stats
- **models**: win probability: train 399,601 rows (2009-2018), valid 79,345 rows (2019-2020), test 166,371 rows (2021-2024)
- **models**: play call: train 349,283 rows (2009-2018), valid 70,316 rows (2019-2020), test 146,918 rows (2021-2024)
- **models**: fourth-down audit replayed 4,000 held-out plays
- **text_to_sql**: `team-season-pass-rate`: expected [(0.6780028943560058,)], got [(0.6777186478449989,)]
- **text_to_sql**: `third-and-long-epa`: expected [(-0.03741678890747643,)], got [(-0.01771802262449919,)]
- **text_to_sql**: `red-zone-td-rate`: Binder Error: Referenced column "field_zone" not found in FROM clause!
- **text_to_sql**: `coldest-game`: expected [('2015_18_SEA_MIN', -6)], got [(-6, datetime.date(2009, 9, 10))]
- **text_to_sql**: `qb-epa-leader`: expected [('A.Rodgers',)], got []
- **text_to_sql**: `fourth-down-attempts-trend`: expected [(2019, 646), (2020, 726), (2021, 855)], got [(2023, 4490), (2021, 4254), (2022, 4300)]
- **text_to_sql**: `kneel-exclusion`: expected [(506,)], got [(14735,)]
- **text_to_sql**: `two-minute-pass-rate`: expected [(False, 0.5948739666217329), (True, 0.7909684525214381)], got [(0.7909684525214381, None)]
- **routing_llm**: `Which team had the highest EPA per play in 2023?` expected stats, got tendency
- **retrieval**: `super-bowl-49-final-drive`: rank 2 (both)
