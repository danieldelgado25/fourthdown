# Routing evaluation

Run on the CPU: the `num_gpu 0` variant of the same `qwen2.5-coder:7b` weights.

- **keyword**: 29/29 (100%)
- **llm (qwen2.5-coder:7b)**: 28/29 (97%)
- **hybrid (qwen2.5-coder:7b)**: 28/29 (97%)

## By expected tool

| tool | keyword | llm (qwen2.5-coder:7b) | hybrid (qwen2.5-coder:7b) |
| --- | --- | --- | --- |
| advisor | 4/4 | 4/4 | 4/4 |
| narrative | 6/6 | 6/6 | 6/6 |
| stats | 14/14 | 13/14 | 13/14 |
| tendency | 5/5 | 5/5 | 5/5 |

## Misroutes

| router | question | expected | chosen | reason |
| --- | --- | --- | --- | --- |
| llm (qwen2.5-coder:7b) | Which team had the highest EPA per play in 2023? | stats | tendency | model chose tendency |
| hybrid (qwen2.5-coder:7b) | Which team had the highest EPA per play in 2023? | stats | tendency | model chose tendency |
