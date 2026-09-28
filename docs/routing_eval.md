# Routing evaluation

- **keyword**: 19/20 (95%)
- **llm (qwen2.5-coder:7b)**: 19/20 (95%)

## By expected tool

| tool | keyword | llm (qwen2.5-coder:7b) |
| --- | --- | --- |
| advisor | 4/4 | 4/4 |
| narrative | 4/5 | 5/5 |
| stats | 6/6 | 5/6 |
| tendency | 5/5 | 5/5 |

## Misroutes

| router | question | expected | chosen | reason |
| --- | --- | --- | --- | --- |
| keyword | How did the Colts blow their lead against the Vikings in 2022? | narrative | stats | no narrative or decision cue; treated as a statistics question |
| llm (qwen2.5-coder:7b) | Which team had the highest EPA per play in 2023? | stats | tendency | model chose tendency |
