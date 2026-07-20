### slice: procedural

| system | LLM@inf | cycle_rate | mean_cyc | ordering_acc | edges_removed | llm_calls |
|---|---|---|---|---|---|---|
| Baseline CE (plain) | no | 0.140 | 0.860 | 0.9561 | 0 | 0 |
| Baseline CE + confidence-break | no | 0.000 | 0.000 | 0.9498 | 44 | 44 |
| Baseline CE + LLM-assisted break | yes | 0.000 | 0.000 | 0.9471 | 49 | 49 |
| OURS: Reasoning-CE (plain) | no | 0.095 | 0.590 | 0.9432 | 0 | 0 |
| OURS: Reasoning-CE + confidence-break | no | 0.000 | 0.000 | 0.9380 | 38 | 38 |

### slice: proc_n>=6

| system | LLM@inf | cycle_rate | mean_cyc | ordering_acc | edges_removed | llm_calls |
|---|---|---|---|---|---|---|
| Baseline CE (plain) | no | 0.192 | 1.288 | 0.9538 | 0 | 0 |
| Baseline CE + confidence-break | no | 0.000 | 0.000 | 0.9472 | 38 | 38 |
| Baseline CE + LLM-assisted break | yes | 0.000 | 0.000 | 0.9429 | 43 | 43 |
| OURS: Reasoning-CE (plain) | no | 0.128 | 0.888 | 0.9390 | 0 | 0 |
| OURS: Reasoning-CE + confidence-break | no | 0.000 | 0.000 | 0.9324 | 34 | 34 |

### slice: all

| system | LLM@inf | cycle_rate | mean_cyc | ordering_acc | edges_removed | llm_calls |
|---|---|---|---|---|---|---|
| Baseline CE (plain) | no | 0.054 | 0.221 | 0.9474 | 0 | 0 |
| Baseline CE + confidence-break | no | 0.000 | 0.000 | 0.9425 | 66 | 66 |
| Baseline CE + LLM-assisted break | yes | 0.000 | 0.000 | 0.9399 | 73 | 73 |
| OURS: Reasoning-CE (plain) | no | 0.050 | 0.172 | 0.9465 | 0 | 0 |
| OURS: Reasoning-CE + confidence-break | no | 0.000 | 0.000 | 0.9418 | 66 | 66 |

### slice: curated

| system | LLM@inf | cycle_rate | mean_cyc | ordering_acc | edges_removed | llm_calls |
|---|---|---|---|---|---|---|
| Baseline CE (plain) | no | 0.029 | 0.037 | 0.9352 | 0 | 0 |
| Baseline CE + confidence-break | no | 0.000 | 0.000 | 0.9322 | 22 | 22 |
| Baseline CE + LLM-assisted break | yes | 0.000 | 0.000 | 0.9297 | 24 | 24 |
| OURS: Reasoning-CE (plain) | no | 0.037 | 0.052 | 0.9513 | 0 | 0 |
| OURS: Reasoning-CE + confidence-break | no | 0.000 | 0.000 | 0.9471 | 28 | 28 |
