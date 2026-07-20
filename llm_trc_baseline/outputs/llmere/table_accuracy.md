## Table A — accuracy + inference cost

| system | params | rationale | inference | micro_f1 | macro_f1 | src_procedural | src_aep | src_ajo |
|---|---|---|---|---|---|---|---|---|
| LoRA label-only (Qwen-7B, matched hp) | 7B | no | autoregressive (short) | 0.8384 | 0.8383 | 0.8499 | 0.8798 | 0.7126 |
| Reasoning-CE (OURS) | 435M | train-time | 1 fwd (encoder) | 0.8098 |  |  |  |  |
| DeBERTa-CE (encoder) | 435M | no | 1 fwd (encoder) | 0.7834 | 0.7833 | 0.7854 | 0.8721 | 0.5911 |
| LoRA rationale (LLMERE, 1e-4/r64) | 7B | train+infer | autoregressive (long) | 0.7703 | 0.7702 | 0.8396 | 0.7164 | 0.6478 |
| LoRA rationale (LLMERE, 2e-4/r64) | 7B | train+infer | autoregressive (long) | 0.7682 | 0.7682 | 0.8324 | 0.7248 | 0.6410 |
| LoRA rationale (LLMERE, r32/1e-4) | 7B | train+infer | autoregressive (long) | 0.7617 | 0.7617 | 0.8328 | 0.7106 | 0.6275 |
| ICL-CoT (Qwen-7B, k=2, no FT) | 7B | infer | autoregressive (long) | 0.5863 | 0.5816 | 0.6489 | 0.5304 | 0.4912 |
| ICL-CoT (Qwen-7B, k=0, no FT) | 7B | infer | autoregressive (long) | 0.5553 | 0.4948 | 0.5904 | 0.5362 | 0.4764 |
