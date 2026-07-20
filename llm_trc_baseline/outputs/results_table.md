| model | method | setting | micro_f1 | macro_f1 | src_procedural | src_aep | src_ajo |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | LoRA | QA1 ft | 0.8384 | 0.8383 | 0.8499 | 0.8798 | 0.7126 |
| deberta-v3-large_crossencoder2x | encoder | - | 0.7834 | 0.7833 | 0.7854 | 0.8721 | 0.5911 |
| Qwen/Qwen2.5-14B-Instruct | ICL | QA1 k2 | 0.6566 ±0.007 |  |  |  |  |
| Qwen/Qwen2.5-14B-Instruct | ICL | QA2 k2 | 0.6390 ±0.006 |  |  |  |  |
| Qwen/Qwen2.5-14B-Instruct | ICL | QA1 k0 | 0.6353 | 0.6264 | 0.7667 | 0.5019 | 0.4683 |
| Qwen/Qwen2.5-7B-Instruct | ICL | QA2 k2 | 0.6315 ±0.005 |  |  |  |  |
| Qwen/Qwen2.5-7B-Instruct | ICL | QA1 k2 | 0.6286 ±0.009 |  |  |  |  |
| Qwen/Qwen2.5-14B-Instruct | ICL | P k2 | 0.6282 ±0.006 |  |  |  |  |
| Qwen/Qwen2.5-14B-Instruct | ICL | P k0 | 0.6276 | 0.6216 | 0.7564 | 0.5000 | 0.4575 |
| Qwen/Qwen2.5-14B-Instruct | ICL | QA2 k0 | 0.6211 | 0.5913 | 0.7058 | 0.5426 | 0.4980 |
| Qwen/Qwen2.5-7B-Instruct | ICL | QA1 k0 | 0.6163 | 0.6148 | 0.7126 | 0.5265 | 0.4777 |
| Qwen/Qwen2.5-7B-Instruct | ICL | P k2 | 0.5833 ±0.012 |  |  |  |  |
| Qwen/Qwen2.5-7B-Instruct | ICL | P k0 | 0.5786 | 0.5618 | 0.6298 | 0.5310 | 0.5047 |
| Qwen/Qwen2.5-7B-Instruct | ICL | QA2 k0 | 0.5749 | 0.4890 | 0.6254 | 0.5297 | 0.4980 |
| microsoft/Phi-3.5-mini-instruct | ICL | QA1 k2 | 0.5732 ±0.011 |  |  |  |  |
| microsoft/Phi-3.5-mini-instruct | ICL | QA1 k0 | 0.5643 | 0.5636 | 0.6158 | 0.5200 | 0.4818 |
| microsoft/Phi-3.5-mini-instruct | ICL | QA2 k0 | 0.5632 | 0.5414 | 0.6274 | 0.4942 | 0.4899 |
| microsoft/Phi-3.5-mini-instruct | ICL | QA2 k2 | 0.5624 ±0.010 |  |  |  |  |
| microsoft/Phi-3.5-mini-instruct | ICL | P k2 | 0.5480 ±0.018 |  |  |  |  |
| microsoft/Phi-3.5-mini-instruct | ICL | P k0 | 0.5230 | 0.4268 | 0.5259 | 0.5297 | 0.4993 |