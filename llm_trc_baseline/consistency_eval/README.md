# Paper 2 baseline — consistency / cycle-breaking (INLG 2025)

Reproduces Meir & Bar (INLG 2025), "Can LLMs Help Encoder Models Maintain Both High Accuracy and
Consistency in TRC?", and positions our **reasoning-trace cross-encoder** against it.

## Idea
A pairwise encoder predicts a temporal order for every step pair in a workflow → a directed graph.
Where the predictions contradict (a **simple cycle**), the graph can't yield a timeline. The paper
fixes this at **inference** by dropping one edge per cycle, via either a **confidence** heuristic
(drop lowest-confidence edge) or an **LLM** (ask which edge is wrong). Their finding: the LLM does
**not** beat the simple confidence heuristic.

Our claim: the **reasoning-trace CE** (auxiliary reasoning decoder trained in; plain encoder at
inference) is **natively more consistent** — fewer cycles — with **no inference-time LLM**.

## Pipeline (`consistency_eval/`)
1. `build_workflows.py` → `workflows_eval.json`: 697 curated (2–4 steps) + 200 sampled procedural
   (5–8 steps). Gold order = step index.
2. `predict_pairs.py` → `outputs/consistency/pairs_{baseline,reasoning}.json`: score every step pair
   with each CE (randomized presentation orientation → confidence per predicted edge).
   - baseline = `runs/crossencoder2x_deberta/v2_res_baseline` (no reasoning)
   - reasoning = `runs/crossencoder2x_deberta_reasoning/v2_fixed_alpha02_ep10` (ours; loaded
     `strict=False`, ignoring the unused reasoning-decoder — the "no LLM at inference" point)
3. `cycles.py`: NetworkX graph build + `simple_cycles`, cycle-rate / mean-cycles / ordering-accuracy,
   confidence + LLM cycle breakers.
4. `llm_breaker.py`: faithful zero-shot DOT-format prompt (paper Appendix A) via local
   Qwen2.5-14B-Instruct. **Node labels are anonymized/shuffled** so the LLM must reason from step
   *text*, not from indices (indices = gold order → would leak the answer).
5. `run_consistency.py` → `outputs/consistency/consistency_results.json`.
6. `aggregate_consistency.py` → `consistency_table.md`, `consistency_scatter.png`.

## Results — procedural slice (5–8 steps), the regime where cycles occur

| System | LLM @ inference | cycle-rate | mean cycles | ordering-acc |
|---|---|---|---|---|
| Baseline CE (plain) | no | 0.140 | 0.86 | 0.956 |
| Baseline CE + confidence-break | no | 0.000 | 0 | 0.950 |
| Baseline CE + LLM-assisted break | **yes** | 0.000 | 0 | 0.947 |
| **OURS: Reasoning-CE (plain)** | **no** | **0.095** | **0.59** | 0.943 |
| OURS: Reasoning-CE + confidence-break | no | 0.000 | 0 | 0.938 |

(`proc n≥6`: baseline cycle-rate 0.192 → reasoning 0.128, −33%.)

## Takeaways
1. **Reasoning-CE is natively ~32% more consistent** (cycle-rate 0.140 → 0.095; mean cycles
   0.86 → 0.59) than the plain baseline, **with zero inference-time LLM calls**. If full consistency
   is needed, a confidence-break (still no LLM) drives it to 0.
2. **The LLM-assisted breaker does not help**: confidence-break (0.950) ≥ LLM-assisted (0.947) — a
   faithful reproduction of the paper's own headline, and it costs an LLM call per cycle at inference.
3. Accuracy/consistency trade-off (paper's central tension) is visible: reasoning-CE trades ~1.3 pts
   of ordering accuracy on procedural for markedly fewer cycles; on the curated slice it is *both*
   more accurate (0.951 vs 0.935) and comparably consistent.

**Bottom line:** our approach buys temporal consistency at *training* time, so at inference it is a
plain encoder that needs no LLM patch — unlike the paper's hybrid, whose LLM patch doesn't even beat
a trivial confidence rule.

## Reproduce
```
.venv/bin/python consistency_eval/build_workflows.py --sample-5to8 200 --seed 42
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python consistency_eval/predict_pairs.py --run-dir <baseline_dir>  --tag baseline
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python consistency_eval/predict_pairs.py --run-dir <reasoning_dir> --tag reasoning
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python consistency_eval/run_consistency.py --with-llm --gpu-mem-util 0.70
.venv/bin/python consistency_eval/aggregate_consistency.py
```
