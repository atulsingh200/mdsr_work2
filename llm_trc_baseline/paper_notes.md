# Paper notes — arXiv 2410.10476v2 (Roccabruna et al.)
*"Will LLMs Replace the Encoder-Only Models in Temporal Relation Classification?"*

Extracted methodology we reproduce on our `final_data` binary directional TRC task.
Repo: https://github.com/BrownFortress/LLMs-TRC (pickle Document/Event/Relation format — we adapt
the *method*, not the code, because our data is independent step-pairs, not spans in one sentence).

## Task
Given two events, predict the temporal relation. Paper is multi-class (BEFORE/AFTER/EQUAL/…); **ours
is binary**: `label=1` → text_1 BEFORE text_2, `label=0` → NOT_BEFORE. Metric: **micro-F1** (paper);
we also report accuracy + per-class + per-source F1. Balanced classes → acc ≈ micro-F1.

## Prompt schemas (paper Table 1 & Table 9)
Events wrapped with `[event1]…[/event1]` and `[event2]…[/event2]`. Three prompt types:

- **P (example-label)**: `Given the context: … [event1]e1[/event1] … [event2]e2[/event2] … -> LABEL`
  where LABEL ∈ relation set. Binary → {BEFORE, NOT_BEFORE}.
- **QA1 (single question)**: append `Answer the question: Does [event1] happen before [event2]? YES/NO`.
  YES → BEFORE (1), NO → NOT_BEFORE (0). This is the natural binary form.
- **QA2 (sequential)**: ask all class-questions in sequence, feeding earlier answers back as context;
  zeroes contradictory answers. Binary: before? then after? → resolve to one label.

Question phrasing (Table 1): "Does e1 happen before e2?", "Does e1 happen after e2?",
"Does e1 happen at the same time as e2?".

## Few-shot ICL (paper §5.2, Appendix A.1)
- 1 example **per class** (binary → 2 shots; we also try k=4). Exemplars drawn from `train`.
- **5 frozen sampled sets**; report mean ± std over the 5.
- Question order fixed across models/datasets: after, before, equal (we use before, after).
- Gibberish / contradictory / unparseable output → counted wrong (paper maps to `vague`).

## LoRA fine-tuning (paper §4.2, Appendix A.1) — LLM upper bound
- **Zero-shot** prompt format (P and QA1), model learns from back-prop not few-shots.
- LoRA **rank r = 32, alpha = 64**, `init_lora_weights="gaussian"`.
- Optimizer **AdamW, lr 1e-4**; **batch size 8**; linear scheduler, **warmup = 10% of steps**.
- Causal-LM loss on the completion (label / YES-NO) tokens only.

## Encoder recipe (paper §4.1, Appendix A.1) — for reference
RoBERTa: event embeddings = max-pool of sub-tokens of each event span, concatenated → FF → softmax.
Two AdamW optimizers (encoder lr 1e-5, head lr 1e-4), linear warmup 10%, batch size 8.
*Our encoder is different*: an already-trained **DeBERTa-v3-large CrossEncoder2x** (joint
`[CLS] text_1 [SEP] text_2 [SEP]`, one BCE logit). We only **evaluate** it on `final_data` test.

## Paper headline result (Table 2, micro-F1)
Best ICL LLM (Llama2-70B QA2): MATRES 65.3 / TIMELINE 62.5 / TB-Dense 31.4.
Best LoRA LLM (Llama2-13B QA2): MATRES 84.3 / TIMELINE 41.5 (7B QA1 76.9) / TB-Dense 49.3.
**RoBERTa encoder: MATRES 87.6 / TIMELINE 87.9 / TB-Dense 83.1** — beats every LLM in every setting.
Explanation: autoregressive LLMs attend mostly to the last tokens; the encoder uses the whole context.

## Our models to baseline (modern open decoder-only)
Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct, + one large
(Qwen2.5-72B-Instruct or Llama-3.3-70B-Instruct).
