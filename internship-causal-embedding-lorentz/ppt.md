# Follow-up Causal Embeddings — Experiments

Cosine similarity on L2-normalized embeddings.
**AUC** = ROC-AUC over (1 positive + N negatives) per anchor.
**P@1** = fraction of anchors where `sim(anchor, positive) > sim(anchor, every negative)`.

Datasets: `aep_causal`, `aep_followup`, `followupqg`, `multiwoz_v24`, `qrecc`, `workflow` (all under [`data_6/`](data_6/)). Fixed seed `0`.

---

## Master comparison — AUC

| Model | Fine-tune | Eval negatives | aep_causal | aep_followup | followupqg | multiwoz_v24 | qrecc | workflow |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MiniLM-L6-v2 | none | random (4)        | 0.9371 | 0.8435 | 0.9795 | 0.6706 | 0.8713 | 0.8994 |
| MiniLM-L6-v2 | none | BM25 hard (4)     | 0.5153 | 0.5517 | 0.9143 | 0.2386 | 0.4465 | 0.4982 |
| MiniLM-L6-v2 | none | reverse y→x (1)   | 0.5000* | 0.5000* | 0.5000* | 0.5000* | 0.5000* | 0.5000* |
| MiniLM-L6-v2 | none | full-corpus (N−1) | 0.9470 | 0.8881 | 0.9805 | 0.6606 | 0.8823 | 0.9054 |
| BERT-base    | none | random (4)        | 0.7294 | 0.6128 | 0.6350 | 0.5472 | 0.5762 | 0.7817 |
| BERT-base    | none | BM25 hard (4)     | 0.4792 | 0.3765 | 0.4803 | 0.3466 | 0.4869 | 0.5325 |
| BERT-base    | none | reverse y→x (1)   | 0.5000* | 0.5000* | 0.5000* | 0.5000* | 0.5000* | 0.5000* |
| BERT-base    | none | full-corpus (N−1) | 0.7699 | 0.6543 | 0.7166 | 0.5636 | 0.7242 | 0.8217 |
| **BERT 2-tower** | **InfoNCE + 4 sem hard** | random (4)        | **0.9712** | — | **0.9827** | **0.9526** | **0.9227** | **0.9662** |
| **BERT 2-tower** | **InfoNCE + 4 sem hard** | semantic hard (4) | **0.7157** | — | **0.8370** | **0.6223** | **0.6365** | **0.6883** |

## Master comparison — P@1

| Model | Fine-tune | Eval negatives | aep_causal | aep_followup | followupqg | multiwoz_v24 | qrecc | workflow |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MiniLM-L6-v2 | none | random (4)        | 0.8754 | 0.7498 | 0.9441 | 0.4053 | 0.7400 | 0.8046 |
| MiniLM-L6-v2 | none | BM25 hard (4)     | 0.2740 | 0.2598 | 0.7984 | 0.0877 | 0.1359 | 0.3017 |
| MiniLM-L6-v2 | none | reverse y→x (1)   | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* |
| MiniLM-L6-v2 | none | full-corpus (N−1) | 0.1566 | 0.0542 | 0.4331 | 0.0092 | 0.0234 | 0.0042 |
| BERT-base    | none | random (4)        | 0.5338 | 0.4143 | 0.4611 | 0.2782 | 0.4919 | 0.6273 |
| BERT-base    | none | BM25 hard (4)     | 0.2206 | 0.1315 | 0.2435 | 0.0791 | 0.1583 | 0.2636 |
| BERT-base    | none | reverse y→x (1)   | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* |
| BERT-base    | none | full-corpus (N−1) | 0.0285 | 0.0072 | 0.0419 | 0.0005 | 0.0066 | 0.0026 |
| **BERT 2-tower** | **InfoNCE + 4 sem hard** | random (4)        | **0.9128** | — | **0.9541** | **0.8694** | **0.8113** | **0.9104** |
| **BERT 2-tower** | **InfoNCE + 4 sem hard** | semantic hard (4) | **0.5178** | — | **0.7365** | **0.4110** | **0.4137** | **0.4763** |

### Random baselines
- 4 negatives → AUC 0.50, P@1 0.20.
- 1 negative (reverse) → AUC 0.50, P@1 0.50 if scores were independent (but symmetric encoders force ties → strict-P@1 = 0).
- Full corpus (N−1) → P@1 ≈ 1/N (so 1/562 ≈ 0.0018 on aep_causal).

---

# Per-experiment details

## Exp 1 — Inference, random negatives (4)

- **Models**: `sentence-transformers/all-MiniLM-L6-v2` (mean, 384-d); `google-bert/bert-base-uncased` (CLS, 768-d, raw MLM).
- **Fine-tune**: none.
- **Eval negatives**: 4 random other positives per anchor (no replacement).
- **Loss / training**: n/a — pure inference.
- **Source**: [`evaluation_6/`](evaluation_6/README.md).

### MiniLM-L6-v2

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.9371 | 0.8754 |
| aep_followup  |   1,255 | 0.8435 | 0.7498 |
| followupqg    |     501 | 0.9795 | 0.9441 |
| multiwoz_v24  |   7,368 | 0.6706 | 0.4053 |
| qrecc         |  13,633 | 0.8713 | 0.7400 |
| workflow      | 260,487 | 0.8994 | 0.8046 |

### BERT-base (raw MLM)

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.7294 | 0.5338 |
| aep_followup  |   1,255 | 0.6128 | 0.4143 |
| followupqg    |     501 | 0.6350 | 0.4611 |
| multiwoz_v24  |   7,368 | 0.5472 | 0.2782 |
| qrecc         |  13,633 | 0.5762 | 0.4919 |
| workflow      | 260,487 | 0.7817 | 0.6273 |

---

## Exp 2 — Inference, BM25 hard negatives (4)

- **Models**: same as Exp 1.
- **Fine-tune**: none.
- **Eval negatives**: top-4 BM25-scored positives per anchor (lexically similar but wrong continuations). For `workflow`: BM25 index and queries capped at 10 000 rows.
- **Loss / training**: n/a.
- **Source**: [`evaluation_6/bm25negative/`](evaluation_6/bm25negative/README.md).

### MiniLM-L6-v2

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.5153 | 0.2740 |
| aep_followup  |   1,255 | 0.5517 | 0.2598 |
| followupqg    |     501 | 0.9143 | 0.7984 |
| multiwoz_v24  |   7,368 | 0.2386 | 0.0877 |
| qrecc         |   5,204 | 0.4465 | 0.1359 |
| workflow      | 260,487 | 0.4982 | 0.3017 |

### BERT-base (raw MLM)

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.4792 | 0.2206 |
| aep_followup  |   1,255 | 0.3765 | 0.1315 |
| followupqg    |     501 | 0.4803 | 0.2435 |
| multiwoz_v24  |   7,368 | 0.3466 | 0.0791 |
| qrecc         |   5,204 | 0.4869 | 0.1583 |
| workflow      | 260,487 | 0.5325 | 0.2636 |

---

## Exp 3 — Inference, reverse-direction negative (1)

- **Models**: same as Exp 1.
- **Fine-tune**: none.
- **Eval negatives**: 1 per anchor — the reverse pair `(positive, anchor)`. Tests direction sensitivity.
- **Loss / training**: n/a.
- **Note**: symmetric encoders ⇒ `sim(a, p) ≡ sim(p, a)` ⇒ AUC = 0.50, strict-P@1 = 0.
- **Source**: [`evaluation_6/reverse(y->x)/`](evaluation_6/reverse(y->x)/README.md).

### MiniLM-L6-v2

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.5000 | 0.0000 |
| aep_followup  |   1,255 | 0.5000 | 0.0000 |
| followupqg    |     501 | 0.5000 | 0.0000 |
| multiwoz_v24  |   7,368 | 0.5000 | 0.0000 |
| qrecc         |   5,204 | 0.5000 | 0.0000 |
| workflow      | 260,487 | 0.5000 | 0.0000 |

### BERT-base (raw MLM)

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.5000 | 0.0000 |
| aep_followup  |   1,255 | 0.5000 | 0.0000 |
| followupqg    |     501 | 0.5000 | 0.0000 |
| multiwoz_v24  |   7,368 | 0.5000 | 0.0000 |
| qrecc         |   5,204 | 0.5000 | 0.0000 |
| workflow      | 260,487 | 0.5000 | 0.0000 |

> Confirms both encoders are direction-agnostic out of the box. Untied two-tower training (Exp 5) is needed for any direction signal.

---

## Exp 4 — Inference, full-corpus negatives (N−1)

- **Models**: same as Exp 1.
- **Fine-tune**: none.
- **Eval negatives**: every other test positive (N−1 per anchor).
- **AUC** computed analytically from rank: `mean((N − rank_i) / (N − 1))`. **P@1** = fraction ranked strictly first out of N.
- **Loss / training**: n/a.
- **Source**: [`evaluation_6/evaluate_full.py`](evaluation_6/evaluate_full.py).

### MiniLM-L6-v2

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.9470 | 0.1566 |
| aep_followup  |   1,255 | 0.8881 | 0.0542 |
| followupqg    |     501 | 0.9805 | 0.4331 |
| multiwoz_v24  |   7,368 | 0.6606 | 0.0092 |
| qrecc         |  13,633 | 0.8823 | 0.0234 |
| workflow      | 260,487 | 0.9054 | 0.0042 |

### BERT-base (raw MLM)

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.7699 | 0.0285 |
| aep_followup  |   1,255 | 0.6543 | 0.0072 |
| followupqg    |     501 | 0.7166 | 0.0419 |
| multiwoz_v24  |   7,368 | 0.5636 | 0.0005 |
| qrecc         |  13,633 | 0.7242 | 0.0066 |
| workflow      | 260,487 | 0.8217 | 0.0026 |

---

## Exp 5 — Fine-tuned BERT, random eval negatives (4)

- **Architecture**: **untied two-tower** bi-encoder; both towers `google-bert/bert-base-uncased` (CLS, 768-d, L2-norm output). ~219 M trainable params total.
- **Training negatives**: 4 **semantic hard negatives** per anchor, mined offline via frozen `all-MiniLM-L6-v2` kNN over the train positives.
- **Loss**: `HardNegativeInfoNCELoss` — InfoNCE with `B` in-batch positives + `K=4` anchor-specific hard negatives in the softmax denominator. Learnable temperature `τ` (init 0.05).
- **Optimizer**: AdamW (encoder 2e-5, temperature 1e-3), OneCycleLR 10 % warmup, cosine anneal.
- **Train caps** (runtime budget): aep_causal full (5,487), followupqg full (2,790), multiwoz_v24 15 k of 56,719, qrecc 15 k of 52,481, workflow 20 k of 2,116,157.
- **Eval negatives**: 4 random other positives per test anchor — directly comparable to Exp 1 row.
- **Datasets**: 5 of 6 (no `aep_followup` — no train split).
- **Source**: [`finetune_eval/`](finetune_eval/README.md).

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.9712 | 0.9128 |
| followupqg    |     501 | 0.9827 | 0.9541 |
| multiwoz_v24  |   7,368 | 0.9526 | 0.8694 |
| qrecc         |   5,204 | 0.9227 | 0.8113 |
| workflow      | 260,487 | 0.9662 | 0.9104 |

---

## Exp 6 — Fine-tuned BERT, semantic hard eval negatives (4)

- **Architecture, training, loss, train caps**: identical to Exp 5 — **no extra training**.
- **Eval negatives**: 4 semantic hard negatives per test anchor — for each `(anchor_i, positive_i)`, the top-4 nearest other positives of `positive_i` in `all-MiniLM-L6-v2` space.
- **Datasets**: same 5 as Exp 5.
- **Source**: [`finetune_eval/evaluate_finetune_hardneg.py`](finetune_eval/evaluate_finetune_hardneg.py).

| dataset       |       N |    AUC |    P@1 |
|---------------|--------:|-------:|-------:|
| aep_causal    |     562 | 0.7157 | 0.5178 |
| followupqg    |     501 | 0.8370 | 0.7365 |
| multiwoz_v24  |   7,368 | 0.6223 | 0.4110 |
| qrecc         |   5,204 | 0.6365 | 0.4137 |
| workflow      | 260,487 | 0.6883 | 0.4763 |

---

# Delta tables

## Δ — fine-tuned BERT (Exp 5) vs raw BERT (Exp 1B), random eval negs

`aep_followup` excluded (no train split).

| dataset       | Raw BERT AUC | **FT BERT AUC** | ΔAUC | Raw BERT P@1 | **FT BERT P@1** | ΔP@1 |
|---------------|-------------:|----------------:|-----:|-------------:|----------------:|-----:|
| aep_causal    | 0.7294 | **0.9712** | +0.242 | 0.5338 | **0.9128** | +0.379 |
| followupqg    | 0.6350 | **0.9827** | +0.348 | 0.4611 | **0.9541** | +0.493 |
| multiwoz_v24  | 0.5472 | **0.9526** | +0.405 | 0.2782 | **0.8694** | +0.591 |
| qrecc         | 0.5762 | **0.9227** | +0.347 | 0.4919 | **0.8113** | +0.319 |
| workflow      | 0.7817 | **0.9662** | +0.185 | 0.6273 | **0.9104** | +0.283 |

Biggest lift on `multiwoz_v24` (+0.41 AUC / +0.59 P@1) where raw BERT was near chance. Smallest on `workflow` (+0.19 / +0.28) where raw BERT was already strong.

## Δ — semantic hard eval (Exp 6) vs random eval (Exp 5), fine-tuned BERT

| dataset       | Random AUC | **Hard AUC** | ΔAUC   | Random P@1 | **Hard P@1** | ΔP@1   |
|---------------|-----------:|-------------:|-------:|-----------:|-------------:|-------:|
| aep_causal    | 0.9712 | **0.7157** | −0.255 | 0.9128 | **0.5178** | −0.395 |
| followupqg    | 0.9827 | **0.8370** | −0.146 | 0.9541 | **0.7365** | −0.218 |
| multiwoz_v24  | 0.9526 | **0.6223** | −0.330 | 0.8694 | **0.4110** | −0.458 |
| qrecc         | 0.9227 | **0.6365** | −0.286 | 0.8113 | **0.4137** | −0.398 |
| workflow      | 0.9662 | **0.6883** | −0.279 | 0.9104 | **0.4763** | −0.434 |

Drop of 0.15-0.33 AUC under semantic-hard eval — distractors mined from `all-MiniLM-L6-v2` kNN are genuinely harder than random.

---

## Key takeaways

- **Fine-tuned BERT beats every inference baseline** on every shared dataset on both AUC and P@1 (random eval). Biggest gap on `multiwoz_v24`.
- **BM25 negatives expose lexical sensitivity** — MiniLM AUC drops 0.07-0.43 vs random.
- **Direction is not captured by symmetric encoders** — Exp 3 forces 0.5 AUC / 0.0 P@1. Untied two-tower training (Exp 5) is required for `s(A → B) ≠ s(B → A)`.
- **Semantic hard negatives at eval (Exp 6) drop fine-tuned AUC by 0.15-0.33** — still beats chance everywhere but the strictest test we ran.

---

## Source pointers

| Exp | Code | Results | README |
|---|---|---|---|
| 1 | [`evaluation_6/evaluate_inference.py`](evaluation_6/evaluate_inference.py) | [`results.json`](evaluation_6/results.json) | [`evaluation_6/README.md`](evaluation_6/README.md) |
| 2 | [`evaluation_6/bm25negative/evaluate_bm25.py`](evaluation_6/bm25negative/evaluate_bm25.py) | [`results_bm25.json`](evaluation_6/bm25negative/results_bm25.json) | [`evaluation_6/bm25negative/README.md`](evaluation_6/bm25negative/README.md) |
| 3 | [`evaluation_6/reverse(y->x)/evaluate_reverse.py`](evaluation_6/reverse(y->x)/evaluate_reverse.py) | [`results.json`](evaluation_6/reverse(y->x)/results.json) | [`evaluation_6/reverse(y->x)/README.md`](evaluation_6/reverse(y->x)/README.md) |
| 4 | [`evaluation_6/evaluate_full.py`](evaluation_6/evaluate_full.py) | [`results_full.json`](evaluation_6/results_full.json) | — |
| 5 | [`finetune_eval/train_finetune.py`](finetune_eval/train_finetune.py) · [`finetune_eval/evaluate_finetune.py`](finetune_eval/evaluate_finetune.py) | [`eval_summary.json`](finetune_eval/results/eval_summary.json) | [`finetune_eval/README.md`](finetune_eval/README.md) |
| 6 | [`finetune_eval/evaluate_finetune_hardneg.py`](finetune_eval/evaluate_finetune_hardneg.py) | [`eval_hardneg_summary.json`](finetune_eval/results/eval_hardneg_summary.json) | — |
