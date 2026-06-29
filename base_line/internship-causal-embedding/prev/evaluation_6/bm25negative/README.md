# BM25 Hard-Negative Evaluation

Evaluates the same off-the-shelf encoders and datasets as
[`evaluation_6/`](../evaluation_6/) but replaces **random negatives** with
**BM25-mined hard negatives** for the AUC / Precision@1 pair-level metrics.

Retrieval metrics (MRR, Recall@K) continue to use the same random candidate
pool (pool = 1000) so they are directly comparable across evaluations.

---

## What changes vs `evaluation_6`

| Aspect | evaluation_6 | bm25negative |
|--------|-------------|--------------|
| Negative source | Random positives from the dataset | Top-k BM25-scored positives per anchor |
| Negative count | 4 random | 4 BM25 hard |
| Retrieval metrics | Random pool (1000) | Random pool (1000) — identical |
| BM25 index size | n/a | Up to 10 000 positives |
| BM25 query limit | n/a | Up to 10 000 anchors per dataset |

BM25 mines lexically similar positives as negatives — the anchor text is the
query, positives matching the anchor's tokens are returned. This is a
strictly harder discrimination task than random: the model must distinguish
the true continuation from lexically related but wrong texts.

---

## Models

| Model | HuggingFace ID | Pooling | Dim |
|-------|----------------|---------|-----|
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | mean | 384 |
| bert-base-uncased | `google-bert/bert-base-uncased` | CLS | 768 |
| bge-m3 | `BAAI/bge-m3` | CLS | 1024 |
| e5-base-v2 | `intfloat/e5-base-v2` | mean | 768 |

None of the models are fine-tuned — off-the-shelf weights only. E5 inputs are
prefixed with `"query: "` / `"passage: "` (anchor/positive); the other three
models receive the raw text.

---

## Datasets

All from `data_6/` test splits.

| Dataset | N (test) | BM25 queries | BM25 corpus |
|---------|--------:|-------------|------------|
| aep_causal | 562 | 562 (all) | 562 (all) |
| aep_followup | 1 255 | 1 255 (all) | 1 255 (all) |
| followupqg | 501 | 501 (all) | 501 (all) |
| multiwoz_v24 | 7 368 | 7 368 (all) | 7 368 (all) |
| qrecc | 5 204 | 5 204 (all) | 5 204 (all) |
| workflow | 260 487 | 10 000 (sampled) | 10 000 (sampled) |

> **Note:** The `qrecc` test split currently has 5 204 pairs (the dataset was
> updated after the original `evaluation_6` run which used 13 633 pairs).
> Cross-evaluation comparisons for `qrecc` are therefore not apples-to-apples.
> For all other datasets the pair counts are identical.

---

## How to Run

```bash
# Full evaluation (both models × 6 datasets, ~60 min)
.venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py

# Single model / subset of datasets
.venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py \
    --models all-MiniLM-L6-v2 --datasets aep_causal followupqg

# Smaller BM25 index / query limit for faster runs
.venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py \
    --bm25-corpus-size 5000 --max-bm25-queries 5000

# Custom output path
.venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py --out evaluation_6/bm25negative/my_results.json
```

Prerequisites: `rank-bm25` must be installed.
```bash
.venv/bin/pip install '.[hard-negatives]'
```

Default settings: `--n-negatives 4`, `--bm25-corpus-size 10000`,
`--max-bm25-queries 10000`, `--candidate-pool-size 1000`,
`--k 1 3 5 10`, `--seed 0`, `--batch-size 128`.

---

## Results

### all-MiniLM-L6-v2 &nbsp;·&nbsp; `sentence-transformers/all-MiniLM-L6-v2` (mean pool, 384-d)

#### Pair-level — BM25 hard negatives (4 per anchor)

| dataset | N | BM25 AUC | BM25 P@1 |
|---------|--:|--------:|--------:|
| aep_causal | 562 | 0.5153 | 0.2740 |
| aep_followup | 1 255 | 0.5517 | 0.2598 |
| followupqg | 501 | 0.9143 | 0.7984 |
| multiwoz_v24 | 7 368 | 0.2386 | 0.0877 |
| qrecc | 5 204 | 0.4465 | 0.1359 |
| workflow | 260 487 | 0.4982 | 0.3017 |

#### Retrieval — random candidate pool (pool = 1 000)

| dataset | N | MRR | R@1 | R@3 | R@5 | R@10 | Mean rank | Median rank |
|---------|--:|----:|----:|----:|----:|-----:|----------:|------------:|
| aep_causal | 562 | 0.2943 | 0.1477 | 0.3452 | 0.4573 | 0.5943 | 30.74 | 7.0 |
| aep_followup | 1 255 | 0.1853 | 0.1139 | 0.1936 | 0.2510 | 0.3251 | 112.45 | 30.0 |
| followupqg | 501 | 0.6340 | 0.4631 | 0.7705 | 0.8303 | 0.8822 | 10.70 | 2.0 |
| multiwoz_v24 | 7 368 | 0.0888 | 0.0543 | 0.0909 | 0.1124 | 0.1435 | 282.80 | 174.0 |
| qrecc | 5 204 | 0.2350 | 0.1585 | 0.2602 | 0.3088 | 0.3711 | 133.0 | 31.0 |
| workflow | 260 487 | 0.3987 | 0.3109 | 0.4369 | 0.4936 | 0.5672 | 95.50 | 6.0 |

---

### bert-base-uncased &nbsp;·&nbsp; `google-bert/bert-base-uncased` (CLS pool, 768-d)

Raw MLM weights — not a sentence encoder, included as a lower-bound baseline.

#### Pair-level — BM25 hard negatives (4 per anchor)

| dataset | N | BM25 AUC | BM25 P@1 |
|---------|--:|--------:|--------:|
| aep_causal | 562 | 0.4792 | 0.2206 |
| aep_followup | 1 255 | 0.3765 | 0.1315 |
| followupqg | 501 | 0.4803 | 0.2435 |
| multiwoz_v24 | 7 368 | 0.3466 | 0.0791 |
| qrecc | 5 204 | 0.4869 | 0.1583 |
| workflow | 260 487 | 0.5325 | 0.2636 |

#### Retrieval — random candidate pool (pool = 1 000)

| dataset | N | MRR | R@1 | R@3 | R@5 | R@10 | Mean rank | Median rank |
|---------|--:|----:|----:|----:|----:|-----:|----------:|------------:|
| aep_causal | 562 | 0.0927 | 0.0302 | 0.1085 | 0.1406 | 0.1993 | 130.02 | 70.0 |
| aep_followup | 1 255 | 0.0402 | 0.0167 | 0.0343 | 0.0510 | 0.0749 | 345.95 | 243.0 |
| followupqg | 501 | 0.0934 | 0.0439 | 0.1078 | 0.1218 | 0.1537 | 142.64 | 86.0 |
| multiwoz_v24 | 7 368 | 0.0227 | 0.0071 | 0.0176 | 0.0250 | 0.0418 | 396.44 | 347.0 |
| qrecc | 5 204 | 0.0675 | 0.0348 | 0.0657 | 0.0886 | 0.1278 | 311.73 | 213.0 |
| workflow | 260 487 | 0.1660 | 0.1131 | 0.1733 | 0.2073 | 0.2625 | 179.07 | 70.0 |

---

### bge-m3 &nbsp;·&nbsp; `BAAI/bge-m3` (CLS pool, 1024-d)

#### Pair-level — BM25 hard negatives (4 per anchor)

| dataset | N | BM25 AUC | BM25 P@1 |
|---------|--:|--------:|--------:|
| aep_causal | 562 | 0.5189 | 0.2989 |
| aep_followup | 1 255 | 0.5020 | 0.2343 |
| followupqg | 501 | 0.8526 | 0.7086 |
| multiwoz_v24 | 7 368 | 0.1861 | 0.0616 |
| qrecc | 5 204 | 0.4584 | 0.1449 |
| workflow | 260 487 | 0.4438 | 0.2523 |

#### Retrieval — random candidate pool (pool = 1 000)

| dataset | N | MRR | R@1 | R@3 | R@5 | R@10 | Mean rank | Median rank |
|---------|--:|----:|----:|----:|----:|-----:|----------:|------------:|
| aep_causal | 562 | 0.3091 | 0.1548 | 0.3843 | 0.4822 | 0.6157 | 29.64 | 6.0 |
| aep_followup | 1 255 | 0.1829 | 0.1092 | 0.1873 | 0.2406 | 0.3243 | 109.51 | 29.0 |
| followupqg | 501 | 0.5484 | 0.3752 | 0.6906 | 0.7525 | 0.8184 | 17.11 | 2.0 |
| multiwoz_v24 | 7 368 | 0.0873 | 0.0516 | 0.0871 | 0.1126 | 0.1488 | 260.06 | 157.0 |
| qrecc | 5 204 | 0.2306 | 0.1508 | 0.2519 | 0.3065 | 0.3868 | 112.44 | 24.0 |
| workflow | 260 487 | 0.3632 | 0.2800 | 0.3965 | 0.4497 | 0.5221 | 103.58 | 9.0 |

---

### e5-base-v2 &nbsp;·&nbsp; `intfloat/e5-base-v2` (mean pool, 768-d, `"query: "` / `"passage: "` prefixes)

#### Pair-level — BM25 hard negatives (4 per anchor)

| dataset | N | BM25 AUC | BM25 P@1 |
|---------|--:|--------:|--------:|
| aep_causal | 562 | 0.5109 | 0.2669 |
| aep_followup | 1 255 | 0.5165 | 0.2271 |
| followupqg | 501 | 0.9425 | 0.8583 |
| multiwoz_v24 | 7 368 | 0.2449 | 0.0851 |
| qrecc | 5 204 | 0.4687 | 0.1391 |
| workflow | 260 487 | 0.4482 | 0.2509 |

#### Retrieval — random candidate pool (pool = 1 000)

| dataset | N | MRR | R@1 | R@3 | R@5 | R@10 | Mean rank | Median rank |
|---------|--:|----:|----:|----:|----:|-----:|----------:|------------:|
| aep_causal | 562 | 0.3051 | 0.1530 | 0.3843 | 0.4804 | 0.6050 | 30.64 | 6.0 |
| aep_followup | 1 255 | 0.1693 | 0.0932 | 0.1745 | 0.2382 | 0.3267 | 115.85 | 30.0 |
| followupqg | 501 | 0.6840 | 0.5090 | 0.8423 | 0.8822 | 0.9182 | 9.23 | 1.0 |
| multiwoz_v24 | 7 368 | 0.1047 | 0.0641 | 0.1095 | 0.1337 | 0.1726 | 249.34 | 131.0 |
| qrecc | 5 204 | 0.2561 | 0.1726 | 0.2911 | 0.3372 | 0.4051 | 96.06 | 24.0 |
| workflow | 260 487 | 0.3632 | 0.2757 | 0.3988 | 0.4564 | 0.5322 | 100.77 | 8.0 |

---

## BM25 vs Random Negatives — Comparison

BM25 hard negatives reveal how much a model relies on lexical shortcuts.
A large drop from random → BM25 indicates the model can be fooled by
lexically similar but semantically wrong continuations.

### all-MiniLM-L6-v2

| dataset | Random AUC | BM25 AUC | ΔAUC | Random P@1 | BM25 P@1 | ΔP@1 |
|---------|----------:|--------:|----:|----------:|--------:|----:|
| aep_causal | 0.9371 | 0.5153 | **−0.422** | 0.8754 | 0.2740 | **−0.601** |
| aep_followup | 0.8435 | 0.5517 | −0.292 | 0.7498 | 0.2598 | −0.490 |
| followupqg | 0.9795 | 0.9143 | −0.065 | 0.9441 | 0.7984 | −0.146 |
| multiwoz_v24 | 0.6706 | 0.2386 | **−0.432** | 0.4053 | 0.0877 | **−0.318** |
| qrecc ¹ | 0.8713 | 0.4465 | — | 0.7400 | 0.1359 | — |
| workflow | 0.8994 | 0.4982 | **−0.401** | 0.8046 | 0.3017 | **−0.503** |

### bert-base-uncased

| dataset | Random AUC | BM25 AUC | ΔAUC | Random P@1 | BM25 P@1 | ΔP@1 |
|---------|----------:|--------:|----:|----------:|--------:|----:|
| aep_causal | 0.7294 | 0.4792 | −0.250 | 0.5338 | 0.2206 | −0.313 |
| aep_followup | 0.6128 | 0.3765 | −0.236 | 0.4143 | 0.1315 | −0.283 |
| followupqg | 0.6350 | 0.4803 | −0.155 | 0.4611 | 0.2435 | −0.218 |
| multiwoz_v24 | 0.5472 | 0.3466 | −0.201 | 0.2782 | 0.0791 | −0.199 |
| qrecc ¹ | 0.5762 | 0.4869 | — | 0.4919 | 0.1583 | — |
| workflow | 0.7817 | 0.5325 | −0.249 | 0.6273 | 0.2636 | −0.364 |

### bge-m3

| dataset | Random AUC | BM25 AUC | ΔAUC | Random P@1 | BM25 P@1 | ΔP@1 |
|---------|----------:|--------:|----:|----------:|--------:|----:|
| aep_causal | 0.9383 | 0.5189 | **−0.419** | 0.8683 | 0.2989 | **−0.569** |
| aep_followup | 0.8712 | 0.5020 | −0.369 | 0.7530 | 0.2343 | −0.519 |
| followupqg | 0.9649 | 0.8526 | −0.112 | 0.9222 | 0.7086 | −0.214 |
| multiwoz_v24 | 0.7391 | 0.1861 | **−0.553** | 0.4958 | 0.0616 | −0.434 |
| qrecc | 0.8749 | 0.4584 | −0.417 | 0.7371 | 0.1449 | −0.592 |
| workflow | 0.8864 | 0.4438 | **−0.443** | 0.7863 | 0.2523 | **−0.534** |

### e5-base-v2

| dataset | Random AUC | BM25 AUC | ΔAUC | Random P@1 | BM25 P@1 | ΔP@1 |
|---------|----------:|--------:|----:|----------:|--------:|----:|
| aep_causal | 0.9289 | 0.5109 | **−0.418** | 0.8701 | 0.2669 | **−0.603** |
| aep_followup | 0.8644 | 0.5165 | −0.348 | 0.7514 | 0.2271 | −0.524 |
| followupqg | 0.9819 | 0.9425 | −0.039 | 0.9601 | 0.8583 | −0.102 |
| multiwoz_v24 | 0.7440 | 0.2449 | **−0.499** | 0.5233 | 0.0851 | −0.438 |
| qrecc | 0.8863 | 0.4687 | −0.418 | 0.7508 | 0.1391 | **−0.612** |
| workflow | 0.8914 | 0.4482 | **−0.443** | 0.7942 | 0.2509 | **−0.543** |

> ¹ `qrecc` dataset was updated between runs (5 204 pairs in BM25 eval vs 13 633
> in random eval); the Δ columns are omitted to avoid misleading comparisons.

### Key Observations

1. **BM25 negatives expose lexical sensitivity.** AUC drops by 0.07–0.43 for
   MiniLM and 0.16–0.25 for BERT when switching from random to BM25 negatives.
   Both models rely partly on lexical differences to distinguish positives.

2. **`followupqg` is the most robust.** The dataset's positives are full
   follow-up questions that are semantically distinct even when lexically
   similar, so BM25 struggles to mine genuinely hard negatives there
   (MiniLM: ΔAUC = −0.065, ΔP@1 = −0.146).

3. **`aep_causal`, `multiwoz_v24`, and `workflow` are the most sensitive.**
   MiniLM AUC drops by 0.40–0.43 on these three. BM25 can easily find
   positives that share many tokens with the anchor but are wrong continuations.

4. **BERT is less affected in absolute terms** but only because its random-
   negative baseline was already low (closer to chance). The relative drop
   is still substantial (0.16–0.25 AUC across datasets).

5. **Retrieval metrics (MRR, Recall@K) are the same** across both evaluations
   because they use an identical random candidate pool — any differences
   are due to re-encoding (GPU non-determinism is negligible).

---

## Full Results JSON

[`results_bm25.json`](results_bm25.json) stores all raw numbers including
per-metric details, timing, and configuration (all 12 model × dataset
combinations). The workflow results were produced in a separate run
([`results_workflow_only.json`](results_workflow_only.json)) after a bug fix
(the initial run had an indexing error for datasets larger than
`--bm25-corpus-size`) and then merged into `results_bm25.json`.

---

## Files

| File | Purpose |
|------|---------|
| [`evaluate_bm25.py`](evaluate_bm25.py) | Main evaluation runner. Loads each model once, mines BM25 negatives, computes AUC/P@1 and retrieval metrics. |
| [`results_bm25.json`](results_bm25.json) | Complete results — all 12 model × dataset combinations. |
| [`results_workflow_only.json`](results_workflow_only.json) | Workflow-only re-run with bug fix; source for the workflow rows in `results_bm25.json`. |

---

## Implementation Notes

### BM25 index and query limits
For large datasets the full BM25 index would be too slow to query.
`--bm25-corpus-size` (default 10 000) caps the number of positives indexed;
`--max-bm25-queries` (default 10 000) caps the number of anchors queried.
For datasets smaller than these limits, all pairs are used.
For `workflow` (260 487 pairs), both are capped at 10 000: the BM25 index
covers a random 10k-sample of positives and BM25 pair metrics are reported
over 10 000 randomly selected anchor-positive pairs.

### Bug note
The first run had an `IndexError` for the `workflow` dataset because
`neg_idx` contains global indices (0…N−1) but the AUC computation was
accidentally indexing into the smaller aligned embedding slice (size Q =
10 000). The fix separates the gold-positive embeddings (aligned, size Q)
from the full negative-pool embeddings (size N) in `auc_precision_at_1_bm25`.
The workflow results above come from the corrected re-run.
