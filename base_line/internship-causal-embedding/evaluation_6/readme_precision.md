# Precision@1 and AUC — Methodology

## Metrics covered

| Metric | What it measures |
|---|---|
| **P@1 (Precision@1)** | Fraction of anchors where the gold positive ranks strictly above **every** sampled negative. |
| **AUC** | ROC-AUC over the `(1 positive + N negatives)` similarity scores for each anchor, averaged across all anchors. |

---

## How negatives are counted

For each anchor, **N random negatives** are sampled from the test-split positives of *other* anchors (i.e. true unrelated continuations). The default is `N = 4`, giving a pool of **5 candidates per anchor** (1 positive + 4 negatives).

```
--n-negatives 4     →  5-way ranking per anchor  (random baseline P@1 = 1/5 = 0.20)
--n-negatives 9     →  10-way ranking per anchor (random baseline P@1 = 1/10 = 0.10)
```

Negatives are sampled **without replacement per anchor** and re-drawn with a fixed seed (`--seed 0`) for reproducibility. For datasets where the test split is smaller than `N`, all available other positives are used.

---

## How P@1 is computed

For each anchor `a` with gold positive `p` and negatives `n_1 … n_N`:

1. Compute cosine similarities: `s_p = sim(a, p)`, `s_i = sim(a, n_i)` for each `i`.
2. Anchor passes if `s_p > s_i` for **all** N negatives.
3. **P@1 = (number of passing anchors) / (total anchors)**.

> With `N=4`, a perfect model scores 1.00; a random encoder scores ≈ 0.20.

---

## How AUC is computed

For each anchor, the N+1 similarity scores form a binary classification problem:

- Label `1` → the positive
- Label `0` → each of the N negatives

ROC-AUC is computed over these `N+1` scores. Per-anchor AUC values are then **macro-averaged** across all anchors.

> A random encoder scores AUC ≈ 0.50; a perfect encoder scores 1.00.

---

## Results (P@1 and AUC only)

### all-MiniLM-L6-v2

| dataset | N anchors | AUC | P@1 |
|---|---:|---:|---:|
| aep_causal | 562 | 0.9371 | 0.8754 |
| aep_followup | 1,255 | 0.8435 | 0.7498 |
| followupqg | 501 | 0.9795 | 0.9441 |
| multiwoz_v24 | 7,368 | 0.6706 | 0.4053 |
| qrecc | 5,204 | 0.8520 | 0.7100 |
| workflow | 260,487 | 0.8994 | 0.8046 |

### bert-base-uncased (sanity-check lower bound)

| dataset | N anchors | AUC | P@1 |
|---|---:|---:|---:|
| aep_causal | 562 | 0.7294 | 0.5338 |
| aep_followup | 1,255 | 0.6128 | 0.4143 |
| followupqg | 501 | 0.6350 | 0.4611 |
| multiwoz_v24 | 7,368 | 0.5472 | 0.2782 |
| qrecc | 5,204 | 0.5400 | 0.4450 |
| workflow | 260,487 | 0.7817 | 0.6273 |

Random baseline (N=4 negatives): **P@1 ≈ 0.20**, **AUC ≈ 0.50**.

---

## Implementation reference

Both metrics are computed in [`metrics_extra.py`](metrics_extra.py) inside `auc_precision_at_1_with_random_negatives()` using GPU-accelerated torch operations.
