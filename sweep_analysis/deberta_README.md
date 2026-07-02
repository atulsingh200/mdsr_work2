# DeBERTa-v3-large Results — Bi-encoder vs Cross-encoder

Full breakdown of `deberta-v3-large` across all 7 datasets and both architectures.

- **7 datasets**, **2 architectures** (bi-encoder, cross-encoder)
- **In-domain** metrics: test accuracy, AUC, F1
- **OOD — Milan 6-step**: model ranks 6 events by tournament scoring; evaluated on
  pairwise accuracy (30 ordered pairs) and exact order match.
  Ground truth: **t1 > t2 > t3 > t4 > t5 > t6**
- **OOD — AJO 30-sample**: 30 independent 3-step ordering instances;
  metric = perfect-order rate (fraction exactly correct).

## Head-to-head summary (deberta-v3-large)

| Dataset | bi test_acc | ce test_acc | Δ acc | bi Milan acc | ce Milan acc | Δ Milan | bi AJO rate | ce AJO rate | Δ AJO |
|---|---|---|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.6193 | 0.8816 | 0.2623 | 0.5333 | 0.3000 | -0.2333 | 16.7% | 13.3% | -0.033 |
| `aep_dataset` | 0.8835 | 0.9390 | 0.0556 | 0.5000 | 0.4667 | -0.0333 | 63.3% | 20.0% | -0.433 |
| `aep_causal_wf_v3` | 0.7516 | 0.7736 | 0.0220 | 0.3333 | 0.5333 | 0.2000 | 3.3% | 16.7% | 0.133 |
| `ajo_doc_not_tier1` | 0.5034 | 0.4966 | -0.0068 | 0.5000 | 0.5000 | 0.0000 | 96.7% | 53.3% | -0.433 |
| `ajo_newstyle` | 0.5002 | 0.5002 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 3.3% | 36.7% | 0.333 |
| `new_aep_wf_scrap` | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 3.3% | 50.0% | 0.467 |
| `new_ajo_workflows` | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 33.3% | 10.0% | -0.233 |

---

## Bi-encoder (`DirectionalClassifier`)

Two untied encoders; features = [A ; B ; A−B ; A⊙B] → MLP head.

### In-domain test metrics

| Dataset | n_train | n_test | test_acc | test_auc | test_f1 | best_val_acc |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 110775 | 6004 | 0.6193 | 0.6677 | 0.5935 | 0.6606 |
| `aep_dataset` | 67500 | 8460 | 0.8835 | 0.9392 | 0.8847 | 0.9152 |
| `aep_causal_wf_v3` | 21287 | 318 | 0.7516 | 0.8094 | 0.7410 | 0.6605 |
| `ajo_doc_not_tier1` | 88000 | 11200 | 0.5034 | 0.5000 | 0.6697 | 0.5034 |
| `ajo_newstyle` | 20897 | 2613 | 0.5002 | 0.5099 | 0.6668 | 0.5031 |
| `new_aep_wf_scrap` | 13904 | 1548 | 0.5000 | 0.5689 | 0.6667 | 0.5000 |
| `new_ajo_workflows` | 21846 | 2616 | 0.5000 | 0.5343 | 0.0000 | 0.5000 |

### OOD — Milan 6-step workflow

Ground truth: **t1 > t2 > t3 > t4 > t5 > t6**

| Dataset | Predicted order | Pairwise acc | Pairs correct | Exact match |
|---|---|---|---|---|
| `aep_causal_cls34` | t3 > t6 > t4 > t2 > t1 > t5 | 0.5333 | 16/30 | ✗ |
| `aep_dataset` | t1 > t2 > t5 > t3 > t6 > t4 | 0.5000 | 15/30 | ✗ |
| `aep_causal_wf_v3` | t4 > t3 > t6 > t1 > t5 > t2 | 0.3333 | 10/30 | ✗ |
| `ajo_doc_not_tier1` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | 15/30 | ✓ |
| `ajo_newstyle` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | 15/30 | ✓ |
| `new_aep_wf_scrap` | t6 > t5 > t1 > t2 > t4 > t3 | 0.5000 | 15/30 | ✗ |
| `new_ajo_workflows` | t1 > t2 > t6 > t4 > t5 > t3 | 0.5000 | 15/30 | ✗ |

### OOD — AJO 30-sample ordering

| Dataset | Correct / 30 | Perfect-order rate |
|---|---|---|
| `aep_causal_cls34` | 5/30 | 16.7% |
| `aep_dataset` | 19/30 | 63.3% |
| `aep_causal_wf_v3` | 1/30 | 3.3% |
| `ajo_doc_not_tier1` | 29/30 | 96.7% |
| `ajo_newstyle` | 1/30 | 3.3% |
| `new_aep_wf_scrap` | 1/30 | 3.3% |
| `new_ajo_workflows` | 10/30 | 33.3% |

### Per-dataset detail

#### `aep_causal_cls34`

**Milan** — predicted order: `t3 > t6 > t4 > t2 > t1 > t5` (no exact match ✗, pairwise acc 0.5333)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 1 | -0.447 | 5 | 1 |
| t2 | 2 | -0.4467 | 4 | 2 |
| t3 | 5 | 2.234 | 1 | 3 |
| t4 | 3 | -0.4467 | 3 | 4 |
| t5 | 0 | -0.447 | 6 | 5 |
| t6 | 4 | -0.4467 | 2 | 6 |

**AJO** — 5/30 correct (16.7%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 2 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 3 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 4 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 5 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 6 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 7 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 8 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 9 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 11 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 12 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 13 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 14 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 15 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 16 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 17 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 18 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 19 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 20 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 21 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 22 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 23 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 24 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 25 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 26 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 27 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 28 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 29 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 30 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
#### `aep_dataset`

**Milan** — predicted order: `t1 > t2 > t5 > t3 > t6 > t4` (no exact match ✗, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 5 | 0.2257 | 1 | 1 |
| t2 | 4 | 0.0422 | 2 | 2 |
| t3 | 2 | -0.0046 | 4 | 3 |
| t4 | 0 | -0.1969 | 6 | 4 |
| t5 | 3 | 0.0026 | 3 | 5 |
| t6 | 1 | -0.0691 | 5 | 6 |

**AJO** — 19/30 correct (63.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 2 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 3 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 4 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 5 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 6 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 7 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 8 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 9 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 10 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 11 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 12 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 13 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 14 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 17 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 18 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 21 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 22 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 23 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 24 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 25 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 26 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 27 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 28 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 29 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 30 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
#### `aep_causal_wf_v3`

**Milan** — predicted order: `t4 > t3 > t6 > t1 > t5 > t2` (no exact match ✗, pairwise acc 0.3333)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 2 | -2.5112 | 4 | 1 |
| t2 | 0 | -2.895 | 6 | 2 |
| t3 | 4 | 2.5733 | 2 | 3 |
| t4 | 5 | 4.4239 | 1 | 4 |
| t5 | 1 | -2.479 | 5 | 5 |
| t6 | 3 | 0.888 | 3 | 6 |

**AJO** — 1/30 correct (3.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 2 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 3 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 4 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 5 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 6 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 7 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 8 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 9 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 11 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 12 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 13 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 14 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 15 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 16 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 17 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 18 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 19 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 20 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 21 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 22 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 23 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 24 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 25 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 26 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 27 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 28 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 29 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 30 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
#### `ajo_doc_not_tier1`

**Milan** — predicted order: `t1 > t2 > t3 > t4 > t5 > t6` (exact match ✓, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 5 | 0.0 | 1 | 1 |
| t2 | 4 | 0.0 | 2 | 2 |
| t3 | 3 | 0.0 | 3 | 3 |
| t4 | 2 | 0.0 | 4 | 4 |
| t5 | 1 | 0.0 | 5 | 5 |
| t6 | 0 | 0.0 | 6 | 6 |

**AJO** — 29/30 correct (96.7%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 2 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 3 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 4 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 5 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 6 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 7 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 8 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 9 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 10 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 11 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 12 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 13 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 14 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 17 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 18 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 21 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 22 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 23 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 24 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 25 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 26 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 27 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 28 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 29 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 30 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
#### `ajo_newstyle`

**Milan** — predicted order: `t1 > t2 > t3 > t4 > t5 > t6` (exact match ✓, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 5 | 0.0 | 1 | 1 |
| t2 | 4 | 0.0 | 2 | 2 |
| t3 | 3 | -0.0 | 3 | 3 |
| t4 | 2 | -0.0 | 4 | 4 |
| t5 | 1 | -0.0 | 5 | 5 |
| t6 | 0 | -0.0 | 6 | 6 |

**AJO** — 1/30 correct (3.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 2 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 3 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 4 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 5 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 6 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 7 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 8 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 9 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 11 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 12 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 13 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 14 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 15 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 16 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 17 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 18 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 19 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 20 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 21 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 22 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 23 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 24 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 25 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 26 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 27 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 28 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 29 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 30 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
#### `new_aep_wf_scrap`

**Milan** — predicted order: `t6 > t5 > t1 > t2 > t4 > t3` (no exact match ✗, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 3 | 0.0029 | 3 | 1 |
| t2 | 2 | -0.0001 | 4 | 2 |
| t3 | 0 | -0.0078 | 6 | 3 |
| t4 | 1 | -0.0036 | 5 | 4 |
| t5 | 4 | 0.0038 | 2 | 5 |
| t6 | 5 | 0.0048 | 1 | 6 |

**AJO** — 1/30 correct (3.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 2 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 3 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 4 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 5 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 6 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 7 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 8 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 9 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 10 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 11 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 12 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 13 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 14 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 15 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 16 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 17 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 18 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 19 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 20 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 21 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 22 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 23 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 24 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 25 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 26 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 27 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 28 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 29 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 30 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
#### `new_ajo_workflows`

**Milan** — predicted order: `t1 > t2 > t6 > t4 > t5 > t3` (no exact match ✗, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 5 | 0.0009 | 1 | 1 |
| t2 | 4 | 0.0009 | 2 | 2 |
| t3 | 0 | -0.0012 | 6 | 3 |
| t4 | 2 | -0.0002 | 4 | 4 |
| t5 | 1 | -0.0006 | 5 | 5 |
| t6 | 3 | 0.0003 | 3 | 6 |

**AJO** — 10/30 correct (33.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 2 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 3 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 4 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 5 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 6 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 7 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 8 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 9 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 10 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 11 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 12 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 13 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 14 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 17 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 18 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 21 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 22 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 23 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 24 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 25 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 26 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 27 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 28 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 29 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 30 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |


---

## Cross-encoder (`CrossEncoderClassifier`)

Single joint transformer with `[CLS] text_1 [SEP] text_2 [SEP]` → Linear head.

### In-domain test metrics

| Dataset | n_train | n_test | test_acc | test_auc | test_f1 | best_val_acc |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 110775 | 6004 | 0.8816 | 0.9427 | 0.8791 | 0.8834 |
| `aep_dataset` | 67500 | 8460 | 0.9390 | 0.9808 | 0.9406 | 0.9696 |
| `aep_causal_wf_v3` | 21287 | 318 | 0.7736 | 0.8714 | 0.7616 | 0.6575 |
| `ajo_doc_not_tier1` | 88000 | 11200 | 0.4966 | 0.4970 | 0.0000 | 0.4966 |
| `ajo_newstyle` | 20897 | 2613 | 0.5002 | 0.5319 | 0.6668 | 0.5031 |
| `new_aep_wf_scrap` | 13904 | 1548 | 0.5000 | 0.5966 | 0.6667 | 0.5000 |
| `new_ajo_workflows` | 21846 | 2616 | 0.5000 | 0.4845 | 0.0000 | 0.5000 |

### OOD — Milan 6-step workflow

Ground truth: **t1 > t2 > t3 > t4 > t5 > t6**

| Dataset | Predicted order | Pairwise acc | Pairs correct | Exact match |
|---|---|---|---|---|
| `aep_causal_cls34` | t6 > t3 > t4 > t2 > t5 > t1 | 0.3000 | 9/30 | ✗ |
| `aep_dataset` | t5 > t2 > t6 > t3 > t1 > t4 | 0.4667 | 14/30 | ✗ |
| `aep_causal_wf_v3` | t3 > t1 > t6 > t4 > t2 > t5 | 0.5333 | 16/30 | ✗ |
| `ajo_doc_not_tier1` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | 15/30 | ✓ |
| `ajo_newstyle` | t4 > t1 > t6 > t2 > t3 > t5 | 0.5000 | 15/30 | ✗ |
| `new_aep_wf_scrap` | t1 > t3 > t5 > t4 > t6 > t2 | 0.5000 | 15/30 | ✗ |
| `new_ajo_workflows` | t3 > t4 > t6 > t2 > t5 > t1 | 0.5000 | 15/30 | ✗ |

### OOD — AJO 30-sample ordering

| Dataset | Correct / 30 | Perfect-order rate |
|---|---|---|
| `aep_causal_cls34` | 4/30 | 13.3% |
| `aep_dataset` | 6/30 | 20.0% |
| `aep_causal_wf_v3` | 5/30 | 16.7% |
| `ajo_doc_not_tier1` | 16/30 | 53.3% |
| `ajo_newstyle` | 11/30 | 36.7% |
| `new_aep_wf_scrap` | 15/30 | 50.0% |
| `new_ajo_workflows` | 3/30 | 10.0% |

### Per-dataset detail

#### `aep_causal_cls34`

**Milan** — predicted order: `t6 > t3 > t4 > t2 > t5 > t1` (no exact match ✗, pairwise acc 0.3000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 0 | -4.0357 | 6 | 1 |
| t2 | 2 | -1.065 | 4 | 2 |
| t3 | 4 | 3.3822 | 2 | 3 |
| t4 | 3 | 1.6878 | 3 | 4 |
| t5 | 1 | -3.8966 | 5 | 5 |
| t6 | 5 | 3.9272 | 1 | 6 |

**AJO** — 4/30 correct (13.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 2 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 3 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 4 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 5 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 6 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 7 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 8 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 9 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 11 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 12 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 13 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 14 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 17 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 18 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 19 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 20 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 21 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 22 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 23 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 24 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 25 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 26 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 27 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 28 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 29 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 30 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
#### `aep_dataset`

**Milan** — predicted order: `t5 > t2 > t6 > t3 > t1 > t4` (no exact match ✗, pairwise acc 0.4667)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 1 | -0.8831 | 5 | 1 |
| t2 | 4 | 1.0227 | 2 | 2 |
| t3 | 3 | 0.9479 | 4 | 3 |
| t4 | 0 | -4.9086 | 6 | 4 |
| t5 | 4 | 1.9726 | 1 | 5 |
| t6 | 3 | 1.8485 | 3 | 6 |

**AJO** — 6/30 correct (20.0%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 2 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 3 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 4 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 5 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 6 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 7 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 8 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 9 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 11 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 12 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 13 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 14 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 15 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 16 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 17 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 18 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 21 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 22 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 23 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 24 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 25 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 26 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 27 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 28 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 29 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 30 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
#### `aep_causal_wf_v3`

**Milan** — predicted order: `t3 > t1 > t6 > t4 > t2 > t5` (no exact match ✗, pairwise acc 0.5333)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 4 | 2.99 | 2 | 1 |
| t2 | 1 | -3.8892 | 5 | 2 |
| t3 | 5 | 3.0726 | 1 | 3 |
| t4 | 2 | -0.8889 | 4 | 4 |
| t5 | 0 | -4.0446 | 6 | 5 |
| t6 | 3 | 2.7602 | 3 | 6 |

**AJO** — 5/30 correct (16.7%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 2 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 3 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 4 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 5 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 6 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 7 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 8 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 9 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 11 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 12 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 13 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 14 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 17 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 18 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 21 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 22 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 23 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 24 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 25 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 26 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 27 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 28 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 29 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 30 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
#### `ajo_doc_not_tier1`

**Milan** — predicted order: `t1 > t2 > t3 > t4 > t5 > t6` (exact match ✓, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 5 | 0.0 | 1 | 1 |
| t2 | 4 | 0.0 | 2 | 2 |
| t3 | 3 | 0.0 | 3 | 3 |
| t4 | 2 | 0.0 | 4 | 4 |
| t5 | 1 | 0.0 | 5 | 5 |
| t6 | 0 | 0.0 | 6 | 6 |

**AJO** — 16/30 correct (53.3%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 2 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 3 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 4 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 5 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 6 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 7 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 8 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 9 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 10 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 11 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 12 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 13 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 14 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 17 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 18 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 19 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 20 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 21 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 22 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 23 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 24 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 25 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 26 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 27 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 28 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 29 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 30 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
#### `ajo_newstyle`

**Milan** — predicted order: `t4 > t1 > t6 > t2 > t3 > t5` (no exact match ✗, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 3 | 0.0 | 2 | 1 |
| t2 | 2 | 0.0 | 4 | 2 |
| t3 | 1 | -0.0 | 5 | 3 |
| t4 | 5 | 0.0001 | 1 | 4 |
| t5 | 1 | -0.0001 | 6 | 5 |
| t6 | 3 | -0.0 | 3 | 6 |

**AJO** — 11/30 correct (36.7%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 2 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 3 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 4 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 5 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 6 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 7 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 8 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 9 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 10 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 11 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 12 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 13 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 14 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 17 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 18 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 21 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 22 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 23 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 24 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 25 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 26 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 27 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 28 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 29 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 30 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
#### `new_aep_wf_scrap`

**Milan** — predicted order: `t1 > t3 > t5 > t4 > t6 > t2` (no exact match ✗, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 5 | 0.0005 | 1 | 1 |
| t2 | 0 | -0.0005 | 6 | 2 |
| t3 | 4 | 0.0006 | 2 | 3 |
| t4 | 2 | -0.0004 | 4 | 4 |
| t5 | 3 | 0.0003 | 3 | 5 |
| t6 | 1 | -0.0005 | 5 | 6 |

**AJO** — 15/30 correct (50.0%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 2 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 3 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 4 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 5 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 6 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 7 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 8 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 9 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 10 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 11 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 12 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 13 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 14 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 15 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 16 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 17 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 18 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 19 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 20 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 21 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 22 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 23 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 24 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 25 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 26 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 27 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 28 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 29 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 30 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
#### `new_ajo_workflows`

**Milan** — predicted order: `t3 > t4 > t6 > t2 > t5 > t1` (no exact match ✗, pairwise acc 0.5000)

| Event | Tournament wins | Net precedence | Predicted rank | True rank |
|---|---|---|---|---|
| t1 | 1 | -0.0001 | 6 | 1 |
| t2 | 2 | 0.0 | 4 | 2 |
| t3 | 4 | 0.0003 | 1 | 3 |
| t4 | 3 | 0.0 | 2 | 4 |
| t5 | 2 | -0.0002 | 5 | 5 |
| t6 | 3 | -0.0 | 3 | 6 |

**AJO** — 3/30 correct (10.0%)

| Sample | Ground truth | Predicted | Correct |
|---|---|---|---|
| 1 | t1 > t2 > t3 | t2 > t1 > t3 | ✗ |
| 2 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 3 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 4 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 5 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 6 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 7 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 8 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 9 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 10 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 11 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 12 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 13 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 14 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 15 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 16 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 17 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 18 | t1 > t2 > t3 | t3 > t1 > t2 | ✗ |
| 19 | t1 > t2 > t3 | t1 > t2 > t3 | ✓ |
| 20 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 21 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 22 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 23 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 24 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |
| 25 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 26 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 27 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 28 | t1 > t2 > t3 | t3 > t2 > t1 | ✗ |
| 29 | t1 > t2 > t3 | t2 > t3 > t1 | ✗ |
| 30 | t1 > t2 > t3 | t1 > t3 > t2 | ✗ |


---

## CSV files

| File | Contents |
|---|---|
| [deberta_results.csv](deberta_results.csv) | In-domain + OOD summary, one row per (encoder, dataset) |
| [deberta_milan_per_event.csv](deberta_milan_per_event.csv) | Milan per-event tournament wins, net precedence, predicted rank |
| [deberta_ajo_per_sample.csv](deberta_ajo_per_sample.csv) | AJO 30 samples — ground truth vs predicted order, correct flag |

*Generated by [build_deberta_analysis.py](build_deberta_analysis.py).*
