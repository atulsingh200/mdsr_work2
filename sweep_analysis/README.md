# Sweep Analysis — Bi-Encoder vs Cross-Encoder

Comparison of two training sweeps over **7 datasets × 6 models** for the
document-pair ordering task, plus out-of-distribution (OOD) ordering evaluation
on both architectures.

- **Bi-encoder** (`/mnt/localssd/baseline_sweep/`) — `DirectionalClassifier`
  (2 untied encoders). **42/42 runs complete**, each with in-domain test
  metrics and OOD eval (Milan 6-step ordering + AJO 30-sample ordering).
- **Cross-encoder** (`/mnt/localssd/crossencoder_sweep/`) — `CrossEncoderClassifier`
  (single joint transformer, `[CLS] text_1 [SEP] text_2 [SEP]`). **42/42 runs
  complete**. OOD eval was run for all 42 cross-encoder runs.

## Headline Numbers

### In-domain test accuracy
- Best **bi-encoder**: `ajo_doc_not_tier1` / `all-mpnet-base-v2` = **0.9942**
- Best **cross-encoder**: `ajo_doc_not_tier1` / `all-mpnet-base-v2` = **0.9893**

### OOD — AJO 30-sample perfect-order rate
- Best **bi-encoder**: `ajo_doc_not_tier1` / `deberta-v3-large` = **0.967**
- Best **cross-encoder**: `aep_causal_cls34` / `e5-large-v2` = **1.000**

### OOD — Milan 6-step pairwise accuracy
- Best **bi-encoder**: `aep_dataset` / `all-mpnet-base-v2` = **0.8667**
- Best **cross-encoder**: `new_aep_wf_scrap` / `bge-base-en-v1.5` = **0.9000**

## Metrics Glossary

- **test_acc / test_auc / test_f1** — in-domain test-set pairwise-ordering metrics.
- **Milan 6-step workflow** — a single fixed 6-event chain; the model scores all
  ordered pairs, events are ranked by tournament wins, and the result is scored by
  `pairwise_acc` (fraction of the 30 ordered pairs correct) and `order_match`
  (whether the full predicted order equals ground truth). Ground truth: **t1 > t2 > t3 > t4 > t5 > t6**.
- **AJO 30-sample ordering** — 30 small ordering instances; `perfect_order_rate`
  is the fraction whose full predicted order is exactly correct.
- **Bi-encoder** — `DirectionalClassifier`: two separate encoders produce
  embeddings `A`, `B`; the classifier uses `[A; B; A−B; A⊙B]` as input features.
- **Cross-encoder** — `CrossEncoderClassifier`: a single transformer sees
  `[CLS] text_1 [SEP] text_2 [SEP]` jointly; the `[CLS]` token feeds a linear head.

---

## Table A — In-domain test accuracy: Bi-encoder

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.6193 | 0.8877 | 0.8744 | 0.8728 | 0.8609 | 0.8774 |
| `aep_dataset` | 0.8835 | 0.9329 | 0.9121 | 0.9293 | 0.9228 | 0.9275 |
| `aep_causal_wf_v3` | 0.7516 | 0.7767 | 0.7296 | 0.7296 | 0.7390 | 0.7862 |
| `ajo_doc_not_tier1` | 0.5034 | 0.9942 | 0.9815 | 0.9705 | 0.9933 | 0.9918 |
| `ajo_newstyle` | 0.5002 | 0.6529 | 0.6169 | 0.5320 | 0.6334 | 0.6414 |
| `new_aep_wf_scrap` | 0.5000 | 0.6835 | 0.6479 | 0.5807 | 0.6815 | 0.6686 |
| `new_ajo_workflows` | 0.5000 | 0.6284 | 0.5975 | 0.5432 | 0.6277 | 0.6323 |

## Table B — In-domain test accuracy: Cross-encoder

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.8816 | 0.8719 | 0.7876 | 0.4888 | 0.8145 | 0.8286 |
| `aep_dataset` | 0.9390 | 0.9251 | 0.8690 | 0.5106 | 0.8963 | 0.9126 |
| `aep_causal_wf_v3` | 0.7736 | 0.6698 | 0.6447 | 0.4654 | 0.6321 | 0.6792 |
| `ajo_doc_not_tier1` | 0.4966 | 0.9893 | 0.9663 | 0.4966 | 0.9696 | 0.4966 |
| `ajo_newstyle` | 0.5002 | 0.6100 | 0.5522 | 0.5002 | 0.5515 | 0.5002 |
| `new_aep_wf_scrap` | 0.5000 | 0.6957 | 0.5866 | 0.5000 | 0.6124 | 0.5000 |
| `new_ajo_workflows` | 0.5000 | 0.6422 | 0.5558 | 0.5000 | 0.5558 | 0.5818 |

## Table C — In-domain test AUC: Bi-encoder

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.6677 | 0.9460 | 0.9197 | 0.9101 | 0.9064 | 0.9154 |
| `aep_dataset` | 0.9392 | 0.9654 | 0.9537 | 0.9445 | 0.9530 | 0.9500 |
| `aep_causal_wf_v3` | 0.8094 | 0.8553 | 0.8089 | 0.8124 | 0.8219 | 0.8508 |
| `ajo_doc_not_tier1` | 0.5000 | 0.9973 | 0.9953 | 0.9726 | 0.9984 | 0.9965 |
| `ajo_newstyle` | 0.5099 | 0.7546 | 0.7034 | 0.5534 | 0.7264 | 0.7344 |
| `new_aep_wf_scrap` | 0.5689 | 0.7565 | 0.7194 | 0.6168 | 0.7616 | 0.7461 |
| `new_ajo_workflows` | 0.5343 | 0.6839 | 0.6464 | 0.5680 | 0.6817 | 0.6934 |

## Table D — In-domain test AUC: Cross-encoder

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.9427 | 0.9129 | 0.8479 | 0.5000 | 0.8646 | 0.8674 |
| `aep_dataset` | 0.9808 | 0.9625 | 0.9381 | 0.5000 | 0.9396 | 0.9432 |
| `aep_causal_wf_v3` | 0.8714 | 0.7467 | 0.6713 | 0.5000 | 0.7176 | 0.7225 |
| `ajo_doc_not_tier1` | 0.4970 | 0.9953 | 0.9862 | 0.4991 | 0.9890 | 0.5000 |
| `ajo_newstyle` | 0.5319 | 0.6936 | 0.5790 | 0.5000 | 0.6101 | 0.5000 |
| `new_aep_wf_scrap` | 0.5966 | 0.7727 | 0.6292 | 0.4910 | 0.6658 | 0.4946 |
| `new_ajo_workflows` | 0.4845 | 0.6886 | 0.5907 | 0.4992 | 0.5986 | 0.6326 |

## Table E — In-domain test F1: Bi-encoder

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.5935 | 0.8795 | 0.8722 | 0.8671 | 0.8565 | 0.8738 |
| `aep_dataset` | 0.8847 | 0.9342 | 0.9141 | 0.9308 | 0.9247 | 0.9285 |
| `aep_causal_wf_v3` | 0.7410 | 0.7657 | 0.7190 | 0.7244 | 0.7331 | 0.7606 |
| `ajo_doc_not_tier1` | 0.6697 | 0.9942 | 0.9817 | 0.9707 | 0.9934 | 0.9918 |
| `ajo_newstyle` | 0.6668 | 0.6571 | 0.5858 | 0.4002 | 0.6267 | 0.5781 |
| `new_aep_wf_scrap` | 0.6667 | 0.6742 | 0.6482 | 0.5755 | 0.6584 | 0.6591 |
| `new_ajo_workflows` | 0.0000 | 0.6354 | 0.6150 | 0.5829 | 0.6319 | 0.6254 |

## Table F — In-domain test F1: Cross-encoder

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.8791 | 0.8679 | 0.7892 | 0.6567 | 0.8082 | 0.8286 |
| `aep_dataset` | 0.9406 | 0.9267 | 0.8714 | 0.6761 | 0.8997 | 0.9138 |
| `aep_causal_wf_v3` | 0.7616 | 0.6392 | 0.6246 | 0.6352 | 0.5776 | 0.6792 |
| `ajo_doc_not_tier1` | 0.0000 | 0.9894 | 0.9666 | 0.0000 | 0.9698 | 0.0000 |
| `ajo_newstyle` | 0.6668 | 0.6002 | 0.5004 | 0.6668 | 0.5447 | 0.6668 |
| `new_aep_wf_scrap` | 0.6667 | 0.6955 | 0.5850 | 0.0000 | 0.6324 | 0.0000 |
| `new_ajo_workflows` | 0.0000 | 0.6513 | 0.5805 | 0.0000 | 0.5548 | 0.6045 |

## Table G — Bi vs Cross (best model per dataset, by test_acc)

| Dataset | Best bi-encoder | bi acc | Best cross-encoder | ce acc | Δ (ce − bi) |
|---|---|---|---|---|---|
| `aep_causal_cls34` | `all-mpnet-base-v2` | 0.8877 | `deberta-v3-large` | 0.8816 | -0.0062 |
| `aep_dataset` | `all-mpnet-base-v2` | 0.9329 | `deberta-v3-large` | 0.9390 | 0.0061 |
| `aep_causal_wf_v3` | `bge-large-en-v1.5` | 0.7862 | `deberta-v3-large` | 0.7736 | -0.0126 |
| `ajo_doc_not_tier1` | `all-mpnet-base-v2` | 0.9942 | `all-mpnet-base-v2` | 0.9893 | -0.0049 |
| `ajo_newstyle` | `all-mpnet-base-v2` | 0.6529 | `all-mpnet-base-v2` | 0.6100 | -0.0429 |
| `new_aep_wf_scrap` | `all-mpnet-base-v2` | 0.6835 | `all-mpnet-base-v2` | 0.6957 | 0.0123 |
| `new_ajo_workflows` | `bge-large-en-v1.5` | 0.6323 | `all-mpnet-base-v2` | 0.6422 | 0.0099 |

*Full per-cell comparison in [comparison_biencoder_vs_crossencoder.csv](comparison_biencoder_vs_crossencoder.csv).*

---

## Table H — OOD Milan 6-step workflow: Bi-encoder

Pairwise accuracy over 30 ordered pairs; **✓** = full order exactly matched.

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.5333 | 0.5333 | 0.3000 | 0.4333 | 0.3000 | 0.3667 |
| `aep_dataset` | 0.5000 | 0.8667 | 0.6000 | 0.5333 ✓ | 0.4667 | 0.6333 |
| `aep_causal_wf_v3` | 0.3333 | 0.3667 | 0.3000 | 0.5333 | 0.2667 | 0.5000 |
| `ajo_doc_not_tier1` | 0.5000 ✓ | 0.5333 | 0.3000 | 0.4333 | 0.5333 | 0.4333 |
| `ajo_newstyle` | 0.5000 ✓ | 0.2667 | 0.5333 | 0.5000 | 0.4667 | 0.4667 |
| `new_aep_wf_scrap` | 0.5000 | 0.4333 | 0.6000 | 0.5000 | 0.8000 | 0.6667 |
| `new_ajo_workflows` | 0.5000 | 0.6333 | 0.7333 | 0.7000 | 0.7000 | 0.7000 |

### Milan predicted order per model — Bi-encoder

Ground-truth order: **t1 > t2 > t3 > t4 > t5 > t6**

| Dataset | Model | Predicted order | Pairwise acc | Exact match |
|---|---|---|---|---|
| `aep_causal_cls34` | `deberta-v3-large` | t3 > t6 > t4 > t2 > t1 > t5 | 0.5333 | ✗ |
| `aep_causal_cls34` | `all-mpnet-base-v2` | t4 > t1 > t3 > t5 > t6 > t2 | 0.5333 | ✗ |
| `aep_causal_cls34` | `all-MiniLM-L6-v2` | t6 > t4 > t3 > t2 > t1 > t5 | 0.3000 | ✗ |
| `aep_causal_cls34` | `e5-large-v2` | t4 > t6 > t1 > t5 > t2 > t3 | 0.4333 | ✗ |
| `aep_causal_cls34` | `bge-base-en-v1.5` | t3 > t6 > t4 > t5 > t2 > t1 | 0.3000 | ✗ |
| `aep_causal_cls34` | `bge-large-en-v1.5` | t6 > t3 > t4 > t1 > t5 > t2 | 0.3667 | ✗ |
| `aep_dataset` | `deberta-v3-large` | t1 > t2 > t5 > t3 > t6 > t4 | 0.5000 | ✗ |
| `aep_dataset` | `all-mpnet-base-v2` | t2 > t1 > t3 > t5 > t4 > t6 | 0.8667 | ✗ |
| `aep_dataset` | `all-MiniLM-L6-v2` | t2 > t3 > t5 > t4 > t1 > t6 | 0.6000 | ✗ |
| `aep_dataset` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5333 | ✓ |
| `aep_dataset` | `bge-base-en-v1.5` | t5 > t2 > t3 > t4 > t1 > t6 | 0.4667 | ✗ |
| `aep_dataset` | `bge-large-en-v1.5` | t1 > t5 > t2 > t3 > t4 > t6 | 0.6333 | ✗ |
| `aep_causal_wf_v3` | `deberta-v3-large` | t4 > t3 > t6 > t1 > t5 > t2 | 0.3333 | ✗ |
| `aep_causal_wf_v3` | `all-mpnet-base-v2` | t4 > t3 > t6 > t1 > t2 > t5 | 0.3667 | ✗ |
| `aep_causal_wf_v3` | `all-MiniLM-L6-v2` | t4 > t6 > t5 > t3 > t2 > t1 | 0.3000 | ✗ |
| `aep_causal_wf_v3` | `e5-large-v2` | t4 > t1 > t3 > t6 > t2 > t5 | 0.5333 | ✗ |
| `aep_causal_wf_v3` | `bge-base-en-v1.5` | t3 > t4 > t6 > t5 > t2 > t1 | 0.2667 | ✗ |
| `aep_causal_wf_v3` | `bge-large-en-v1.5` | t3 > t4 > t1 > t2 > t6 > t5 | 0.5000 | ✗ |
| `ajo_doc_not_tier1` | `deberta-v3-large` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `ajo_doc_not_tier1` | `all-mpnet-base-v2` | t5 > t1 > t3 > t4 > t2 > t6 | 0.5333 | ✗ |
| `ajo_doc_not_tier1` | `all-MiniLM-L6-v2` | t5 > t3 > t1 > t6 > t4 > t2 | 0.3000 | ✗ |
| `ajo_doc_not_tier1` | `e5-large-v2` | t6 > t1 > t4 > t5 > t3 > t2 | 0.4333 | ✗ |
| `ajo_doc_not_tier1` | `bge-base-en-v1.5` | t3 > t5 > t2 > t1 > t6 > t4 | 0.5333 | ✗ |
| `ajo_doc_not_tier1` | `bge-large-en-v1.5` | t3 > t4 > t5 > t2 > t6 > t1 | 0.4333 | ✗ |
| `ajo_newstyle` | `deberta-v3-large` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `ajo_newstyle` | `all-mpnet-base-v2` | t5 > t6 > t3 > t1 > t4 > t2 | 0.2667 | ✗ |
| `ajo_newstyle` | `all-MiniLM-L6-v2` | t6 > t1 > t3 > t4 > t2 > t5 | 0.5333 | ✗ |
| `ajo_newstyle` | `e5-large-v2` | t3 > t1 > t2 > t4 > t5 > t6 | 0.5000 | ✗ |
| `ajo_newstyle` | `bge-base-en-v1.5` | t6 > t5 > t3 > t4 > t1 > t2 | 0.4667 | ✗ |
| `ajo_newstyle` | `bge-large-en-v1.5` | t3 > t5 > t2 > t6 > t4 > t1 | 0.4667 | ✗ |
| `new_aep_wf_scrap` | `deberta-v3-large` | t6 > t5 > t1 > t2 > t4 > t3 | 0.5000 | ✗ |
| `new_aep_wf_scrap` | `all-mpnet-base-v2` | t1 > t6 > t4 > t5 > t3 > t2 | 0.4333 | ✗ |
| `new_aep_wf_scrap` | `all-MiniLM-L6-v2` | t6 > t1 > t3 > t4 > t2 > t5 | 0.6000 | ✗ |
| `new_aep_wf_scrap` | `e5-large-v2` | t3 > t1 > t4 > t6 > t5 > t2 | 0.5000 | ✗ |
| `new_aep_wf_scrap` | `bge-base-en-v1.5` | t1 > t3 > t2 > t6 > t5 > t4 | 0.8000 | ✗ |
| `new_aep_wf_scrap` | `bge-large-en-v1.5` | t1 > t3 > t6 > t4 > t2 > t5 | 0.6667 | ✗ |
| `new_ajo_workflows` | `deberta-v3-large` | t1 > t2 > t6 > t4 > t5 > t3 | 0.5000 | ✗ |
| `new_ajo_workflows` | `all-mpnet-base-v2` | t3 > t6 > t1 > t2 > t5 > t4 | 0.6333 | ✗ |
| `new_ajo_workflows` | `all-MiniLM-L6-v2` | t2 > t3 > t1 > t6 > t4 > t5 | 0.7333 | ✗ |
| `new_ajo_workflows` | `e5-large-v2` | t3 > t2 > t1 > t5 > t6 > t4 | 0.7000 | ✗ |
| `new_ajo_workflows` | `bge-base-en-v1.5` | t1 > t3 > t6 > t2 > t5 > t4 | 0.7000 | ✗ |
| `new_ajo_workflows` | `bge-large-en-v1.5` | t2 > t3 > t1 > t4 > t6 > t5 | 0.7000 | ✗ |

---

## Table I — OOD Milan 6-step workflow: Cross-encoder

Pairwise accuracy over 30 ordered pairs; **✓** = full order exactly matched.

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.3000 | 0.4000 | 0.3333 | 0.5000 ✓ | 0.2333 | 0.6000 |
| `aep_dataset` | 0.4667 | 0.7000 | 0.4667 | 0.5000 ✓ | 0.6667 | 0.5667 |
| `aep_causal_wf_v3` | 0.5333 | 0.5333 | 0.5333 | 0.5000 ✓ | 0.6000 | 0.5000 |
| `ajo_doc_not_tier1` | 0.5000 ✓ | 0.3667 | 0.5000 | 0.5000 | 0.4667 | 0.5000 ✓ |
| `ajo_newstyle` | 0.5000 | 0.3333 | 0.4333 | 0.5000 ✓ | 0.5667 | 0.5000 ✓ |
| `new_aep_wf_scrap` | 0.5000 | 0.5667 | 0.5667 | 0.5000 ✓ | 0.9000 ✓ | 0.5000 |
| `new_ajo_workflows` | 0.5000 | 0.6667 | 0.5333 | 0.5000 ✓ | 0.4333 | 0.5000 |

### Milan predicted order per model — Cross-encoder

Ground-truth order: **t1 > t2 > t3 > t4 > t5 > t6**

| Dataset | Model | Predicted order | Pairwise acc | Exact match |
|---|---|---|---|---|
| `aep_causal_cls34` | `deberta-v3-large` | t6 > t3 > t4 > t2 > t5 > t1 | 0.3000 | ✗ |
| `aep_causal_cls34` | `all-mpnet-base-v2` | t4 > t3 > t6 > t2 > t1 > t5 | 0.4000 | ✗ |
| `aep_causal_cls34` | `all-MiniLM-L6-v2` | t3 > t4 > t6 > t2 > t1 > t5 | 0.3333 | ✗ |
| `aep_causal_cls34` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `aep_causal_cls34` | `bge-base-en-v1.5` | t6 > t4 > t3 > t2 > t5 > t1 | 0.2333 | ✗ |
| `aep_causal_cls34` | `bge-large-en-v1.5` | t4 > t1 > t6 > t3 > t2 > t5 | 0.6000 | ✗ |
| `aep_dataset` | `deberta-v3-large` | t5 > t2 > t6 > t3 > t1 > t4 | 0.4667 | ✗ |
| `aep_dataset` | `all-mpnet-base-v2` | t1 > t2 > t6 > t5 > t3 > t4 | 0.7000 | ✗ |
| `aep_dataset` | `all-MiniLM-L6-v2` | t2 > t5 > t3 > t1 > t6 > t4 | 0.4667 | ✗ |
| `aep_dataset` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `aep_dataset` | `bge-base-en-v1.5` | t2 > t1 > t5 > t3 > t6 > t4 | 0.6667 | ✗ |
| `aep_dataset` | `bge-large-en-v1.5` | t1 > t2 > t3 > t5 > t6 > t4 | 0.5667 | ✗ |
| `aep_causal_wf_v3` | `deberta-v3-large` | t3 > t1 > t6 > t4 > t2 > t5 | 0.5333 | ✗ |
| `aep_causal_wf_v3` | `all-mpnet-base-v2` | t3 > t4 > t1 > t6 > t5 > t2 | 0.5333 | ✗ |
| `aep_causal_wf_v3` | `all-MiniLM-L6-v2` | t4 > t1 > t3 > t5 > t6 > t2 | 0.5333 | ✗ |
| `aep_causal_wf_v3` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `aep_causal_wf_v3` | `bge-base-en-v1.5` | t4 > t5 > t1 > t3 > t6 > t2 | 0.6000 | ✗ |
| `aep_causal_wf_v3` | `bge-large-en-v1.5` | t3 > t5 > t4 > t6 > t1 > t2 | 0.5000 | ✗ |
| `ajo_doc_not_tier1` | `deberta-v3-large` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `ajo_doc_not_tier1` | `all-mpnet-base-v2` | t4 > t5 > t6 > t3 > t1 > t2 | 0.3667 | ✗ |
| `ajo_doc_not_tier1` | `all-MiniLM-L6-v2` | t5 > t2 > t3 > t1 > t4 > t6 | 0.5000 | ✗ |
| `ajo_doc_not_tier1` | `e5-large-v2` | t1 > t6 > t2 > t3 > t4 > t5 | 0.5000 | ✗ |
| `ajo_doc_not_tier1` | `bge-base-en-v1.5` | t5 > t3 > t6 > t4 > t2 > t1 | 0.4667 | ✗ |
| `ajo_doc_not_tier1` | `bge-large-en-v1.5` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `ajo_newstyle` | `deberta-v3-large` | t4 > t1 > t6 > t2 > t3 > t5 | 0.5000 | ✗ |
| `ajo_newstyle` | `all-mpnet-base-v2` | t3 > t6 > t5 > t1 > t4 > t2 | 0.3333 | ✗ |
| `ajo_newstyle` | `all-MiniLM-L6-v2` | t2 > t3 > t5 > t4 > t6 > t1 | 0.4333 | ✗ |
| `ajo_newstyle` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `ajo_newstyle` | `bge-base-en-v1.5` | t2 > t5 > t6 > t1 > t3 > t4 | 0.5667 | ✗ |
| `ajo_newstyle` | `bge-large-en-v1.5` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `new_aep_wf_scrap` | `deberta-v3-large` | t1 > t3 > t5 > t4 > t6 > t2 | 0.5000 | ✗ |
| `new_aep_wf_scrap` | `all-mpnet-base-v2` | t1 > t3 > t5 > t4 > t6 > t2 | 0.5667 | ✗ |
| `new_aep_wf_scrap` | `all-MiniLM-L6-v2` | t2 > t3 > t1 > t4 > t6 > t5 | 0.5667 | ✗ |
| `new_aep_wf_scrap` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `new_aep_wf_scrap` | `bge-base-en-v1.5` | t1 > t2 > t3 > t4 > t5 > t6 | 0.9000 | ✓ |
| `new_aep_wf_scrap` | `bge-large-en-v1.5` | t1 > t3 > t2 > t4 > t5 > t6 | 0.5000 | ✗ |
| `new_ajo_workflows` | `deberta-v3-large` | t3 > t4 > t6 > t2 > t5 > t1 | 0.5000 | ✗ |
| `new_ajo_workflows` | `all-mpnet-base-v2` | t2 > t3 > t6 > t1 > t4 > t5 | 0.6667 | ✗ |
| `new_ajo_workflows` | `all-MiniLM-L6-v2` | t2 > t3 > t1 > t4 > t6 > t5 | 0.5333 | ✗ |
| `new_ajo_workflows` | `e5-large-v2` | t1 > t2 > t3 > t4 > t5 > t6 | 0.5000 | ✓ |
| `new_ajo_workflows` | `bge-base-en-v1.5` | t1 > t6 > t3 > t5 > t2 > t4 | 0.4333 | ✗ |
| `new_ajo_workflows` | `bge-large-en-v1.5` | t3 > t6 > t5 > t1 > t2 > t4 | 0.5000 | ✗ |

---

## Table J — OOD AJO 30-sample ordering: Bi-encoder

`perfect_order_rate` (correct / 30).

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.167 (5/30) | 0.267 (8/30) | 0.133 (4/30) | 0.167 (5/30) | 0.133 (4/30) | 0.133 (4/30) |
| `aep_dataset` | 0.633 (19/30) | 0.267 (8/30) | 0.267 (8/30) | 0.300 (9/30) | 0.400 (12/30) | 0.467 (14/30) |
| `aep_causal_wf_v3` | 0.033 (1/30) | 0.000 (0/30) | 0.033 (1/30) | 0.133 (4/30) | 0.233 (7/30) | 0.067 (2/30) |
| `ajo_doc_not_tier1` | 0.967 (29/30) | 0.600 (18/30) | 0.300 (9/30) | 0.500 (15/30) | 0.500 (15/30) | 0.467 (14/30) |
| `ajo_newstyle` | 0.033 (1/30) | 0.400 (12/30) | 0.067 (2/30) | 0.700 (21/30) | 0.200 (6/30) | 0.267 (8/30) |
| `new_aep_wf_scrap` | 0.033 (1/30) | 0.367 (11/30) | 0.300 (9/30) | 0.367 (11/30) | 0.133 (4/30) | 0.100 (3/30) |
| `new_ajo_workflows` | 0.333 (10/30) | 0.233 (7/30) | 0.167 (5/30) | 0.567 (17/30) | 0.233 (7/30) | 0.367 (11/30) |

---

## Table K — OOD AJO 30-sample ordering: Cross-encoder

`perfect_order_rate` (correct / 30).

| Dataset | deberta-v3-large | all-mpnet-base-v2 | all-MiniLM-L6-v2 | e5-large-v2 | bge-base-en-v1.5 | bge-large-en-v1.5 |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.133 (4/30) | 0.000 (0/30) | 0.133 (4/30) | 1.000 (30/30) | 0.100 (3/30) | 0.100 (3/30) |
| `aep_dataset` | 0.200 (6/30) | 0.567 (17/30) | 0.333 (10/30) | 1.000 (30/30) | 0.433 (13/30) | 0.200 (6/30) |
| `aep_causal_wf_v3` | 0.167 (5/30) | 0.100 (3/30) | 0.300 (9/30) | 1.000 (30/30) | 0.167 (5/30) | 0.300 (9/30) |
| `ajo_doc_not_tier1` | 0.533 (16/30) | 0.500 (15/30) | 0.400 (12/30) | 0.233 (7/30) | 0.500 (15/30) | 1.000 (30/30) |
| `ajo_newstyle` | 0.367 (11/30) | 0.500 (15/30) | 0.233 (7/30) | 1.000 (30/30) | 0.267 (8/30) | 1.000 (30/30) |
| `new_aep_wf_scrap` | 0.500 (15/30) | 0.233 (7/30) | 0.333 (10/30) | 1.000 (30/30) | 0.267 (8/30) | 0.567 (17/30) |
| `new_ajo_workflows` | 0.100 (3/30) | 0.233 (7/30) | 0.433 (13/30) | 0.967 (29/30) | 0.333 (10/30) | 0.067 (2/30) |

---

## Table L — OOD comparison: Bi vs Cross (best per dataset)

| Dataset | bi Milan acc | ce Milan acc | Δ Milan | bi AJO rate | ce AJO rate | Δ AJO |
|---|---|---|---|---|---|---|
| `aep_causal_cls34` | 0.5333 | 0.6000 | 0.0667 | 0.267 | 1.000 | 0.733 |
| `aep_dataset` | 0.8667 | 0.7000 | -0.1667 | 0.633 | 1.000 | 0.367 |
| `aep_causal_wf_v3` | 0.5333 | 0.6000 | 0.0667 | 0.233 | 1.000 | 0.767 |
| `ajo_doc_not_tier1` | 0.5333 | 0.5000 | -0.0333 | 0.967 | 1.000 | 0.033 |
| `ajo_newstyle` | 0.5333 | 0.5667 | 0.0333 | 0.700 | 1.000 | 0.300 |
| `new_aep_wf_scrap` | 0.8000 | 0.9000 | 0.1000 | 0.367 | 1.000 | 0.633 |
| `new_ajo_workflows` | 0.7333 | 0.6667 | -0.0667 | 0.567 | 0.967 | 0.400 |

*Full per-cell OOD comparison in [comparison_ood_biencoder_vs_crossencoder.csv](comparison_ood_biencoder_vs_crossencoder.csv).*

---

## CSV Files

| File | Contents |
|---|---|
| [test_accuracy_all.csv](test_accuracy_all.csv) | In-domain test metrics (acc, auc, f1), every run, both sweeps |
| [ood_30sample_all.csv](ood_30sample_all.csv) | OOD Milan + AJO 30-sample summary, both encoders |
| [ood_6step_workflow_milan.csv](ood_6step_workflow_milan.csv) | Milan 6-step predicted order per model, both encoders |
| [comparison_biencoder_vs_crossencoder.csv](comparison_biencoder_vs_crossencoder.csv) | Bi vs cross in-domain test acc/auc/f1 per (dataset, model) |
| [comparison_ood_biencoder_vs_crossencoder.csv](comparison_ood_biencoder_vs_crossencoder.csv) | Bi vs cross OOD Milan + AJO per (dataset, model) |
| [ood_milan_per_event.csv](ood_milan_per_event.csv) | Milan per-event ranks + tournament wins, both encoders |
| [ood_ajo_per_sample.csv](ood_ajo_per_sample.csv) | AJO per-sample predicted vs ground-truth order, both encoders |
| [per_tier_pair_all.csv](per_tier_pair_all.csv) | In-domain per-tier-pair accuracy, both sweeps |

*Generated by [build_analysis.py](build_analysis.py).*
