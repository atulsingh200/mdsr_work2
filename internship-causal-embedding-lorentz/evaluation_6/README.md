# evaluation_6

Inference-only evaluation of off-the-shelf encoders on every dataset under
[`data_6/`](../data_6/). No fine-tuning — each model is loaded from
HuggingFace, run forward once over the anchors + positives in each test
split, and scored.

## What it computes

Per (model, dataset):

| Metric | Definition |
|---|---|
| **AUC** | ROC-AUC over `(N+1)` similarity scores per anchor (1 positive + N random negatives). |
| **P@1** | Precision@1 — fraction of anchors where `sim(a, pos) > sim(a, neg_i)` for *every* sampled negative. With `N=4`, random baseline = `1/5 = 0.20`. |
| **MRR** | Mean Reciprocal Rank of the gold positive in a per-anchor candidate pool. |
| **Recall@K** | Fraction of golds found in top-K (== Hit@K here since 1 gold per query). |
| **Mean / Median rank** | Average rank of the gold across queries. |

### Methodology — full anchors, capped candidate pool

Every anchor in the test split is evaluated (no subsampling on the query
side). For the retrieval metrics, each anchor is ranked against a **random
candidate pool** of `--candidate-pool-size` positives (default 1000):
the anchor's gold positive plus `pool_size - 1` other positives drawn at
random (with replacement; collisions are rare for `pool_size ≪ N`). This
gives O(N · pool_size) work instead of O(N²), so even workflow's 260k-row
test split is tractable.

For datasets smaller than `pool_size`, the pool collapses to the full
test split — i.e. true same-split retrieval.

All similarities are computed on GPU (CUDA when available).

## How to run

```bash
.venv/bin/python evaluation_6/evaluate_inference.py                           # default: 2 models × 6 datasets, pool=1000
.venv/bin/python evaluation_6/evaluate_inference.py --candidate-pool-size 500 # tighter pool
.venv/bin/python evaluation_6/evaluate_inference.py --n-negatives 9           # P@1 vs 9 negatives
.venv/bin/python evaluation_6/evaluate_inference.py \
    --models all-MiniLM-L6-v2 --datasets aep_causal qrecc                     # subset
```

Default settings: `--candidate-pool-size 1000`, `--n-negatives 4`, `--k 1 3 5 10`, `--seed 0`, `--batch-size 128`.

## Results

Full test set used for every dataset. Candidate pool capped at 1000 per anchor (or the dataset size, whichever is smaller). AUC / P@1 use 4 random negatives.

### all-MiniLM-L6-v2 &nbsp;·&nbsp; `sentence-transformers/all-MiniLM-L6-v2` (mean pool, 384-d)

| dataset       |       N | pool | AUC    | P@1    | MRR    | R@1    | R@3    | R@5    | R@10   | Mean rank | Median rank |
|---------------|--------:|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|
| aep_causal    |     562 |  562 | 0.9371 | 0.8754 | 0.2943 | 0.1477 | 0.3915 | 0.4573 | 0.5943 |     30.74 |         7.0 |
| aep_followup  |   1,255 | 1000 | 0.8435 | 0.7498 | 0.1853 | 0.1139 | 0.2024 | 0.2510 | 0.3251 |    112.45 |        30.0 |
| followupqg    |     501 |  501 | 0.9795 | 0.9441 | 0.6340 | 0.4631 | 0.7565 | 0.8303 | 0.8822 |     10.70 |         2.0 |
| multiwoz_v24  |   7,368 | 1000 | 0.6706 | 0.4053 | 0.0644 | 0.0390 | 0.0617 | 0.0790 | 0.1008 |    340.16 |       249.0 |
| qrecc         |   5,204 | 1000 | 0.8520 | 0.7100 | 0.2350 | 0.1585 | 0.2602 | 0.3088 | 0.3711 |    132.99 |        31.0 |
| workflow      | 260,487 | 1000 | 0.8994 | 0.8046 | 0.3987 | 0.3109 | 0.4337 | 0.4936 | 0.5672 |     95.50 |         6.0 |

### bert-base-uncased &nbsp;·&nbsp; `google-bert/bert-base-uncased` (CLS pool, 768-d)

Raw MLM weights — not a sentence encoder, included as a sanity-check lower bound.

| dataset       |       N | pool | AUC    | P@1    | MRR    | R@1    | R@3    | R@5    | R@10   | Mean rank | Median rank |
|---------------|--------:|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|
| aep_causal    |     562 |  562 | 0.7294 | 0.5338 | 0.0927 | 0.0302 | 0.1085 | 0.1406 | 0.1993 |    130.02 |        70.0 |
| aep_followup  |   1,255 | 1000 | 0.6128 | 0.4143 | 0.0402 | 0.0167 | 0.0343 | 0.0510 | 0.0749 |    345.95 |       243.0 |
| followupqg    |     501 |  501 | 0.6350 | 0.4611 | 0.0934 | 0.0439 | 0.1078 | 0.1218 | 0.1537 |    142.64 |        86.0 |
| multiwoz_v24  |   7,368 | 1000 | 0.5472 | 0.2782 | 0.0190 | 0.0058 | 0.0143 | 0.0209 | 0.0347 |    436.67 |       415.0 |
| qrecc         |   5,204 | 1000 | 0.5400 | 0.4450 | 0.0675 | 0.0348 | 0.0657 | 0.0886 | 0.1278 |    311.73 |       213.0 |
| workflow      | 260,487 | 1000 | 0.7817 | 0.6273 | 0.1660 | 0.1131 | 0.1733 | 0.2073 | 0.2625 |    179.07 |        70.0 |

### bge-m3 &nbsp;·&nbsp; `BAAI/bge-m3` (CLS pool, 1024-d)

| dataset       |       N | pool | AUC    | P@1    | MRR    | R@1    | R@3    | R@5    | R@10   | Mean rank | Median rank |
|---------------|--------:|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|
| aep_causal    |     562 |  562 | 0.9383 | 0.8683 | 0.3091 | 0.1548 | 0.3843 | 0.4822 | 0.6157 |     29.64 |         6.0 |
| aep_followup  |   1,255 | 1000 | 0.8712 | 0.7530 | 0.1829 | 0.1092 | 0.1873 | 0.2406 | 0.3243 |    109.51 |        29.0 |
| followupqg    |     501 |  501 | 0.9649 | 0.9222 | 0.5484 | 0.3752 | 0.6906 | 0.7525 | 0.8184 |     17.11 |         2.0 |
| multiwoz_v24  |   7,368 | 1000 | 0.7391 | 0.4958 | 0.0873 | 0.0516 | 0.0871 | 0.1126 | 0.1488 |    260.06 |       157.0 |
| qrecc         |   5,204 | 1000 | 0.8749 | 0.7371 | 0.2306 | 0.1508 | 0.2519 | 0.3065 | 0.3868 |    112.44 |        24.0 |
| workflow      | 260,487 | 1000 | 0.8864 | 0.7863 | 0.3632 | 0.2800 | 0.3965 | 0.4497 | 0.5221 |    103.58 |         9.0 |

### e5-base-v2 &nbsp;·&nbsp; `intfloat/e5-base-v2` (mean pool, 768-d, `"query: "` / `"passage: "` prefixes)

| dataset       |       N | pool | AUC    | P@1    | MRR    | R@1    | R@3    | R@5    | R@10   | Mean rank | Median rank |
|---------------|--------:|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|
| aep_causal    |     562 |  562 | 0.9289 | 0.8701 | 0.3051 | 0.1530 | 0.3843 | 0.4804 | 0.6050 |     30.64 |         6.0 |
| aep_followup  |   1,255 | 1000 | 0.8644 | 0.7514 | 0.1693 | 0.0932 | 0.1745 | 0.2382 | 0.3267 |    115.85 |        30.0 |
| followupqg    |     501 |  501 | 0.9819 | 0.9601 | 0.6840 | 0.5090 | 0.8423 | 0.8822 | 0.9182 |      9.23 |         1.0 |
| multiwoz_v24  |   7,368 | 1000 | 0.7440 | 0.5233 | 0.1047 | 0.0641 | 0.1095 | 0.1337 | 0.1726 |    249.34 |       131.0 |
| qrecc         |   5,204 | 1000 | 0.8863 | 0.7508 | 0.2561 | 0.1726 | 0.2911 | 0.3372 | 0.4051 |     96.06 |        24.0 |
| workflow      | 260,487 | 1000 | 0.8914 | 0.7942 | 0.3632 | 0.2757 | 0.3988 | 0.4564 | 0.5322 |    100.77 |         8.0 |

Random baseline (1 gold in pool of 1000): `R@K ≈ K/1000`, i.e. `R@1 ≈ 0.001`, `R@10 ≈ 0.01`.

## Files

- [`evaluate_inference.py`](evaluate_inference.py) — main runner. Loads each model once and re-uses it across datasets; similarity on GPU.
- [`metrics_extra.py`](metrics_extra.py) — `auc_precision_at_1_with_random_negatives()` and `random_pool_retrieval_metrics()` (torch / GPU-accelerated).
- [`results.json`](results.json) — full output of the last run (config + per-cell metrics).
