# evaluation_6 / reverse(y→x)

Inference-only evaluation with **exactly 1 negative per pair**, where the
negative is the *reverse* of the positive pair. Same two encoders and
same metric set as the parent [`evaluation_6/`](../README.md)
(`all-MiniLM-L6-v2`, `bert-base-uncased`) — the only difference is the
negative source.

For each pair `(a, p)` in the test split:

```
positive instance: (a, p)   → score = sim( enc(a) , enc(p) )    # forward (a→p)
negative instance: (p, a)   → score = sim( enc(p) , enc(a) )    # reverse (p→a)  ← label 0
```

The candidate pool per anchor is `{ p_i (gold), a_i (reverse) }` so
`n_candidates_per_query = 2` and `n_negatives = 1`. All other metrics
are computed exactly as in the parent eval.

## Models

| Short key           | HF id                                       | Pooling | Max seq len | Prefix |
|---------------------|---------------------------------------------|---------|------------:|--------|
| `all-MiniLM-L6-v2`  | `sentence-transformers/all-MiniLM-L6-v2`    | mean    |         256 | none   |
| `bert-base-uncased` | `google-bert/bert-base-uncased`             | CLS     |         512 | none   |

## Metrics (same set as `evaluation_6/results.json`)

| Metric         | Definition                                                                                  |
|----------------|---------------------------------------------------------------------------------------------|
| **auc**        | ROC-AUC over `2N` scores (label 1 for forward, 0 for reverse).                              |
| **precision_at_1** | Fraction of pairs where `pos > neg` (strict `>`; ties count as a loss).                |
| **mrr**        | Mean reciprocal rank of the gold (rank ∈ {1, 2}).                                           |
| **mean_rank / median_rank** | Aggregate rank of the gold in the 2-candidate pool.                            |
| **recall_at_k {1,3,5,10}**  | Fraction of golds with rank ≤ k.                                               |
| **hit_at_k**   | Same as `recall_at_k` (one gold per query).                                                 |
| **ties**       | Fraction of pairs where `pos == neg`.  *(diagnostic, not in parent eval)*                   |
| **mean_pos_sim / mean_neg_sim** | Mean cosine of forward / reverse direction.  *(diagnostic)*                |

**Tie-breaking convention.** Ranks use the same rule as
`evaluation_6.metrics_extra.random_pool_retrieval_metrics`:
`rank_i = #{distractors with sim STRICTLY > gold} + 1`. So a tie counts
as rank-1 (optimistic) for retrieval metrics, but as a loss for
`precision_at_1` (which uses strict `pos > neg`). This is exactly the
same inconsistency the parent eval has — kept on purpose so the two
JSONs are directly comparable.

## ⚠ Symmetric encoders ⇒ degenerate result

Both encoders are symmetric — same encoder, no per-side prefix — so
`sim(enc(a), enc(p)) ≡ sim(enc(p), enc(a))` *exactly* for every pair.
That means `pos_score == neg_score` everywhere, which is mathematically
forced by the encoders, not a measurement issue.

With the parent's tie-breaking convention this produces:

```
precision_at_1 (strict)  = 0.0        (no pair has pos > neg)
auc                      = 0.5        (identical score distributions)
ties                     = 1.0        (every pair)
rank_i                   = 1   ∀ i    (0 distractors strictly better)
mrr / recall@k / hit@k   = 1.0        (rank always 1)
```

The R@k = 1.0 / MRR = 1.0 numbers below are *not* a model success — they
are the optimistic-tie-breaking shadow of the same fact that
`precision_at_1 = 0`. To get any real signal on this task the encoder
has to be asymmetric (per-side prefix, untied weights, or a
cross-encoder).

## How to run

```bash
# default: 2 models × 6 datasets, batch_size 128, k=[1,3,5,10]
.venv/bin/python 'evaluation_6/reverse(y->x)/evaluate_reverse.py'

# subset
.venv/bin/python 'evaluation_6/reverse(y->x)/evaluate_reverse.py' \
    --models all-MiniLM-L6-v2 --datasets aep_causal qrecc

# quick smoke run capped at 1000 pairs per dataset
.venv/bin/python 'evaluation_6/reverse(y->x)/evaluate_reverse.py' --max-pairs 1000
```

All encoders run on GPU (CUDA when available). Each pair requires four
forward passes (`enc(a) × {q-side, d-side}`, `enc(p) × {q-side, d-side}`)
— for symmetric encoders the q-side and d-side are identical, so the
two halves coincide, but they are computed independently to keep the
code path identical to the asymmetric case.

## Results

Full test set for every dataset. Pool size = 2 (gold + 1 reverse-pair
negative).

### all-MiniLM-L6-v2 &nbsp;·&nbsp; `sentence-transformers/all-MiniLM-L6-v2` (mean pool, 384-d)

| dataset       |       N | pool | AUC    | P@1    | MRR    | R@1    | R@3    | R@5    | R@10   | Mean rank | Median rank | ties   |
|---------------|--------:|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|-------:|
| aep_causal    |     562 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| aep_followup  |   1,255 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| followupqg    |     501 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| multiwoz_v24  |   7,368 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| qrecc         |   5,204 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| workflow      | 260,487 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |

### bert-base-uncased &nbsp;·&nbsp; `google-bert/bert-base-uncased` (CLS pool, 768-d)

| dataset       |       N | pool | AUC    | P@1    | MRR    | R@1    | R@3    | R@5    | R@10   | Mean rank | Median rank | ties   |
|---------------|--------:|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|-------:|
| aep_causal    |     562 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| aep_followup  |   1,255 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| followupqg    |     501 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| multiwoz_v24  |   7,368 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| qrecc         |   5,204 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |
| workflow      | 260,487 |    2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |    1.0000 |         1.0 | 1.0000 |

Random baseline (2-candidate pool): R@k = min(k/2, 1) = 0.5 / 1.0 / 1.0 / 1.0 for k = 1/3/5/10; MRR = 0.75; mean rank = 1.5. Both encoders sit *above* this baseline on R@1/MRR purely because of the optimistic tie-breaking — the underlying signal (AUC, P@1) is at chance.

## Files

- [`evaluate_reverse.py`](evaluate_reverse.py) — runner. Encodes each text once with the (empty) anchor prefix and once with the (empty) positive prefix; computes pos vs reverse scores per pair and the full retrieval metric set.
- [`results.json`](results.json) — full output of the last run (config + per-cell metrics), with the same key set as `evaluation_6/results.json`.
- [`run.log`](run.log) — stdout from the last run.
