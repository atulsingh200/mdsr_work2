# Architectural improvement, SAME base model — Cross-Encoder

**Constraint:** keep the base model identical to the BiEncoder baseline
(`google-bert/bert-base-uncased`) and improve the **architecture** to gain ≥10%.

**Result: achieved.** Swapping the two-tower bi-encoder for a **cross-encoder**
(same `bert-base-uncased`) gives **+11.3% N×N MRR** and **+59% Recall@1** on the
`aep_causal` test split.

## Headline (aep_causal test, 562 pairs, identical metric code)

| Metric | BERT BiEncoder (baseline) | **Cross-encoder (ours)** | Δ% |
|---|---|---|---|
| N×N MRR | 0.4033 | **0.4491** | **+11.3%** |
| Recall@1 | 0.1833 | **0.2918** | **+59.2%** |
| Recall@5 | 0.5712 | **0.6406** | +12.1% |
| Recall@10 | 0.7153 | 0.7633 | +6.7% |
| AUC | 0.9688 | 0.9736 | +0.5% |
| P@1 (vs 4 rand) | 0.9146 | 0.9288 | +1.6% |
| median rank | — | 3.0 | — |

Same `bert-base-uncased` backbone, same hard-negative mining, same eval code &
seed. The only change is the **architecture**.

## The architecture change

```
BiEncoder (baseline):                 Cross-encoder (this):
  anchor   → BERT_A → vec_a             [CLS] anchor [SEP] candidate [SEP]
  candidate→ BERT_B → vec_b                        │
  score = vec_a · vec_b                          BERT  (joint, full cross-attention)
  (towers never see each other)                    │
                                                 [CLS] → Linear → score
```

- **Reference:** Nogueira & Cho, 2019, *Passage Re-ranking with BERT* (monoBERT).
- **Why it helps:** the two texts attend to each other token-by-token, so the
  model can reason about *specific* causal links ("does this candidate answer
  *this* anchor?") instead of comparing two independently-pooled vectors. This
  is strictly more expressive than the bi-encoder, with the same weights.
- **Training:** listwise softmax / InfoNCE over each anchor's
  `[positive + 7 hard negatives + 4 random negatives]`; cross-entropy with the
  positive as target. Hard negatives = semantic kNN (MiniLM, as in
  `finetune_eval`) + anchor-similarity filter.

## What mattered (ablation)

| Run | max_len (combined) | tokens/text | negatives | val MRR | test N×N MRR |
|---|---|---|---|---|---|
| broken | 384 | ~190 | 7 hard | 0.369 | (below baseline) |
| **fixed** | **512** | **~256** | **7 hard + 4 random** | **0.553** | **0.4491 (+11.3%)** |

The single biggest fix: **`max_length=512`**. At 384 the combined pair gave each
text only ~190 tokens — *less* than the bi-encoder's 256 per text — so the
cross-encoder was handicapped and lost. With 512 (≈256 per text, a fair budget)
plus a mix of hard + random negatives, it clears the target comfortably.

## Tradeoff (honest)

A cross-encoder is a **re-ranker**: it must run BERT once per (anchor, candidate)
pair, so it can't be pre-indexed like a bi-encoder. In production you'd retrieve
top-k with the bi-encoder and re-rank that shortlist with this cross-encoder —
the standard two-stage IR pipeline — getting most of the +59% R@1 at a fraction
of the cost. Here we score the full 562×562 matrix to measure the architecture's
accuracy ceiling.

## Reproduce

```bash
PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY arch_improve/train_crossencoder.py \
  --dataset aep_causal --backbone google-bert/bert-base-uncased \
  --epochs 6 --batch-size 16 --k-neg 7 --n-random 4 --max-seq-length 512 \
  --lr 2e-5 --anchor-threshold 0.9 --val-max 300 --bf16
CUDA_VISIBLE_DEVICES=0 $PY arch_improve/evaluate_crossencoder.py --dataset aep_causal
```

## Other ideas explored / available next

- **Late interaction (ColBERT; Khattab & Zaharia 2020)** — token-level MaxSim,
  keeps indexability, between bi- and cross-encoder in power. Good next step if
  an indexable (not re-ranker) architecture is required.
- **Contextual Document Embeddings (CDE; Morris & Rush 2024)** — corpus-conditioned
  dual-encoder; same backbone, indexable, beats vanilla bi-encoders.
