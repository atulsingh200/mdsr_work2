# arch_improve — Architectural improvement on the *same* base model

**Question this answers:** can we beat the two-tower BiEncoder by **≥10%**
*without* changing the base model — only the architecture?

**Answer: yes.** Keeping `google-bert/bert-base-uncased` identical and switching
the architecture from a **bi-encoder** to a **cross-encoder (monoBERT)** gives
**+11.3% N×N MRR** and **+59% Recall@1** on `aep_causal`.

| Model (all `bert-base-uncased`) | N×N MRR | R@1 | R@5 | AUC |
|---|---|---|---|---|
| Two-tower BiEncoder (baseline) | 0.4033 | 0.1833 | 0.5712 | 0.9688 |
| **Cross-encoder (this dir)** | **0.4491** (+11.3%) | **0.2918** (+59%) | **0.6406** (+12%) | 0.9736 |

---

## 1. The architecture

```
BiEncoder (baseline)                     Cross-encoder (this)
────────────────────                     ────────────────────
anchor    → BERT_A → vec_a               [CLS] anchor [SEP] candidate [SEP]
candidate → BERT_B → vec_b                            │
score = vec_a · vec_b                          bert-base-uncased   ← ONE encoder,
(two towers, never interact)                          │              joint attention
                                              CLS (768-d)
                                                      │
                                          Dropout → Linear(768→1)
                                                      │
                                          relevance score  s(anchor, candidate)
```

- **Reference:** Nogueira & Cho, 2019, *Passage Re-ranking with BERT* (monoBERT).
- **Same weights, more expressive:** because both texts share one sequence, every
  anchor token attends to every candidate token in all 12 layers. The model
  reasons about the *specific* anchor→candidate link instead of comparing two
  independently-pooled vectors.
- **Base model is byte-for-byte the baseline's** (`google-bert/bert-base-uncased`).
  Only the *scoring architecture* changed.

---

## 2. Why an earlier cross-encoder attempt *lost* to the bi-encoder

There was a previous cross-encoder (`my_work/kl/cross_encoder/`, MRR **0.19** —
worse than the 0.40 two-tower, and it got *worse* with more epochs). Same
architecture, so what went wrong? **The training objective.**

| | Previous CE (MRR 0.19) | This CE (MRR 0.45) |
|---|---|---|
| **Loss** | **pointwise BCE** — each pair → {0,1} in isolation | **listwise softmax / InfoNCE** — positive must outrank its own negatives |
| `max_length` | 256 *combined* (≈128 tok/text) | 512 *combined* (≈256 tok/text) |
| Negatives | 4 hard | 7 hard + 4 random |
| Checkpoint selected on | val AUC / P@1 (pair classification) | val MRR (ranking) |

**The primary bug was BCE.** `nn.BCEWithLogitsLoss` pushes each pair's *absolute*
score toward 0 or 1 independently — it never compares the positive against the
negatives of the *same* anchor, which is exactly what ranking needs. Worse, a
hard negative is near-identical to the positive, so BCE forces near-identical
inputs to opposite labels → conflicting, noisy gradients. The result: the model
got better at *pair classification* (val P@1 0.28→0.41) while getting worse at
*ranking* (10-epoch MRR 0.170 < 3-epoch MRR 0.191) — a textbook objective
mismatch, made worse by selecting the checkpoint on a classification metric.

**The fix:** train it as a *ranker*. For each anchor, score
`[positive, neg₁, …, neg_K]` together and apply cross-entropy with the positive
as the target (listwise softmax). Now the gradient says "make the positive's
score higher than its negatives'", which is the eval task. Combined with the
`max_length 256→512` fix (give each text a fair token budget), MRR went
0.19 → 0.45.

---

## 3. Files

```
arch_improve/
├── train_crossencoder.py     # model + listwise-softmax training + per-epoch val
├── evaluate_crossencoder.py  # full N×N score matrix → MRR / R@K / AUC / P@1 vs baseline
├── RESULTS.md                # detailed results + ablation
├── README.md                 # this file
└── results/<dataset>_crossencoder/   # checkpoints + metrics (gitignored)
```

- **Hard negatives**: semantic k-NN (MiniLM, same as `finetune_eval`) **plus an
  anchor-similarity filter** — a candidate negative is dropped if
  `cos(anchor_i, anchor_j) > 0.9` (its positive may be a true effect of anchor_i,
  i.e. a false negative). Reuses `gnn/negatives.py`.
- **Metrics** are computed from the cross-encoder score matrix and match the
  baseline exactly (same code path / seed) for an apples-to-apples comparison.

---

## 4. How to run

```bash
PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python

# Train (≈35 min on one A100; mines hard negatives on first run, then caches)
CUDA_VISIBLE_DEVICES=0 $PY arch_improve/train_crossencoder.py \
  --dataset aep_causal --backbone google-bert/bert-base-uncased \
  --epochs 6 --batch-size 16 --k-neg 7 --n-random 4 --max-seq-length 512 \
  --lr 2e-5 --anchor-threshold 0.9 --val-max 300 --bf16

# Evaluate on the test split (scores the full 562×562 matrix), prints Δ% vs baseline
CUDA_VISIBLE_DEVICES=0 $PY arch_improve/evaluate_crossencoder.py --dataset aep_causal
```

Key flags: `--k-neg` / `--n-random` (hard + random negatives per anchor),
`--max-seq-length 512` (**important** — do not drop below 512), `--val-max`
(cap val anchors for faster per-epoch validation; final test eval always uses
the full split).

---

## 5. Tradeoff & next steps

A cross-encoder is a **re-ranker**: it runs BERT once per (anchor, candidate)
pair, so unlike the bi-encoder it can't be pre-indexed (O(N·pool) BERT forwards
vs O(N+pool) for embeddings). In production: retrieve top-k with the bi-encoder,
then re-rank that shortlist with this cross-encoder — the standard two-stage IR
pipeline captures most of the +59% R@1 at a fraction of the cost.

If an **indexable** (single-vector) architecture is required, the next ideas to
try on the same backbone are:
- **ColBERT** (Khattab & Zaharia, 2020) — token-level late interaction (MaxSim),
  keeps indexability, sits between bi- and cross-encoder in power.
- **CDE / Contextual Document Embeddings** (Morris & Rush, 2024) — corpus-conditioned
  dual-encoder; same backbone, indexable, beats vanilla bi-encoders.

## Memory safety

All scripts call `check_memory()` before heavy steps and abort if CPU RAM drops
below 4 GB (see the repo `CLAUDE.md`).
