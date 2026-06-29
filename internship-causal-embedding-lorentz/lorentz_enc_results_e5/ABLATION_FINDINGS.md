# Lorentz-Enc: Ablation Findings (backbone vs method, and the time head)

This document isolates **where the improvement over the two-tower BERT baseline
actually comes from**, and reports an honest evaluation of the Lorentzian time head.

All MRR numbers are test-set, random-pool retrieval (pool = 562/501/1000/1000/1000
per dataset), seed 0, identical eval harness for every row — apples-to-apples.

---

## 1. Three-way comparison: backbone vs method

| Dataset | BERT two-tower (baseline) | e5-vanilla (K=4, fwd InfoNCE) | e5-Lorentz (K=7, bidir) |
|---------|--------------------------:|------------------------------:|------------------------:|
| aep_causal   | 0.3525 | **0.4089** (+16.0%) | 0.4056 (+15.1%) |
| followupqg   | 0.5844 | 0.6693 (+14.5%) | **0.6869** (+17.5%) |
| multiwoz_v24 | 0.2493 | **0.3229** (+29.5%) | 0.3201 (+28.4%) |
| qrecc        | 0.2758 | 0.3175 (+15.1%) | **0.3436** (+24.6%)¹ |
| workflow     | 0.5500 | — | **0.6345** (+15.4%) |

¹ qrecc e5-Lorentz used 52k training pairs; e5-vanilla used the 15k cap. Not
  strictly apples-to-apples on data size — the method delta for qrecc is inflated
  by the extra data.

### What this means

- **The backbone swap (BERT → e5-base-v2) is responsible for essentially all of
  the retrieval gain.** `e5-vanilla` — plain forward InfoNCE with K=4 in-batch +
  hard negatives, *no Lorentzian machinery at all* — already clears ≥10% on every
  dataset.
- **The "method" (bidirectional InfoNCE + K=7 + e5-mined negatives) adds little
  on equal footing.** On aep_causal and multiwoz it is within noise (±0.3 pt);
  on followupqg it helps modestly (+1.8 pt). The large qrecc delta is mostly the
  extra training data, not the method.

**Honest conclusion:** the headline "+20% over BERT" is a *backbone story*, not a
*Lorentzian-architecture story*. A well-trained e5 bi-encoder with good hard
negatives is what wins.

---

## 2. Does the Lorentzian time head earn its place?

The time head (now a deeper residual MLP: `hidden → 256 → [residual×3] → 32 →
LayerNorm`) was trained with an explicit **ordering loss** (push `t_positive >
t_anchor`, `t_negative < t_anchor`) plus the SteepSigmoid time term in the InfoNCE
(`β = 0.3`).

### Result A — Directional accuracy: **1.000** (both datasets)

For every test pair `(a, b⁺)`, the model scores the forward direction higher than
the reverse:

```
fwd = mean_k σ(τ·(t_b − t_a))   >   rev = mean_k σ(τ·(t_a − t_b))     for 100% of pairs
dir_margin (fwd − rev):  aep_causal +0.625,  followupqg +0.519
```

A pure cosine (space-only) scorer is **symmetric** — `score(a,b) = score(b,a)` —
so it has **no** directional capability by construction (dir-acc ≡ 0.5). The time
head genuinely breaks that symmetry and recovers cause→effect direction perfectly.
This is a real, interpretable capability the baseline lacks.

### Result B — Retrieval contribution: **none** (identical-pool test)

Ranking the *same* candidate pool by space-only vs space+time:

| Dataset | β=0.0 (space) | β=0.3 (space+time) | β=1.0 |
|---------|--------------:|-------------------:|------:|
| aep_causal | 0.4604 | 0.4590 | 0.4551 |
| followupqg | 0.7718 | 0.7724 | 0.7696 |

Adding the time term is neutral-to-slightly-harmful for retrieval MRR.

> ⚠️ An earlier run reported "combined MRR > space MRR". That was a **measurement
> artifact**: the combined metric used a different random candidate pool than the
> space-only metric. On *identical* pools (above) the effect vanishes.

### Why both can be true

The time head learned a consistent **per-pair** ordering (`t_b⁺` is reliably a hair
above `t_a`, which the steep sigmoid amplifies to dir-acc 1.0), but the dataset-level
magnitude is tiny (`mean(t_pos − t_anc) ≈ +0.005`) and **orthogonal to "which
candidate is the right one."** Retrieval is dominated by topical (space) similarity;
direction doesn't tell you *which* passage, only *which way* a known pair points.

**Conclusion:** the time head earns its place on a **directional / cause-vs-effect
task** (1.0 vs 0.5 for the symmetric baseline), **not** on retrieval. If the
deliverable is retrieval MRR, ship the space-only e5 bi-encoder. If the deliverable
is "does this model understand causal direction," the Lorentzian time head is the
component that provides it.

---

## 3. Reproduce

```bash
VENV=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python

# e5-vanilla baseline (isolates backbone)
CUDA_VISIBLE_DEVICES=0 $VENV scripts/train_lorentz.py --dataset all \
  --results-dir lorentz_enc_results_e5_vanilla --backbone intfloat/e5-base-v2 \
  --anchor-prefix "query: " --positive-prefix "passage: " --pooling mean \
  --space-dim 0 --max-negatives 4 --no-bidirectional --lambda-ord 0 --beta 0

# time head active (deeper head + ordering loss)
CUDA_VISIBLE_DEVICES=0 $VENV scripts/train_lorentz.py --dataset aep_causal \
  --results-dir lorentz_enc_results_e5_time --backbone intfloat/e5-base-v2 \
  --anchor-prefix "query: " --positive-prefix "passage: " --pooling mean \
  --space-dim 0 --time-dim 32 --time-hidden 256 --time-layers 3 \
  --lambda-ord 0.3 --beta 0.3 --select-on combined

# eval (reports directional accuracy + combined MRR when time head was trained)
CUDA_VISIBLE_DEVICES=0 $VENV scripts/evaluate_lorentz.py \
  --results-dir lorentz_enc_results_e5_time --dataset all
```
