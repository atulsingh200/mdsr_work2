# CDEv2 + Cross-Encoder Rerank — Full Reproduction Pipeline

The exact end-to-end recipe that beat the 2-tower **BiEncoder baseline by +12.9% MRR**
on `aep_causal` (562 test pairs), using the same base model
(`google-bert/bert-base-uncased`). Every command below was run for `aep_causal`.

> **To run on another dataset:** replace `aep_causal` with one of
> `followupqg`, `multiwoz_v24`, `qrecc`, `workflow` everywhere. The whole thing
> is also wrapped in one driver — see [§ One command](#one-command-for-any-dataset).

All commands run from the repo root, using the project venv:

```bash
cd /mnt/localssd/internship-causal-embedding-merge-followup
PY=.venv/bin/python      # use the venv python for everything
```

---

## The pipeline at a glance

```
                 (1)                   (2)              (3)                  (4)              (5)
          BiEncoder baseline  →  mine MiniLM negs  →  CDEv2 retriever  →  Cross-Encoder  →  rerank-eval
          (2-tower, InfoNCE)      (K=4 train/test)     (warm-started)       (reranker)      (val→test)
                 │                                          │                   │                │
            checkpoint_best.pt                       cde_v2_run5c_best.pt   checkpoint_best.pt   rerank_eval.json
                 │                                          │                   │
                 └──────────── warm-start ──────────────────┘            reranks top-50
```

- **Step 1** trains the baseline we must beat (and whose weights warm-start CDEv2).
- **Step 2** mines hard negatives (semantically similar but causally-wrong) for training.
- **Step 3** trains the CDEv2 single-vector retriever (Gaussian + causal-transport, `−KL/D + λ·cosine`).
- **Step 4** trains a BERT cross-encoder that jointly reads `[CLS] A [SEP] B`.
- **Step 5** reranks the retriever's top-K with the cross-encoder; config picked on val, reported on test.

---

## Step 1 — Train the BiEncoder baseline

Two-tower (untied anchor/positive encoders), InfoNCE with in-batch negatives +
learnable temperature. This is the model we must beat, **and** its weights
warm-start CDEv2 in Step 3.

```bash
$PY finetune_eval/train_finetune.py --dataset aep_causal \
    --backbone google-bert/bert-base-uncased --pooling cls \
    --epochs 3 --batch-size 32 --lr 2e-5 --max-seq-length 256 --seed 42
```

**Writes:** `finetune_eval/results/aep_causal/checkpoint_best.pt` (+ `config.json`).

> Epochs used per dataset in the original runs: `aep_causal:3  followupqg:3
> multiwoz_v24:2  qrecc:2  workflow:2` (small datasets get more epochs).

Evaluate the baseline (the number to beat):

```bash
$PY finetune_eval/evaluate_finetune.py --dataset aep_causal
```

→ **BiEncoder MRR = 0.4033**, R@1 0.256, R@10 0.715, AUC(hard-neg) 0.697.

---

## Step 2 — Mine hard negatives

A frozen `all-MiniLM-L6-v2` retrieves the top-K most-similar *wrong* positives
for each anchor — the confusables the model must learn to reject.

```bash
# K=4 negatives for train + test (used by the BiEncoder + cross-encoder)
$PY finetune_eval/mine_hard_negatives.py --dataset aep_causal --split train --k 4 --no-cap
$PY finetune_eval/mine_hard_negatives.py --dataset aep_causal --split test  --k 4 --no-cap

# K=8 negatives for the CDEv2 retriever (writes to my_work/kl/results/aep_causal/)
$PY my_work/kl/v2/mine_hard_neg_v2.py --dataset aep_causal --split both --k 8
```

**Writes:** `hard_negatives.npy` (K=4) under `finetune_eval/results/aep_causal/`,
and `hard_negatives_k8.npy` / `test_hard_negatives_k8.npy` under
`my_work/kl/results/aep_causal/`.

---

## Step 3 — Train the CDEv2 retriever (warm-started)

CDEv2 = Gaussian distribution embeddings + causal transport, scored with
`−KL(N_B ‖ T(N_A))/D + λ·cosine`. **Warm-started from the Step-1 BiEncoder**
(`mu-identity` + large initial σ make KL≈0 at init, so it reproduces the
baseline, then fine-tunes up). Winning config = **run5c**: pure InfoNCE, all
3000 steps, no aux losses.

```bash
cd my_work/kl
$PY v2/train_v2.py --dataset aep_causal \
   --init-from-biencoder ../../finetune_eval/results/aep_causal/checkpoint_best.pt \
   --proj-dim 768 --pooling cls --mu-identity --log-sigma-init 3.0 \
   --kl-scale dim --init-cos-weight 1.0 \
   --backbone-lr 2e-5 --batch-size 64 --max-seq-length 256 \
   --phase-a-steps 3000 --phase-b-steps 0 --total-steps 3000 \
   --lambda-entropy 0 --lambda-antisym 0 --lambda-bpr 0 \
   --hard-neg-file results/aep_causal/hard_negatives_k8.npy \
   --out-suffix _run5c
cd ..
```

**Writes:** `my_work/kl/results/aep_causal/cde_v2_run5c_best.pt`.
→ **CDEv2 stage-1 MRR = 0.4094** (single-vector ceiling ~0.41).

---

## Step 4 — Train the cross-encoder reranker

`[CLS] text_A [SEP] text_B → BERT → Linear(1)`, BCE loss, trained on the Step-2
hard negatives. Its joint token attention is what disambiguates the hard negatives.

```bash
$PY my_work/kl/cross_encoder/train_ce.py --dataset aep_causal \
    --data-dir finetune_eval/results \
    --epochs 4 --out-dir my_work/kl/cross_encoder/results/aep_causal_rerank
```

**Writes:** `my_work/kl/cross_encoder/results/aep_causal_rerank/checkpoint_best.pt`.

> **The extra lever (2-CE ensemble → the full +12.9%):** the headline result
> used a *second* cross-encoder trained on in-domain CDEv2-retrieved negatives,
> z-scored and averaged with the first. A single CE gives ≈ +6–7%; the 2-CE
> ensemble pushed it to +12.9%. To reproduce, mine in-domain negatives from the
> run5c retriever into `cross_encoder/data_indomain/aep_causal/` and train a
> second CE with `--data-dir cross_encoder/data_indomain --epochs 6 --out-dir
> .../aep_causal_rerank_v2`, then pass **both** dirs to `--ce-dir` in Step 5.

---

## Step 5 — Retrieve-then-rerank evaluation

Rerank the retriever's top-`pool` candidates with the cross-encoder:
`final = blend·CE_z + (1−blend)·retrieval_z`. **Config selected on val**, then
**reported on test** (no test tuning).

```bash
cd my_work/kl
# 5a. tune (pool, blend) on VAL
$PY v2/rerank_eval.py --dataset aep_causal --split val \
   --cde-suffix _run5c \
   --ce-dir cross_encoder/results/aep_causal_rerank \
   --pool 20 50 --blend 0.3 0.5 0.7

# 5b. report on TEST (best val config was pool=50, blend=0.5)
$PY v2/rerank_eval.py --dataset aep_causal --split test \
   --cde-suffix _run5c \
   --ce-dir cross_encoder/results/aep_causal_rerank \
   --pool 50 --blend 0.5
cd ..
```

**Writes:** `my_work/kl/results/aep_causal/rerank_eval.json`.

For the **2-CE ensemble**, pass both CE dirs (z-scored + averaged automatically):

```bash
   --ce-dir cross_encoder/results/aep_causal_rerank \
            cross_encoder/results/aep_causal_rerank_v2
```

---

## Headline result (aep_causal)

| System | MRR | R@1 | R@10 | AUC(hard-neg) |
|--------|-----|-----|------|---------------|
| BiEncoder (baseline) | 0.4033 | 0.256 | 0.715 | 0.697 |
| CDEv2 stage-1 (retriever) | 0.4094 | 0.260 | 0.724 | 0.729 |
| **CDEv2 + CE-rerank (2-CE)** | **0.4555** | **0.310** | **0.754** | **0.834** |
| **Lift over baseline** | **+12.9%** | **+21%** | **+5.5%** | **+19.7%** |

---

## One command for any dataset

All five steps above (3–5, plus data staging + K=8 mining) are wrapped in a
single driver. Step 1 (BiEncoder) and Step 2's K=4 mining are prerequisites and
already done for all 5 datasets under `finetune_eval/results/<dataset>/`:

```bash
# run ONE dataset at a time (watch GPU usage; avoid CPU OOM)
.venv/bin/python my_work/kl/v2/run_pipeline.py --dataset followupqg --gpu 0
```

Swap `--dataset` for `multiwoz_v24`, `qrecc`, `workflow`, or `aep_causal`.
The driver stages data, holds out 10% of train as val (for datasets without a
val split), mines K=8 negs, trains CDEv2 + one cross-encoder, tunes on val, and
reports test MRR + the rerank lift.

> The driver runs the **single-CE** version (≈ +6–7%). For the full **2-CE
> +12.9%** recipe, add the second in-domain cross-encoder as described in Step 4.

### If you want to re-run Step 1 + 2 for a brand-new dataset

```bash
$PY finetune_eval/train_finetune.py --dataset <NEW> --epochs 3 --batch-size 32
$PY finetune_eval/mine_hard_negatives.py --dataset <NEW> --split train --k 4 --no-cap
$PY finetune_eval/mine_hard_negatives.py --dataset <NEW> --split test  --k 4 --no-cap
.venv/bin/python my_work/kl/v2/run_pipeline.py --dataset <NEW> --gpu 0
```

(The dataset must be registered in `finetune_eval/datasets.py`.)
