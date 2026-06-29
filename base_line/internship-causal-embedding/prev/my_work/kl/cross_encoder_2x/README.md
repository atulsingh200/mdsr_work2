# CrossEncoder-2x — parameter-matched cross-encoder vs. BiEncoder

This directory implements a **parameter-matched cross-encoder** designed for a fair
architectural comparison against the two-tower BiEncoder baseline.

## Motivation

The BiEncoder (`src/biencoder/`) uses **two untied BERT-base towers** — one for
anchors, one for positives — giving ~218 M total parameters.  The 1x cross-encoder
(`cross_encoder/`) uses a single BERT-base (~109 M params), so any performance gap
could be partly explained by the 2× parameter advantage of the BiEncoder.

This experiment closes that gap by using a **24-layer BERT-base cross-encoder**
(~195 M params), initialized from stacked bert-base-uncased weights, so the
comparison is closer to apples-to-apples.

## Parameter budget

| Model                          | Params      | Note                                   |
|--------------------------------|-------------|----------------------------------------|
| BiEncoder (2× BERT-base)       | 218,964,480 | two untied towers, each with embeddings|
| **CrossEncoder-2x (24-layer)** | 194,537,473 | single encoder, shared embeddings      |
| CrossEncoder-1x (12-layer)     | 109,482,240 | baseline cross-encoder                 |

The ~24 M gap between 2× BERT-base and the 24-layer model is exactly the duplicated
embedding table the BiEncoder carries (each tower has its own 23.8 M embedding
matrix).  The transformer body of the 24-layer model is exactly 2× that of
BERT-base.

## Architecture

```
[CLS] anchor [SEP] candidate [SEP]
        │
   BERT-24L (stacked-init)
        │
   [CLS] vector  →  Dropout  →  Linear(768→1)  →  logit
```

**Weight initialisation:** layer i in \[0..23\] is copied from pretrained
`google-bert/bert-base-uncased` layer `(i % 12)`.  Embeddings and pooler are
copied directly.  This is the standard "layer stacking" warm-start used in the
literature (e.g. ALBERT layer-sharing studies).

## Setup

Data: `finetune_eval/results/<dataset>/` — same `train/val/test_pairs.jsonl`
and `hard_negatives.npy` files used by the 1x cross-encoder runs.

## Training

```bash
PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
CE2X=/mnt/localssd/internship-causal-embedding-merge-followup/my_work/kl/cross_encoder_2x
DATA=/mnt/localssd/internship-causal-embedding-merge-followup/finetune_eval/results

# Small / medium datasets (same settings as 1x CE)
CUDA_VISIBLE_DEVICES=0 $PY $CE2X/train_ce2x.py --dataset aep_causal \
    --data-dir $DATA --epochs 10 --batch-size 128 --val-batch-size 256 \
    --max-length 256 --num-workers 2 \
    --early-stop-patience 2 --early-stop-threshold 0.001

CUDA_VISIBLE_DEVICES=1 $PY $CE2X/train_ce2x.py --dataset followupqg \
    --data-dir $DATA --epochs 10 --batch-size 128 --val-batch-size 256 \
    --max-length 256 --num-workers 2 \
    --early-stop-patience 2 --early-stop-threshold 0.001

CUDA_VISIBLE_DEVICES=2 $PY $CE2X/train_ce2x.py --dataset multiwoz_v24 \
    --data-dir $DATA --epochs 10 --batch-size 128 --val-batch-size 256 \
    --max-length 256 --num-workers 2 \
    --early-stop-patience 2 --early-stop-threshold 0.001

CUDA_VISIBLE_DEVICES=3 $PY $CE2X/train_ce2x.py --dataset qrecc \
    --data-dir $DATA --epochs 10 --batch-size 128 --val-batch-size 256 \
    --max-length 256 --num-workers 2 \
    --early-stop-patience 2 --early-stop-threshold 0.001

# Workflow (large — 3 ep, shorter max_length)
CUDA_VISIBLE_DEVICES=0 $PY $CE2X/train_ce2x.py --dataset workflow \
    --data-dir $DATA --epochs 3 --batch-size 256 --val-batch-size 512 \
    --max-length 128 --num-workers 2 \
    --early-stop-patience 1 --early-stop-threshold 0.005
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 $PY $CE2X/evaluate_ce2x.py --dataset aep_causal   --data-dir $DATA --pool-size 1000 --max-queries 2000 --batch-size 64 --max-length 256
CUDA_VISIBLE_DEVICES=1 $PY $CE2X/evaluate_ce2x.py --dataset followupqg   --data-dir $DATA --pool-size 1000 --max-queries 2000 --batch-size 64 --max-length 256
CUDA_VISIBLE_DEVICES=2 $PY $CE2X/evaluate_ce2x.py --dataset multiwoz_v24 --data-dir $DATA --pool-size 1000 --max-queries 2000 --batch-size 64 --max-length 256
CUDA_VISIBLE_DEVICES=3 $PY $CE2X/evaluate_ce2x.py --dataset qrecc        --data-dir $DATA --pool-size 1000 --max-queries 2000 --batch-size 64 --max-length 256
CUDA_VISIBLE_DEVICES=0 $PY $CE2X/evaluate_ce2x.py --dataset workflow     --data-dir $DATA --pool-size 1000 --max-queries 2000 --batch-size 64 --max-length 128
```

## Training summary

| Dataset      | Best val AUC | Best val P@1 | Best epoch | Stopped early? |
|--------------|-------------:|-------------:|:----------:|:--------------:|
| aep_causal   | 0.6680       | 0.3470       | 1 / 10     | yes (epoch 3)  |
| followupqg   | 0.9812       | 0.9424       | 2 / 10     | yes (epoch 4)  |
| multiwoz_v24 | 0.8578       | 0.6080       | 4 / 10     | yes (epoch 6)  |
| qrecc        | 0.9243       | 0.7627       | 2 / 10     | yes (epoch 4)  |
| workflow     | 0.9466       | 0.8490       | 2 / 3      | no             |

## Test-set results

### CrossEncoder-2x (this work, ~195 M params, 24 layers)

| Dataset      | N test  | AUC(rand) | P@1(rand) | AUC(HN) | P@1(HN) | MRR    | R@1   | R@3   | R@5   | R@10  | Median |
|--------------|--------:|----------:|----------:|--------:|--------:|-------:|------:|------:|------:|------:|-------:|
| aep_causal   |     562 | 0.8676    | 0.7420    | 0.6590  | 0.3541  | 0.2707 | 0.167 | 0.306 | 0.367 | 0.482 |   12.0 |
| followupqg   |     501 | 0.9843    | 0.9641    | 0.9254  | 0.8483  | 0.7537 | 0.661 | 0.828 | 0.868 | 0.900 |    1.0 |
| multiwoz_v24 |   7,368 | 0.8686    | 0.6469    | 0.6966  | 0.4228  | 0.1149 | 0.064 | 0.117 | 0.151 | 0.212 |   78.0 |
| qrecc        |   5,204 | 0.8911    | 0.7125    | 0.7017  | 0.4349  | 0.2340 | 0.171 | 0.248 | 0.286 | 0.346 |   39.0 |
| workflow     | 260,487 | 0.9462    | 0.8340    | 0.7504  | 0.4930  | 0.4413 | 0.366 | 0.473 | 0.519 | 0.581 |    5.0 |

Retrieval columns (MRR / R@K / Median) on multiwoz_v24, qrecc, workflow use
`max_queries=2000` test anchors (same cap as the 1x CE).

### CrossEncoder-1x baseline (~109 M params, 12 layers, from `cross_encoder/`)

| Dataset      | N test  | AUC(rand) | P@1(rand) | AUC(HN) | P@1(HN) | MRR    | R@1   | R@3   | R@5   | R@10  | Median |
|--------------|--------:|----------:|----------:|--------:|--------:|-------:|------:|------:|------:|------:|-------:|
| aep_causal   |     562 | 0.8453    | 0.6957    | 0.7113  | 0.4484  | 0.1914 | 0.110 | 0.212 | 0.265 | 0.345 |   28.0 |
| followupqg   |     501 | 0.9711    | 0.9421    | 0.9252  | 0.8543  | 0.7305 | 0.649 | 0.780 | 0.830 | 0.874 |    1.0 |
| multiwoz_v24 |   7,368 | 0.7823    | 0.5050    | 0.7444  | 0.4673  | 0.0788 | 0.044 | 0.075 | 0.094 | 0.133 |  156.0 |
| qrecc        |   5,204 | 0.8552    | 0.6489    | 0.6943  | 0.4262  | 0.2084 | 0.155 | 0.221 | 0.254 | 0.304 |   72.0 |
| workflow     | 260,487 | 0.9403    | 0.8131    | 0.8801  | 0.6692  | 0.3276 | 0.259 | 0.346 | 0.390 | 0.450 |   16.0 |

### BiEncoder baseline (~219 M params, 2× BERT-base, from `finetune_eval/`)

| Dataset      | N test  | AUC(rand) | P@1(rand) | AUC(HN) | P@1(HN) | MRR    | R@1   | R@3   | R@5   | R@10  | Median |
|--------------|--------:|----------:|----------:|--------:|--------:|-------:|------:|------:|------:|------:|-------:|
| aep_causal   |     562 | 0.9712    | 0.9128    | 0.7157  | 0.5178  | 0.3525 | 0.164 | 0.457 | 0.585 | 0.715 |    4.0 |
| followupqg   |     501 | 0.9827    | 0.9541    | 0.8370  | 0.7365  | 0.5844 | 0.403 | 0.731 | 0.806 | 0.854 |    2.0 |
| multiwoz_v24 |   7,368 | 0.9526    | 0.8694    | 0.6223  | 0.4110  | 0.2493 | 0.161 | 0.264 | 0.327 | 0.426 |   16.0 |
| qrecc        |   5,204 | 0.9227    | 0.8113    | 0.6365  | 0.4137  | 0.2758 | 0.182 | 0.306 | 0.365 | 0.457 |   14.0 |
| workflow     | 260,487 | 0.9662    | 0.9104    | 0.6883  | 0.4763  | 0.5500 | 0.466 | 0.593 | 0.643 | 0.706 |    2.0 |

### Head-to-head deltas: CE-2x vs CE-1x (2x − 1x)

| Dataset      | ΔAUC(rand) | ΔAUC(HN)  | ΔP@1(HN)   | ΔMRR    | ΔR@1   | Winner (overall) |
|--------------|----------:|-----------:|-----------:|--------:|-------:|:-----------------|
| aep_causal   | +0.022    | −0.052     | **−0.094** | +0.079  | +0.057 | mixed            |
| followupqg   | +0.013    | **+0.000** | **−0.006** | +0.023  | +0.012 | CE-2x on rand+MRR|
| multiwoz_v24 | **+0.086**| −0.048     | **−0.044** | +0.036  | +0.020 | CE-2x on rand+MRR|
| qrecc        | +0.036    | **+0.007** | +0.009     | +0.026  | +0.016 | **CE-2x sweep**  |
| workflow     | +0.006    | **−0.130** | **−0.176** | +0.114  | +0.107 | CE-2x on rand+MRR|

### Head-to-head deltas: CE-2x vs BiEncoder (CE-2x − BE)

| Dataset      | ΔAUC(rand) | ΔAUC(HN)   | ΔP@1(HN)   | ΔMRR    | ΔR@1    | Winner           |
|--------------|----------:|-----------:|-----------:|--------:|--------:|:-----------------|
| aep_causal   | −0.104    | −0.057     | −0.164     | −0.082  | +0.003  | BE (sweep)       |
| followupqg   | +0.002    | **+0.088** | **+0.112** | +0.169  | +0.258  | **CE-2x sweep**  |
| multiwoz_v24 | −0.084    | **+0.074** | +0.012     | −0.134  | −0.097  | mixed            |
| qrecc        | −0.032    | **+0.065** | +0.021     | −0.042  | −0.011  | mixed            |
| workflow     | −0.020    | **+0.062** | +0.017     | −0.109  | −0.100  | mixed            |

## Findings

### 1. Extra depth helps retrieval — CE-2x beats CE-1x on MRR across all datasets.

Doubling the transformer layers consistently improves pool retrieval:

| Dataset      | CE-1x MRR | CE-2x MRR | Δ       |
|--------------|----------:|----------:|--------:|
| aep_causal   | 0.1914    | 0.2707    | **+0.079** |
| followupqg   | 0.7305    | 0.7537    | **+0.023** |
| multiwoz_v24 | 0.0788    | 0.1149    | **+0.036** |
| qrecc        | 0.2084    | 0.2340    | **+0.026** |
| workflow     | 0.3276    | 0.4413    | **+0.114** |

This is the clearest result: more transformer capacity directly improves the
cross-encoder's corpus-retrieval ability, partially closing the gap with the BiEncoder.

### 2. Hard-negative pair discrimination regresses with extra depth.

Counter-intuitively, CE-2x is *worse* than CE-1x on AUC(HN) / P@1(HN) for most
datasets. The 24-layer model is harder to train on the small datasets (aep_causal,
5.5K pairs) and over-refines on the limited hard-negative signal for workflow.
This suggests the capacity benefit requires more data to fully manifest.

### 3. CE-2x still loses retrieval to BiEncoder — but the gap narrows.

| Dataset      | BE MRR | CE-1x MRR | CE-2x MRR | CE-2x vs BE |
|--------------|-------:|----------:|----------:|------------:|
| aep_causal   | 0.3525 | 0.1914    | 0.2707    | −0.082      |
| followupqg   | 0.5844 | 0.7305    | 0.7537    | **+0.169**  |
| multiwoz_v24 | 0.2493 | 0.0788    | 0.1149    | −0.134      |
| qrecc        | 0.2758 | 0.2084    | 0.2340    | −0.042      |
| workflow     | 0.5500 | 0.3276    | 0.4413    | −0.109      |

The CE-2x closes ~40–60% of the CE-1x retrieval gap vs. BiEncoder on most datasets,
and beats the BiEncoder outright on followupqg (+0.169 MRR).

### 4. followupqg: CE-2x is the clear winner across all three architectures.

followupqg (short question → follow-up question pairs) uniquely favors cross-attention:
- CE-2x MRR 0.754 > CE-1x 0.730 > BiEncoder 0.584
- CE-2x AUC(HN) 0.925 vs BiEncoder 0.837 (+0.088)
- Median rank 1.0 for both cross-encoders vs 2.0 for BiEncoder

### 5. aep_causal remains the hardest dataset for cross-encoders.

aep_causal (5.5K train pairs) shows both cross-encoders underperforming the
BiEncoder on every metric. The BCE objective does not calibrate scores across anchors,
and the small training set limits what even the deeper model can learn.

## Caveats

1. **Data caps.** The BiEncoder baseline was trained with legacy caps
   (`multiwoz_v24=15K`, `qrecc=15K`, `workflow=20K`). Both CE variants use the
   same capped data from `finetune_eval/results/`. The comparison is internally
   consistent across CE-1x and CE-2x, but note the BiEncoder in `cross_encoder/README.md`
   was trained with `finetune_eval/` (capped) data — same as here.
2. **Val splits.** Only aep_causal has a mined val split; the other four fall back
   to a 5% holdout of train. This is identical to the 1x CE setup.
3. **Pool subsampling.** Retrieval metrics for multiwoz_v24, qrecc, workflow use
   2000-anchor subsamples — same cap as the 1x CE, so the comparison is fair.
4. **Loss.** BCE is the simplest cross-encoder loss. Listwise softmax over
   (1 pos + 4 hardneg) groups would directly address corpus calibration and
   likely further improve retrieval MRR.

## Binary Classification Experiment

A separate experiment trains the CrossEncoder2x directly as a **0/1 binary classifier**
on `text_1` / `text_2` pairs — no hard negatives, no retrieval setup.  Each example
is labelled 1 (causal follow-up) or 0 (not), and the model learns to predict that label.

### Setup

```bash
VENV=/mnt/localssd/automation/internship-causal-embedding/.venv/bin/python
CE2X=/mnt/localssd/internship-causal-embedding-merge-followup/my_work/kl/cross_encoder_2x

# aep_causal_classification  (110 775 train / 6 004 val / 6 004 test)
CUDA_VISIBLE_DEVICES=0 $VENV $CE2X/train_ce2x_classification.py \
    --data-dir /mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification \
    --out-dir results/classification \
    --epochs 5 --batch-size 32 --grad-accum 2 --max-length 256

# aep_dataset  (67 500 train / 8 460 val / 8 460 test)
CUDA_VISIBLE_DEVICES=1 $VENV $CE2X/train_ce2x_classification.py \
    --data-dir /mnt/localssd/automation/internship-causal-embedding/data/aep_dataset \
    --out-dir results/classification_aep_dataset \
    --epochs 5 --batch-size 32 --grad-accum 2 --max-length 256
```

Input format: `[CLS] text_1 [SEP] text_2 [SEP]` → 24-layer BERT → CLS → Linear(1) → BCE loss.
Threshold at logit > 0 to predict class 1 at inference.

### Training history

**aep_causal_classification**

| Epoch | val AUC | val Acc |
|------:|--------:|--------:|
| 1     | 0.9284  | 87.3%   |
| 2     | **0.9393** | —   |
| 3     | —       | —       |
| 4     | 0.9046  | 88.3%   |
| 5     | 0.9179  | 88.7%   |

Early stopped at epoch 5 (patience 2). Best checkpoint = epoch 2.

**aep_dataset**

| Epoch | val AUC | val Acc |
|------:|--------:|--------:|
| 1     | —       | —       |
| 2     | —       | —       |
| 3     | **0.9805** | 93.8% |
| 4     | 0.9778  | 94.2%   |
| 5     | 0.9801  | 94.3%   |

Early stopped at epoch 5 (patience 2). Best checkpoint = epoch 3.

### Test-set results

| Dataset                    | Test AUC | Test Accuracy | N test |
|----------------------------|:--------:|:-------------:|-------:|
| `aep_causal_classification`| 0.8970   | **85.5%**     |  6,004 |
| `aep_dataset`              | 0.9598   | **91.0%**     |  8,460 |

Accuracy = correct predictions / total samples (threshold: logit > 0 → class 1).

`aep_dataset` is substantially easier — the model achieves 91% accuracy vs 85.5% on
`aep_causal_classification`, likely because `aep_causal_classification` contains harder,
more semantically similar negative pairs.

## Files

| File                            | Purpose                                                   |
|---------------------------------|-----------------------------------------------------------|
| `model_ce2x.py`                 | `CrossEncoder2x` + `_build_stacked_bert()`                |
| `train_ce2x.py`                 | Training loop for retrieval task (hard-negative pairs)    |
| `train_ce2x_classification.py`  | Training loop for binary classification (0/1 labels)      |
| `evaluate_ce2x.py`              | Evaluation for retrieval task                             |
| `results/`                      | Per-dataset checkpoints + metric JSON files               |
