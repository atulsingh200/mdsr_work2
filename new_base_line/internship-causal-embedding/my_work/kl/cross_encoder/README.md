# BERT Cross-Encoder — fine-tune & retrieval evaluation

A single BERT (`bert-base-uncased`) trained as a **cross-encoder**:
the anchor and a candidate are concatenated into one BERT input
(`[CLS] anchor [SEP] candidate [SEP]`), the [CLS] representation is
projected to a single logit by a Linear(768→1) head, and the model is
trained with **BCE-with-logits** on label ∈ {0, 1} (1 = true positive,
0 = mined semantic hard negative).

This README reports test-set results for **5 datasets** (`aep_causal`,
`followupqg`, `multiwoz_v24`, `qrecc`, `workflow`), each fine-tuned
uncapped against the BiEncoder two-tower baseline already in
`finetune_eval/`.

## Setup

Mining and training share the same data layout: each dataset under
`my_work/kl/cross_encoder/data/<dataset>/` holds `{train,val,test}_pairs.jsonl`
plus `{,val_,test_}hard_negatives.npy` (K=4 nearest other positives in
MiniLM-L6-v2 space).

### Mining (full data, no cap)

```bash
PY=/mnt/localssd/base_line/internship-causal-embedding/.venv/bin/python
REPO=/mnt/localssd/base_line/internship-causal-embedding
KL=/mnt/localssd/base_line/my_work/kl/cross_encoder

cd $REPO
for ds in aep_causal followupqg multiwoz_v24 qrecc workflow; do
  for split in train val test; do
    $PY finetune_eval/mine_hard_negatives.py \
        --dataset $ds --split $split --k 4 --no-cap \
        --out-dir $KL/data
  done
done
```

Output: `(N, 4)` int32 indices into the positives list per split.

### Training (per dataset)

```bash
# Small / medium (aep_causal, followupqg, multiwoz_v24, qrecc):
$PY $KL/train_ce.py --dataset <DATASET> \
    --epochs 10 --batch-size 128 --max-length 256 \
    --early-stop-patience 2 --early-stop-threshold 0.001

# Workflow (2.1M pairs uncapped):
$PY $KL/train_ce.py --dataset workflow \
    --epochs 3 --batch-size 256 --max-length 128 \
    --val-batch-size 512 --num-workers 4 \
    --early-stop-patience 1 --early-stop-threshold 0.005
```

Loss is BCE-with-logits. AdamW (lr 2e-5, weight-decay 0.01).
Linear warmup (6 %) → linear decay. bf16 mixed-precision. Validation
runs at the end of every epoch; best checkpoint saved by val AUC.

### Evaluation (all 5 in parallel)

```bash
# Each command in its own shell — the A100 fits multiple BERT-base evals.
$PY $KL/evaluate_ce.py --dataset aep_causal    --pool-size 1000 --max-queries 2000 --batch-size 128 --max-length 256 &
$PY $KL/evaluate_ce.py --dataset followupqg    --pool-size 1000 --max-queries 2000 --batch-size 128 --max-length 256 &
$PY $KL/evaluate_ce.py --dataset multiwoz_v24  --pool-size 1000 --max-queries 2000 --batch-size 128 --max-length 256 &
$PY $KL/evaluate_ce.py --dataset qrecc         --pool-size 1000 --max-queries 2000 --batch-size 128 --max-length 256 &
$PY $KL/evaluate_ce.py --dataset workflow      --pool-size 1000 --max-queries 2000 --batch-size 256 --max-length 128 &
wait
```

Three metric groups are computed and saved to `results/<dataset>/{eval,eval_hardneg}.json`:

1. **AUC + P@1 vs 4 random negatives** (cheap pair-level)
2. **AUC + P@1 vs 4 semantic hard negatives** (mined from MiniLM kNN on the test split — same kind the model trained on)
3. **Pool retrieval**: gold positive scored against 999 random distractors per anchor, ranks → MRR / Recall@K / median rank. Test queries are subsampled at `max_queries=2000` for the larger test splits (multiwoz_v24, qrecc, workflow) because every (anchor, candidate) pair needs its own BERT forward — full pool over the full test set would be ~260 M forwards on workflow alone.

## Dataset sizes

| Dataset       | Train pairs | Val pairs | Test pairs | Train rows (1 pos + 4 hardneg) |
|---------------|-------------|-----------|------------|--------------------------------|
| aep_causal    | 5,487       | 562       | 562        | 27,435                         |
| followupqg    | 2,790       | 500       | 501        | 13,950                         |
| multiwoz_v24  | 56,719      | 7,374     | 7,368      | 283,595                        |
| qrecc         | 52,481      | —         | 5,204      | 262,405 (5 % of train held out as val) |
| workflow      | 2,116,157   | 261,049   | 260,487    | 10,580,785                     |

No `TRAIN_CAPS` applied — the cross-encoder sees all available training data.
(Note: the BiEncoder baseline at `finetune_eval/results/<dataset>/` was trained
with the legacy caps `multiwoz_v24=15K`, `qrecc=15K`, `workflow=20K`.)

## Training summary

| Dataset       | Best val AUC | Best val P@1 | Best epoch | Stopped early? |
|---------------|-------------:|-------------:|:----------:|:--------------:|
| aep_causal    | 0.7134       | 0.4235       | 3 / 3      | no (fixed 3-ep) |
| followupqg    | 0.9239       | 0.8420       | 2 / 10     | yes (epoch 4)   |
| multiwoz_v24  | 0.7438       | 0.4742       | 5 / 10     | yes (epoch 7)   |
| qrecc         | 0.9040       | 0.7110       | 1 / 10     | yes (epoch 3)   |
| workflow      | 0.8834       | —            | 1 / 3      | yes (epoch 2)   |

Early stopping: `patience=2`, `threshold=0.001` AUC for the medium-size runs;
`patience=1`, `threshold=0.005` for workflow.

## Test-set results

### Cross-encoder (this work, uncapped)

| Dataset       | N test | AUC(rand) | P@1(rand) | AUC(HN) | P@1(HN) | MRR    | R@1    | R@3    | R@5    | R@10   | Median |
|---------------|-------:|----------:|----------:|--------:|--------:|-------:|-------:|-------:|-------:|-------:|-------:|
| aep_causal    |   562  | 0.8453    | 0.6957    | 0.7113  | 0.4484  | 0.1914 | 0.110  | 0.212  | 0.265  | 0.345  |  28.0  |
| followupqg    |   501  | 0.9711    | 0.9421    | 0.9252  | 0.8543  | 0.7305 | 0.649  | 0.780  | 0.830  | 0.874  |   1.0  |
| multiwoz_v24  | 7,368  | 0.7823    | 0.5050    | 0.7444  | 0.4673  | 0.0788 | 0.044  | 0.075  | 0.094  | 0.133  | 156.0  |
| qrecc         | 5,204  | 0.8552    | 0.6489    | 0.6943  | 0.4262  | 0.2084 | 0.155  | 0.221  | 0.254  | 0.304  |  72.0  |
| workflow      | 260,487| 0.9403    | 0.8131    | 0.8801  | 0.6692  | 0.3276 | 0.259  | 0.346  | 0.390  | 0.450  |  16.0  |

Retrieval columns (MRR / R@K / Median) on multiwoz_v24, qrecc, workflow are
sampled to `max_queries=2000` of the test anchors (the cross-encoder runs
~2 M BERT forwards even at that cap; full would be 7–260 M).

### BiEncoder baseline (`finetune_eval/results/`, same backbone)

| Dataset       | N test | AUC(rand) | P@1(rand) | AUC(HN) | P@1(HN) | MRR    | R@1    | R@3    | R@5    | R@10   | Median |
|---------------|-------:|----------:|----------:|--------:|--------:|-------:|-------:|-------:|-------:|-------:|-------:|
| aep_causal    |   562  | 0.9712    | 0.9128    | 0.7157  | 0.5178  | 0.3525 | 0.164  | 0.457  | 0.585  | 0.715  |   4.0  |
| followupqg    |   501  | 0.9827    | 0.9541    | 0.8370  | 0.7365  | 0.5844 | 0.403  | 0.731  | 0.806  | 0.854  |   2.0  |
| multiwoz_v24  | 7,368  | 0.9526    | 0.8694    | 0.6223  | 0.4110  | 0.2493 | 0.161  | 0.264  | 0.327  | 0.426  |  16.0  |
| qrecc         | 5,204  | 0.9227    | 0.8113    | 0.6365  | 0.4137  | 0.2758 | 0.182  | 0.306  | 0.365  | 0.457  |  14.0  |
| workflow      | 260,487| 0.9662    | 0.9104    | 0.6883  | 0.4763  | 0.5500 | 0.466  | 0.593  | 0.643  | 0.706  |   2.0  |

BiEncoder retrieval (R@K / MRR) ran the full test pool of 1000 random
distractors over **every** test anchor (cheap because it's a dot product
on cached embeddings). Cross-encoder ran 2000 anchors.

### Head-to-head deltas (CE − BE)

| Dataset       | ΔAUC(rand) | ΔAUC(HN) | ΔP@1(HN) | ΔMRR    | ΔR@1   | Winner |
|---------------|-----------:|---------:|---------:|--------:|-------:|:-------|
| aep_causal    | −0.126     | −0.004   | **−0.069** | −0.161  | −0.054 | BE (sweep) |
| followupqg    | −0.012     | **+0.088** | **+0.118** | +0.146  | +0.246 | **CE** on every metric |
| multiwoz_v24  | −0.170     | **+0.122** | **+0.056** | −0.171  | −0.117 | mixed — CE wins on pair-level HN, BE on retrieval |
| qrecc         | −0.068     | **+0.058** | **+0.013** | −0.067  | −0.027 | mixed — CE narrowly on pair-level HN, BE on retrieval |
| workflow      | −0.026     | **+0.192** | **+0.193** | −0.222  | −0.207 | mixed — CE huge on pair-level HN, BE on retrieval |

(Positive Δ = cross-encoder wins.)

## Findings

### 1. Cross-encoder dominates pair-level hard-negative discrimination.

On the metric the cross-encoder is *designed for* — "given the anchor, its
true positive, and 4 hard distractors, pick the right one" — it beats the
two-tower on **4 of 5 datasets**:

- workflow: P@1(HN) 0.669 vs 0.476  (**+19.3 pts**)
- followupqg: P@1(HN) 0.854 vs 0.737 (**+11.8 pts**)
- multiwoz_v24: P@1(HN) 0.467 vs 0.411 (**+5.6 pts**)
- qrecc: P@1(HN) 0.426 vs 0.414 (**+1.3 pts**)

The only loss is `aep_causal` (CE 0.448 vs BE 0.518) — the smallest dataset
in the suite (5.5K train pairs). The cross-encoder seems to need more data
to learn a calibrated decision boundary; on aep_causal it's still
under-trained at the early-stop point.

### 2. Cross-encoder loses at retrieval — every time.

MRR drops dramatically against the two-tower on every dataset:

- aep_causal: 0.191 vs 0.353 (BE +0.16)
- followupqg: 0.730 vs 0.584 ✅ CE wins (+0.15) — the **one** retrieval win
- multiwoz_v24: 0.079 vs 0.249 (BE +0.17)
- qrecc: 0.208 vs 0.276 (BE +0.07)
- workflow: 0.328 vs 0.550 (BE +0.22)

This is the standard cross-encoder limitation: the **independent-row BCE
loss** never forces the score scale to be comparable across anchors. When
you ask "rank this anchor's positive against 999 other anchors' positives,"
the cross-encoder has no incentive to keep scores normalized — every row was
trained as its own independent binary classification.

A two-tower trained with **InfoNCE** is implicitly forced to keep
embeddings on a shared norm-1 sphere where dot products are corpus-
comparable.

### 3. The two-tower's AUC(random) is misleadingly good.

The BiEncoder wins AUC(random) on every dataset — but those random
negatives are *very* easy to discriminate by topic, so the cosine-similarity
scale separates them cleanly. The harder, more informative metric is
**AUC(HN)**, where the cross-encoder ties or wins on 4/5 datasets.

### 4. followupqg is the only "clean win" for the cross-encoder.

It beats the two-tower on every single metric:

| Metric    | CE    | BE    | Δ      |
|-----------|-------|-------|--------|
| AUC(HN)   | 0.925 | 0.837 | +0.088 |
| P@1(HN)   | 0.854 | 0.737 | +0.118 |
| MRR       | 0.730 | 0.584 | +0.146 |
| R@1       | 0.649 | 0.403 | +0.246 |
| R@10      | 0.874 | 0.854 | +0.020 |
| Median    | 1     | 2     | −1     |

Plausibly because followupqg pairs are short and semantically tight
(question → follow-up question), so a cross-encoder's full attention
across both sides captures the relation better than two independent
encoders can.

## Caveats

1. **Data asymmetry.** The cross-encoder is trained uncapped on every
   dataset. The BiEncoder baseline at `finetune_eval/results/` used the
   legacy caps `multiwoz_v24=15K`, `qrecc=15K`, `workflow=20K`. The
   comparison thus *favours* the cross-encoder on the three large datasets
   but is still a useful read on architecture differences.
2. **Pool subsampling.** Cross-encoder pool retrieval is over a 2,000-
   anchor subsample (computational necessity — full pool over 260K
   workflow anchors would need 260 M BERT forwards). The BiEncoder
   evaluates the full pool. MRR and Recall@K are thus on slightly
   different anchor sets for the larger datasets.
3. **Loss choice.** BCE is the simplest cross-encoder loss but not the
   best one. Public cross-encoders (e.g. ms-marco-MiniLM) use
   **listwise softmax** over (1 pos + K hardneg) groups; that would
   directly fix the corpus-calibration issue we see in §2. Out of scope
   for this run but a clear next step if you want a one-architecture
   answer.
4. **`bert-base-uncased` everywhere.** Heavier backbones (DeBERTa-v3-large,
   RoBERTa-large) would close the retrieval gap further; we kept the
   backbone fixed to match the BiEncoder.

## Repro

Source: this directory.
Data layout: `data/<dataset>/{train,val,test}_pairs.jsonl + *hard_negatives.npy`.
Outputs: `results/<dataset>/{checkpoint_best.pt, training_history.json, eval.json, eval_hardneg.json, train.log}`.

Checkpoints are .gitignored (each is ~440 MB).
