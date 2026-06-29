# finetune_eval

Fine-tunes a BERT bi-encoder on each `data_6/` training split with **semantic
hard negatives**, then evaluates on the dataset's test split using the same
metrics as [`evaluation_6/`](../evaluation_6/).

Trains 5 models in total — one per dataset that has a `train` split:
`aep_causal`, `followupqg`, `multiwoz_v24`, `qrecc`, `workflow`.
`aep_followup` is excluded since it has no train split.

## Hard-negative mining (the key idea)

For each `(anchor x, positive y)` pair in the training data we need 4
negative examples to drive a contrastive loss. Random negatives are too
easy — most random `y'` is nothing like `y` and the model trivially
separates them. We use **semantically similar** negatives instead:

```
For every anchor x with true positive y:
    Encode y and every OTHER y' with all-MiniLM-L6-v2 (frozen).
    Take the top-4 nearest y' in cosine-similarity space.
    These 4 y* are the hard negatives.

Training labels:
    (x, y)     → 1      true positive
    (x, y*_k)  → 0      hard negative, k = 1..4   (semantically close to y)
```

Why this works: `y*` looks topically similar to `y` (same vocabulary,
same general topic) but isn't the actual follow-up. The model has to
learn a *finer* discrimination than "is this even in the right domain"
— exactly the signal we want for follow-up retrieval.

The miner is a separate frozen model (all-MiniLM-L6-v2) so its judgement
isn't influenced by the BERT we're training. Mining happens once per
dataset and is cached in `results/<ds>/hard_negatives.npy`.

## Two-tower architecture (re-used from `src/biencoder/`)

The `BiEncoder` in [`src/biencoder/model.py`](../src/biencoder/model.py)
is used as-is. Two untied encoders (one for anchors, one for positives),
both initialized from the same `bert-base-uncased` checkpoint, with CLS
pooling and L2-normalized output embeddings.

```
anchor x  →  AnchorEncoder  (BERT) →  pool → ℓ2-norm  →  a ∈ ℝ^768
positive y →  PositiveEncoder (BERT) →  pool → ℓ2-norm  →  p ∈ ℝ^768
hard neg y* →  PositiveEncoder (BERT) →  pool → ℓ2-norm  →  n ∈ ℝ^768
```

## Loss: InfoNCE + explicit hard negatives

For a batch of `B` pairs, each anchor has `B` candidate positives
(in-batch, `B-1` of them act as easy negatives) **plus** `K=4` explicit
hard negatives (specific to that anchor):

```
        positives                hard negatives
        ──────────                ────────────────
sim = [ a_i·p_0  a_i·p_1 … a_i·p_{B-1} | a_i·n_i,0 … a_i·n_i,K-1 ]   (B+K scores)
                          ↑
            anchor i's softmax target = index i  (its own positive)

ℓ = − log( exp(sim_i,i / τ) / Σ_j exp(sim_i,j / τ) )    (over B+K terms)
```

Implemented as `HardNegativeInfoNCELoss` in [losses.py](losses.py).
Temperature `τ` is a learnable parameter starting at 0.05.

## Pipeline

```
finetune_eval/
├── mine_hard_negatives.py    1) offline: kNN mine via MiniLM
├── train_finetune.py         2) fine-tune BERT bi-encoder
├── evaluate_finetune.py      3) test-set eval (AUC, P@1, MRR, R@K)
└── results/<ds>/
    ├── hard_negatives.npy    (N, 4) int indices
    ├── train_pairs.jsonl     the sampled training pairs (deterministic seed)
    ├── checkpoint_best.pt    best epoch by val MRR
    ├── config.json           run config
    ├── training_history.json per-epoch metrics
    └── eval.json             test-set metrics
```

## How to run

```bash
# 1. Mine all 5 datasets (~15s on A100)
.venv/bin/python finetune_eval/mine_hard_negatives.py --dataset all --k 4

# 2. Train one or all datasets
.venv/bin/python finetune_eval/train_finetune.py --dataset aep_causal --epochs 3
bash finetune_eval/train_all.sh   # orchestrates all 5

# 3. Evaluate trained checkpoints on test sets
.venv/bin/python finetune_eval/evaluate_finetune.py --dataset all
```

## Training configuration

- **Backbone**: `google-bert/bert-base-uncased` (CLS pooling, untied two-tower)
- **Max seq length**: 256
- **Batch size**: 32
- **Optimizer**: AdamW (lr 2e-5, encoder; 1e-3, temperature; weight_decay 0.01)
- **Schedule**: OneCycleLR, 10% warmup, cosine anneal
- **Loss**: HardNegativeInfoNCE with `K=4` mined hard negatives, learnable temperature `τ` (init 0.05)
- **Epochs**: 3 for small datasets (aep_causal, followupqg), 2 for larger
- **Early stopping**: 2-epoch patience on val MRR
- **Train caps** (for runtime budget):
  - aep_causal: full (5,487)
  - followupqg: full (2,790)
  - multiwoz_v24: 15,000 of 56,719
  - qrecc: 15,000 of 52,481
  - workflow: 20,000 of 2,116,157

## Results (test set, full split)

Same metric definitions as [`evaluation_6/README.md`](../evaluation_6/README.md):
candidate pool of 1000 (gold + 999 random distractors) per anchor; AUC / P@1 use 4 random negatives.

| dataset       |     N | AUC    | P@1    | MRR    | R@1    | R@5    | R@10   | Mean rank | Median rank |
|---------------|------:|-------:|-------:|-------:|-------:|-------:|-------:|----------:|------------:|
| aep_causal    |   562 | 0.9712 | 0.9128 | 0.3525 | 0.1637 | 0.5854 | 0.7153 |     16.82 |         4.0 |
| followupqg    |   501 | 0.9827 | 0.9541 | 0.5844 | 0.4032 | 0.8064 | 0.8543 |      8.53 |         2.0 |
| multiwoz_v24  | 7,368 | 0.9526 | 0.8694 | 0.2493 | 0.1606 | 0.3275 | 0.4256 |     43.21 |        16.0 |
| qrecc         | 5,204 | 0.9227 | 0.8113 | 0.2758 | 0.1822 | 0.3653 | 0.4570 |     68.97 |        14.0 |
| workflow      | 260k  | 0.9662 | 0.9104 | 0.5500 | 0.4656 | 0.6428 | 0.7055 |     32.30 |         2.0 |

## Fine-tuning lift over the baselines

Comparing to the inference-only numbers in [`evaluation_6/results.json`](../evaluation_6/results.json):

### AUC: fine-tuned BERT vs. raw BERT vs. MiniLM

| dataset       | BERT (raw, no FT) | MiniLM (no FT) | **BERT (FT here)** | Δ over raw BERT | Δ over MiniLM |
|---------------|------------------:|---------------:|-------------------:|----------------:|--------------:|
| aep_causal    | 0.7294 | 0.9371 | **0.9712** | +0.242 | +0.034 |
| followupqg    | 0.6350 | 0.9795 | **0.9827** | +0.348 | +0.003 |
| multiwoz_v24  | 0.5472 | 0.6706 | **0.9526** | +0.405 | +0.282 |
| qrecc         | 0.5762 | 0.8713 | **0.9227** | +0.347 | +0.051 |
| workflow      | 0.7817 | 0.8994 | **0.9662** | +0.185 | +0.067 |

Fine-tuned BERT beats both baselines on every dataset. The most dramatic
gains are on `multiwoz_v24` (+0.28 vs MiniLM) where pretrained semantic
encoders barely do better than chance — a domain that requires
in-distribution training to score correctly. The smallest gain is on
`followupqg` where MiniLM was already near-perfect (similar surface text
between anchor and positive).

### Training validation curves (best val MRR)

| dataset      | val MRR ep 1 | val MRR ep 2 | val MRR ep 3 |
|--------------|-------------:|-------------:|-------------:|
| aep_causal   | 0.34         | 0.40         | **0.42**     |
| followupqg   | 0.69         | **0.70**     | 0.70         |
| multiwoz_v24 | 0.17         | **0.19**     | —            |
| qrecc        | 0.28         | **0.29**     | —            |
| workflow     | 0.48         | **0.50**     | —            |

All runs improved monotonically across epochs (no overfitting observed
on val MRR within the epoch budget).

## Files

- [datasets.py](datasets.py) — per-dataset split paths + training caps.
- [data.py](data.py) — `TripleDataset` (anchor, positive, K hard negatives) and tokenizing collate.
- [losses.py](losses.py) — `HardNegativeInfoNCELoss` (B in-batch + K hard negs).
- [mine_hard_negatives.py](mine_hard_negatives.py) — kNN miner CLI.
- [train_finetune.py](train_finetune.py) — training CLI.
- [evaluate_finetune.py](evaluate_finetune.py) — test eval CLI.
- [train_all.sh](train_all.sh) / [train_rest.sh](train_rest.sh) — convenience runners.
- [results/](results/) — per-dataset checkpoints, configs, and metrics JSON.
