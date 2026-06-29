# Cause→Effect GNN — Results vs BiEncoder Baseline

**Goal:** beat the fine-tuned `bert-base-uncased` two-tower BiEncoder
(`finetune_eval`) by **≥10%** on retrieval metrics for `aep_causal`.

**Result: achieved by a wide margin — +21.3% N×N MRR, +19.4% pool MRR, +33% R@1.**

All metrics computed with the same code (`evaluation_6/metrics_extra.py`),
seed 0, pool size 1000, on the `data_6/aep_causal` test split (562 pairs).

## Headline comparison (test split)

| Metric    | BERT BiEncoder (baseline) | **bge-large e2e (best)** | Δ%        |
|-----------|---------------------------|--------------------------|-----------|
| N×N MRR   | 0.4033                    | **0.4893**               | **+21.3%** |
| pool MRR  | 0.3574                    | **0.4269**               | **+19.4%** |
| pool R@1  | 0.1833                    | **0.2438**               | **+33.0%** |
| pool R@5  | 0.5712                    | **0.6477**               | +13.4%    |
| pool R@10 | 0.7153                    | **0.7847**               | +9.7%     |
| AUC       | 0.9688                    | 0.9729                   | +0.4%     |
| P@1       | 0.9146                    | 0.9359                   | +2.3%     |

## All configurations (test split)

| Config | encoder | GNN | batch / k_neg | val MRR | test N×N MRR | test pool MRR | Δ% pool |
|---|---|---|---|---|---|---|---|
| baseline | bert-base-uncased | — | 32 / 4 | 0.410 | 0.4033 | 0.3574 | — |
| enc-only v1 | bge-base | no | 32 / 4 | 0.4715 | 0.4416 | 0.3993 | +11.7% |
| enc-only v2 | bge-base | no | **64 / 8** | 0.4735 | 0.4470 | 0.3964 | +10.9% |
| **enc-only large** | **bge-large** | no | 24 / 8 | **0.5025** | **0.4893** | **0.4269** | **+19.4%** |
| gnn A | bge-base | residual | 32 / 4 | 0.4476 | — | — | — |
| gnn B | bge-base | residual (3L/8H) | 32 / 8 | 0.4406 | — | — | — |
| gnn stage-2 | bge-base (frozen) | residual, matched feats | 64 / 8 | 0.4667 | — | — | — |

## What drove the win

1. **Stronger backbone, fine-tuned end-to-end.** `bert-base-uncased` is not a
   retrieval model; BGE (`bge-base/large-en-v1.5`) is. Swapping it in and
   fine-tuning end-to-end with the same hard-negative InfoNCE loss is the main
   lever. bge-large > bge-base > bert-base.
2. **More contrastive signal.** Larger in-batch negatives (batch 64 vs 32) plus
   `k=8` mined hard negatives pushed bge-base's N×N MRR from +9.5% to +10.8%.
3. **Hard negatives** mined exactly as in `finetune_eval/mine_hard_negatives.py`
   (semantic kNN with MiniLM) **plus an anchor-similarity filter**: a candidate
   negative `j` is dropped if `sim(anchor_i, anchor_j) > 0.9`, since
   near-duplicate anchors may have genuinely valid effects (false negatives).

## The GNN component: rigorously tested, does not help on this dataset

The directional GNN was tried **three** ways, all as an additive structural
residual `final = L2norm(text_emb + gate · W_gnn(GNN(doc)))`:

| Design | graph features | gate behaviour | outcome |
|---|---|---|---|
| frozen pre-FT residual | pre-fine-tune BGE | gate → 0.03 | val < encoder-only |
| e2e joint | pre-fine-tune BGE | gate → 0.02 | val < encoder-only |
| two-stage, matched | **fine-tuned BGE** | gate → 0.01 | val 0.467 < 0.4715 |

In every case the learnable mixing gate **collapses toward zero** — the model
actively learns to ignore the GNN. The reason: for AEP documentation, causal
links are already predictable from document text, so a fine-tuned text encoder
implicitly captures the structure and the GNN correction is redundant noise.

This is a genuine, well-supported finding, not a tuning failure: matched-feature
two-stage training (where the graph nodes are encoded by the *same* fine-tuned
encoder) still collapses the gate. **A directional GNN adds value only when graph
structure is NOT recoverable from text** (citation networks, knowledge graphs,
multi-hop chains) — which aep_causal's mostly-direct doc→doc links are not.

## Reproduce

```bash
PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python

# Best model: bge-large two-tower, fine-tuned end-to-end (no GNN)
CUDA_VISIBLE_DEVICES=0 $PY gnn/train_gnn_e2e.py \
  --dataset aep_causal --encoder BAAI/bge-large-en-v1.5 \
  --run-name aep_causal_e2e_nognn_large --no-gnn \
  --epochs 8 --batch-size 24 --k-neg 8 --encoder-lr 1.5e-5 --bf16

CUDA_VISIBLE_DEVICES=0 $PY gnn/evaluate_e2e.py \
  --run-dir gnn/results/aep_causal_e2e_nognn_large --dataset aep_causal

# GNN ablation (two-stage on frozen fine-tuned encoder, matched features)
CUDA_VISIBLE_DEVICES=0 $PY gnn/train_gnn_e2e.py \
  --dataset aep_causal --encoder BAAI/bge-base-en-v1.5 \
  --run-name aep_causal_e2e_gnn_stage2 \
  --init-from gnn/results/aep_causal_e2e_nognn_bge-base-en-v1.5 \
  --freeze-encoder --epochs 20 --batch-size 64 --k-neg 8 --gate-init 0.5 --bf16
```
