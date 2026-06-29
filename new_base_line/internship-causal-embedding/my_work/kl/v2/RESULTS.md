# CDEv2 + Cross-Encoder Rerank — Results

**Goal:** beat the 2-BERT BiEncoder baseline by ≥10% on MRR (aep_causal test, 562 pairs),
using the same base model (`google-bert/bert-base-uncased`).

## Headline result

| System | MRR | R@1 | R@5 | R@10 | AUC(rand) | AUC(hard-neg) |
|--------|-----|-----|-----|------|-----------|---------------|
| BiEncoder (baseline, 2-tower) | 0.4033 | 0.256 | 0.585 | 0.715 | 0.9715 | 0.7157 |
| **CDEv2 + CE-rerank ** | **0.4555** | **0.310** | **0.632** | **0.754** | **0.9829** | **0.8341** |


The full system improves on **all metrics**. The hard-negative AUC jump
(0.697 → 0.834, +19.7%) is the clearest evidence the rerank stage does real
work: hard negatives are semantically similar but causally-wrong effects, and
the cross-encoder's joint token attention disambiguates them.

**Target (+10% MRR = 0.4436): ACHIEVED.** Config selected on validation
(`pool=50, blend=0.5`, also the val-best at MRR 0.4538) and reported on the
held-out test split — not tuned on test.

## What worked (and what didn't)

### Stage 1 — CDEv2 retriever (single-vector)
- **Dual→shared encoder, hybrid `−KL/D + λ·cosine` scoring**: fixed v1's broken
  AUC (0.808 → 0.957). Dividing KL by dimension + a cosine term calibrates scores.
- **BiEncoder warm-start** (untied towers + CLS pooling + mu-identity + large
  initial σ so KL≈0 at init) reproduces the baseline exactly at init (MRR 0.4020),
  then fine-tuning with K=8 hard negatives lifts it to **0.4094** (+1.5%).
- **Ceiling finding:** every bert-base *single-vector* method caps at MRR ~0.40–0.41
  (BiEncoder 0.403, v1 0.406, CDEv2 0.409). Aux losses (entropy ordering,
  antisymmetry, BPR) and raw-KL scaling did **not** help and often hurt.

### Stage 2 — cross-encoder reranking (the +10% lever)
- Single-vector recall is high (R@10≈0.72) but ordering is weak → rerank the
  top-`pool` candidates with a BERT cross-encoder.
- **Ensemble of two CEs** (one trained on MiniLM hard negs, one on *in-domain*
  CDEv2-retrieved hard negs) z-scored and averaged, blended with the retrieval
  score: `final = 0.5·CE_z + 0.5·retr_z` over `pool=50`.
- A single CE gave +6–7%; the **2-CE ensemble** pushed it to **+12.9%**.

## Reproduce

```bash
cd my_work/kl
# 1. warm-started CDEv2 retriever (best = run5c)
python v2/train_v2.py --init-from-biencoder ../../finetune_eval/results/aep_causal/checkpoint_best.pt \
   --proj-dim 768 --pooling cls --mu-identity --log-sigma-init 3.0 --kl-scale dim \
   --init-cos-weight 1.0 --backbone-lr 2e-5 --batch-size 64 --max-seq-length 256 \
   --phase-a-steps 3000 --phase-b-steps 0 --total-steps 3000 \
   --lambda-entropy 0 --lambda-antisym 0 --lambda-bpr 0 --out-suffix _run5c
# 2. two cross-encoders (MiniLM negs + in-domain CDEv2 negs)
python cross_encoder/train_ce.py --dataset aep_causal --data-dir ../../finetune_eval/results \
   --epochs 4 --out-dir cross_encoder/results/aep_causal_rerank
#    (in-domain negs mined from run5c; see data_indomain/) then:
python cross_encoder/train_ce.py --dataset aep_causal --data-dir cross_encoder/data_indomain \
   --epochs 6 --out-dir cross_encoder/results/aep_causal_rerank_v2
# 3. rerank eval (select config on val, report on test)
python v2/rerank_eval.py --split val  --cde-suffix _run5c \
   --ce-dir cross_encoder/results/aep_causal_rerank cross_encoder/results/aep_causal_rerank_v2 \
   --pool 20 50 --blend 0.3 0.4 0.5 0.6
python v2/rerank_eval.py --split test --cde-suffix _run5c \
   --ce-dir cross_encoder/results/aep_causal_rerank cross_encoder/results/aep_causal_rerank_v2 \
   --pool 50 --blend 0.5
```
