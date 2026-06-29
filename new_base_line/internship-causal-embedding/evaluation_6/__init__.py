"""Inference-only evaluation harness for off-the-shelf encoders on data_6.

A lightweight companion to `src/evaluation/` that runs pure inference (no
fine-tuning) of pretrained HF encoders across every dataset under `data_6/`
and reports both:

  * Pair-level metrics: AUC, Precision@1 (vs N random negatives).
  * Retrieval metrics:  MRR, Recall@K, Hit@K, mean / median rank
                        (same-split candidate pool).
"""
