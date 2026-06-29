"""LorentzEnc workflow experiment.

Causal-aware bi-encoder with BGE-M3 backbone, space/time decomposition,
and asymmetric scoring. Trained on the full workflow dataset (2.1M pairs)
with semantic hard negatives + reverse-pair penalty.

Pipeline:
  1. Mine: finetune_eval/mine_hard_negatives.py --dataset workflow --no-cap
  2. Train: lorentz_enc_workflow/train.py
  3. Eval:  lorentz_enc_workflow/evaluate.py
"""
