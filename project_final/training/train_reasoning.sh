#!/usr/bin/env bash
# =============================================================================
# Train CrossEncoder2x + REASONING DECODER (main reasoning model)
# Model: v2_fixed_alpha02_ep10
#
# Exact config used:
#   backbone=microsoft/deberta-v3-large, native-backbone
#   decoder_layers=4, cls_loss_weight=1.0, decoder_loss_weight=0.2
#   epochs=10, batch=16, grad_accum=4, lr=1e-5, weight_decay=0.01
#   warmup_frac=0.1, grad_clip=1.0, dropout=0.1, seed=42
#   split_prefix=directional_, step_key=step
#
# NOTE: training data must have a 'reasoning' field (used by the decoder).
#       directional_train.jsonl already contains this field.
#
# Reproduces: val_acc=0.8331  val_auc=0.9142  (best at epoch=3)
#
# Usage:
#   cd /mnt/localssd/causal-embedding-research/project_final
#   source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate
#   CUDA_VISIBLE_DEVICES=0 bash training/train_reasoning.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

DATA_DIR="$PROJ/data"
OUT_DIR="$PROJ/runs/crossencoder2x_deberta_reasoning"
mkdir -p "$OUT_DIR"

echo "===== Training CrossEncoder2x + REASONING DECODER ====="
echo "  data : $DATA_DIR/directional_{train,val,test}.jsonl"
echo "  out  : $OUT_DIR/v2_fixed_alpha02_ep10"

python "$HERE/train_encoder_reasoning.py" \
    --data-dir        "$DATA_DIR" \
    --split-prefix    "directional_" \
    --out-dir         "$OUT_DIR" \
    --run-name        "v2_fixed_alpha02_ep10" \
    --backbone        microsoft/deberta-v3-large \
    --native-backbone \
    --decoder-layers  4 \
    --cls-loss-weight 1.0 \
    --decoder-loss-weight 0.2 \
    --epochs          10 \
    --batch-size      16 \
    --eval-batch-size 16 \
    --grad-accum      4 \
    --lr              1e-5 \
    --weight-decay    0.01 \
    --warmup-frac     0.1 \
    --grad-clip       1.0 \
    --dropout         0.1 \
    --seed            42 \
    --step-key        step \
    2>&1 | tee "$OUT_DIR/train_reasoning.log"

echo "===== DONE. Results in $OUT_DIR/v2_fixed_alpha02_ep10 ====="
