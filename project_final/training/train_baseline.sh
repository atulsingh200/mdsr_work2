#!/usr/bin/env bash
# =============================================================================
# Train CrossEncoder2x BASELINE (no reasoning decoder)
# Model: v2_res_baseline
#
# Exact config used:
#   backbone=microsoft/deberta-v3-large, native-backbone
#   epochs=5, batch=16, grad_accum=4, lr=1e-5, weight_decay=0.01
#   warmup_frac=0.1, grad_clip=1.0, dropout=0.1, seed=42
#   split_prefix=directional_, step_key=step
#
# Reproduces: val_acc=0.8225  val_auc=0.9027  (best at epoch=5)
#
# Usage:
#   cd /mnt/localssd/causal-embedding-research/project_final
#   source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate
#   CUDA_VISIBLE_DEVICES=0 bash training/train_baseline.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

DATA_DIR="$PROJ/data"
OUT_DIR="$PROJ/runs/crossencoder2x_deberta"
mkdir -p "$OUT_DIR"

echo "===== Training CrossEncoder2x BASELINE ====="
echo "  data : $DATA_DIR/directional_{train,val,test}.jsonl"
echo "  out  : $OUT_DIR/v2_res_baseline"

python "$HERE/train_encoder.py" \
    --data-dir      "$DATA_DIR" \
    --split-prefix  "directional_" \
    --out-dir       "$OUT_DIR" \
    --run-name      "v2_res_baseline" \
    --backbone      microsoft/deberta-v3-large \
    --native-backbone \
    --epochs        5 \
    --batch-size    16 \
    --eval-batch-size 16 \
    --grad-accum    4 \
    --lr            1e-5 \
    --weight-decay  0.01 \
    --warmup-frac   0.1 \
    --grad-clip     1.0 \
    --dropout       0.1 \
    --seed          42 \
    --step-key      step \
    2>&1 | tee "$OUT_DIR/train_baseline.log"

echo "===== DONE. Results in $OUT_DIR/v2_res_baseline ====="
