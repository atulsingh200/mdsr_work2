#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder BASELINE (no reasoning decoder)
# trained on aep_ajo_procedural_workflow_v2_res.
# This is the alpha=0 ablation to measure the delta from the aux-loss head.
# Best checkpoint selected on val accuracy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

MODEL_CE2X_DIR="$REPO_ROOT/../../new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x"
export PYTHONPATH="$MODEL_CE2X_DIR:${PYTHONPATH:-}"

DATA_DIR="$REPO_ROOT/data/aep_ajo_procedural_workflow_v2_res"

GPU="${GPU:-2}"
echo "[run] using GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x_deberta
mkdir -p "$RUNS"
RUN_NAME=v2_res_baseline

echo "===== CE2x-DeBERTa BASELINE (alpha=0) on aep_ajo_procedural_workflow_v2_res ====="
python3 -m src.classifier.crossencoder.train_ce2x \
    --data-dir "$DATA_DIR" \
    --split-prefix "" \
    --out-dir "$RUNS" --run-name "$RUN_NAME" \
    --backbone microsoft/deberta-v3-large \
    --native-backbone \
    --epochs 5 \
    --batch-size 16 \
    --eval-batch-size 16 \
    --grad-accum 4 \
    --lr 1e-5 \
    --weight-decay 0.01 \
    --warmup-frac 0.1 \
    --grad-clip 1.0 \
    --dropout 0.1 \
    --seed 42 \
    --step-key step \
    2>&1 | tee "$RUNS/ce2x_deberta_${RUN_NAME}_stdout.log"

echo "===== DONE. Results in $RUNS/$RUN_NAME ====="
