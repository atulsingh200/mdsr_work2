#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder + reasoning aux-loss decoder
# trained on aep_ajo_procedural_workflow_v2_fixed_reas (rewritten reasoning, fully aligned).
# Loss = L_BCE + 0.7 * L_LM
# Best checkpoint selected on val AUC.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

MODEL_CE2X_DIR="$REPO_ROOT/../../new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x"
export PYTHONPATH="$MODEL_CE2X_DIR:${PYTHONPATH:-}"

DATA_DIR="$REPO_ROOT/data/aep_ajo_procedural_workflow_v2_fixed_reas"

GPU="${GPU:-1}"
echo "[run] using GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x_deberta_reasoning
mkdir -p "$RUNS"
RUN_NAME=v2_fixed_reas_lm_alpha07

echo "===== CE2x-DeBERTa + reasoning aux-loss (alpha=0.7) on aep_ajo_procedural_workflow_v2_fixed_reas ====="
python3 -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_DIR" \
    --split-prefix "" \
    --out-dir "$RUNS" --run-name "$RUN_NAME" \
    --backbone microsoft/deberta-v3-large \
    --native-backbone \
    --decoder-layers 4 \
    --cls-loss-weight 1.0 \
    --decoder-loss-weight 0.7 \
    --epochs 12 \
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
