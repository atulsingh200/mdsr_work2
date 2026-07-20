#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder trained on aep_ajo_procedural_fixed_reas
# with an auxiliary reasoning-decoder head (weight 0.3) alongside the
# classification BCE loss (weight 0.7). Reasoning is verdict-stripped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

MODEL_CE2X_DIR="$REPO_ROOT/../../new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x"
export PYTHONPATH="$MODEL_CE2X_DIR:${PYTHONPATH:-}"

DATA_DIR="$REPO_ROOT/data/aep_ajo_procedural_fixed_reas"

GPU=6
echo "[run] using GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x_deberta_reasoning
mkdir -p "$RUNS"

CE2X_ARGS="
  --backbone microsoft/deberta-v3-large
  --native-backbone
  --decoder-layers 4
  --cls-loss-weight 0.7
  --decoder-loss-weight 0.3
  --epochs 10
  --batch-size 16
  --eval-batch-size 16
  --grad-accum 4
  --lr 1e-5
  --weight-decay 0.01
  --warmup-frac 0.1
  --grad-clip 1.0
  --dropout 0.1
  --seed 42
"

echo "===== training CE2x-deberta-v3-large + reasoning aux-loss (cls=0.7, dec=0.3) on aep_ajo_procedural_fixed_reas ====="
python3 -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_DIR" \
    --split-prefix "" \
    --out-dir "$RUNS" --run-name aep_ajo_procedural_fixed_reas_auxloss030 \
    $CE2X_ARGS \
    2>&1 | tee "$RUNS/ce2x_deberta_aep_ajo_procedural_fixed_reas_auxloss030_stdout.log"

echo "===== DONE. Results in $RUNS/ ====="
