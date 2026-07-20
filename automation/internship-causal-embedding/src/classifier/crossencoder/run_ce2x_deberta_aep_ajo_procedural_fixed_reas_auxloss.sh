#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder trained on aep_ajo_procedural_fixed_reas
# with an auxiliary reasoning-decoder head. Loss = L_BCE + alpha*L_LM, i.e.
# classification kept at full weight 1.0 and the reasoning LM as an auxiliary
# with alpha=0.7 (matches the proven proc_reasoning_lm_alpha07 recipe).
# Reasoning target is verdict-stripped. Best checkpoint selected on val AUC.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

MODEL_CE2X_DIR="$REPO_ROOT/../../new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x"
export PYTHONPATH="$MODEL_CE2X_DIR:${PYTHONPATH:-}"

DATA_DIR="$REPO_ROOT/data/aep_ajo_procedural_fixed_reas"

GPU="${GPU:-6}"          # override with: GPU=7 bash <script>
echo "[run] using GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x_deberta_reasoning
mkdir -p "$RUNS"

# New run-name so the earlier (mis-weighted) run dir is left untouched.
RUN_NAME=aep_ajo_procedural_fixed_reas_lm_alpha07

CE2X_ARGS="
  --backbone microsoft/deberta-v3-large
  --native-backbone
  --decoder-layers 4
  --cls-loss-weight 1.0
  --decoder-loss-weight 0.7
  --epochs 12
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

echo "===== training CE2x-deberta-v3-large + reasoning aux-loss (L_BCE + 0.7*L_LM) on aep_ajo_procedural_fixed_reas ====="
python3 -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_DIR" \
    --split-prefix "" \
    --out-dir "$RUNS" --run-name "$RUN_NAME" \
    $CE2X_ARGS \
    2>&1 | tee "$RUNS/ce2x_deberta_${RUN_NAME}_stdout.log"

echo "===== DONE. Results in $RUNS/$RUN_NAME ====="
