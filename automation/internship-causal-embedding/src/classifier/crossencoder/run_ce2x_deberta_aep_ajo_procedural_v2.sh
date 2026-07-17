#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder trained on merged_gapdecay_procedural
# (procedural 19.3k + AEP 13k + AJO 6.3k gap-decay, ~38.6k train).
# Same hyperparameters as run_ce2x_deberta_new_ajo.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

# model_ce2x.py lives outside the package; the trainer's hardcoded _CE2X_MODEL_DIR
# is stale after the repo move, and it is imported at module-load (before args are
# parsed), so make it importable via PYTHONPATH instead of --model-ce2x-dir.
MODEL_CE2X_DIR="$REPO_ROOT/../../new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x"
export PYTHONPATH="$MODEL_CE2X_DIR:${PYTHONPATH:-}"

DATA_DIR="$REPO_ROOT/data/aep_ajo_procedural_workflow_v2"

if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -n1 | awk -F, '{gsub(/ /,"",$1); print $1}')
  echo "[run] auto-selected GPU $GPU (most free memory)"
else
  echo "[run] using explicitly set GPU $GPU"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x_deberta
mkdir -p "$RUNS"

CE2X_ARGS="
  --backbone microsoft/deberta-v3-large
  --native-backbone
  --epochs 3
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

echo "===== training CE2x-deberta-v3-large on aep_ajo_procedural_workflow_v2 ====="
python3 -m src.classifier.crossencoder.train_ce2x \
    --data-dir "$DATA_DIR" \
    --split-prefix "" \
    --out-dir "$RUNS" --run-name aep_ajo_procedural_workflow_v2 \
    $CE2X_ARGS \
    2>&1 | tee "$RUNS/ce2x_deberta_aep_ajo_procedural_workflow_v2_stdout.log"

echo "===== DONE. Results in $RUNS/ ====="
