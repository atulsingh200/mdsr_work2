#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder trained on new_ajo_workflows.
# Identical hyperparameters to run_ce2x_deberta.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

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

echo "===== training CE2x-deberta-v3-large on new_ajo_workflows ====="
python3 -m src.classifier.crossencoder.train_ce2x \
    --data-dir /mnt/localssd/automation/internship-causal-embedding/data/merged_newstyle_procedural \
    --split-prefix "" \
    --out-dir "$RUNS" --run-name merged_newstyle_procedural \
    $CE2X_ARGS \
    2>&1 | tee "$RUNS/ce2x_deberta_merged_newstyle_procedural_stdout.log"

echo "===== DONE. Results in $RUNS/ ====="
