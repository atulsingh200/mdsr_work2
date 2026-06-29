#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x
mkdir -p "$RUNS"

CE2X_ARGS="--backbone google-bert/bert-base-uncased --n-layers 24 --epochs 3 --batch-size 1 --eval-batch-size 8 --lr 2e-5 --weight-decay 0.01 --warmup-frac 0.06 --grad-clip 1.0 --dropout 0.1 --seed 42"

echo "===== training CE2x-semantic ====="
python3 -m src.classifier.crossencoder.train_ce2x \
    --data-dir data/new_aep_workflow_scrap \
    --out-dir "$RUNS" --run-name new_aep_workflow_scrap \
    --step-key step \
    $CE2X_ARGS \
    2>&1 | tee "$RUNS/ce2x_semantic_stdout.log"

echo "===== DONE. Results in $RUNS/ ====="
