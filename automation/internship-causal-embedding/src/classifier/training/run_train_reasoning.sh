#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

python3 -m src.classifier.training.train_reasoning \
  --data-dir data/reasoning_data \
  --out-dir runs/reasoning \
  --run-name reasoning_bge_small \
  --base-model BAAI/bge-small-en-v1.5 \
  --head-hidden-dims 512,128 \
  --head-dropout 0.1 \
  --head-activation gelu \
  --head-norm layer \
  --epochs 5 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr-encoder 2e-5 \
  --lr-head 1e-3 \
  --warmup-frac 0.1 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --seed 42 \
  --reason-weight 1
