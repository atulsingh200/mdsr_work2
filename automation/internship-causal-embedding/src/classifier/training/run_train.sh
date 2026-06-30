#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

python3 -m src.classifier.training.train \
  --data-dir /mnt/localssd/automation/internship-causal-embedding/data/new_aep_workflow_scrap \
  --out-dir runs/classifier \
  --run-name merged_workflows_all-mpnet-base-v2 \
  --base-model sentence-transformers/all-mpnet-base-v2 \
  --head-type mlp \
  --head-hidden-dims 512,128 \
  --head-dropout 0.1 \
  --head-activation gelu \
  --head-norm layer \
  --epochs 3 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr-encoder 2e-5 \
  --lr-head 1e-3 \
  --warmup-frac 0.1 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --max-seq-len 524 \
  --seed 42
