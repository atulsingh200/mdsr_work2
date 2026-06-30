#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

python3 -m src.classifier.training.train \
  --data-dir /mnt/localssd/automation/internship-causal-embedding/data/new_aep_workflow_scrap \
  --out-dir runs/classifier \
  --run-name new_aep_workflow_scrap_snowflake-arctic-embed-l \
  --base-model Snowflake/snowflake-arctic-embed-l \
  --head-type mlp \
  --head-hidden-dims 1024,256 \
  --head-dropout 0.1 \
  --head-activation gelu \
  --head-norm layer \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --lr-encoder 1e-5 \
  --lr-head 1e-3 \
  --warmup-frac 0.1 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --max-seq-len 512 \
  --seed 42
