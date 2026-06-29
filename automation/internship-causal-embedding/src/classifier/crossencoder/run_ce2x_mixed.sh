#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x
mkdir -p "$RUNS"

MIXED_DATA=data/aep_causal_classification_hard_neg_mixed

echo "===== building mixed dataset ====="
python3 -m src.classifier.crossencoder.build_mixed_dataset \
    --semantic-dir data/aep_causal_classification_hard_neg_semantic \
    --indomain-dir data/aep_causal_classification_hard_neg_indomain \
    --out-dir "$MIXED_DATA"

echo "===== training CE2x-mixed ====="
python3 -m src.classifier.crossencoder.train_ce2x \
    --data-dir "$MIXED_DATA" \
    --out-dir "$RUNS" --run-name ce2x_mixed \
    --backbone google-bert/bert-base-uncased --n-layers 24 \
    --epochs 3 --batch-size 32 --eval-batch-size 64 \
    --lr 2e-5 --weight-decay 0.01 --warmup-frac 0.06 \
    --grad-clip 1.0 --dropout 0.1 --seed 42 \
    2>&1 | tee "$RUNS/ce2x_mixed_stdout.log"

echo "===== DONE. Results in $RUNS/ce2x_mixed/ ====="
