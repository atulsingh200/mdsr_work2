#!/usr/bin/env bash
# End-to-end contrastive cross-encoder pipeline:
#   1. Mine contrastive groups (train + val)   -> data/aep_causal_allpos_34
#   2. Copy untouched test split into data dir
#   3. Train CE with InfoNCE loss (BERT-base)
#
# Usage:
#   bash src/classifier/crossencoder/run_contrastive.sh
#   GPU=1 bash src/classifier/crossencoder/run_contrastive.sh          # pick GPU
#   SKIP_MINING=1 bash src/classifier/crossencoder/run_contrastive.sh  # reuse mined data
set -euo pipefail

GPU="${GPU:-0}"
SKIP_MINING="${SKIP_MINING:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

SRC=data/aep_causal_classification_34
CDATA=data/aep_causal_allpos_34
RUNS=runs/crossencoder_contrastive
mkdir -p "$CDATA" "$RUNS"

echo "===== GPU / CPU memory before start ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
free -g | head -2

# Safety net: abort if available RAM is dangerously low.
AVAIL_GB="$(free -g | awk '/^Mem:/{print $7}')"
MIN_GB=30
if [ "${AVAIL_GB:-0}" -lt "$MIN_GB" ]; then
  echo "ABORT: only ${AVAIL_GB} GB RAM available (< ${MIN_GB} GB safety floor)." >&2
  exit 1
fi
echo "RAM check OK: ${AVAIL_GB} GB available."

# ---------------------------------------------------------------------------
# Step 1: mine contrastive groups (train + val)
# ---------------------------------------------------------------------------
if [ "$SKIP_MINING" = "1" ]; then
  echo "===== SKIP_MINING=1: reusing existing mined data ====="
else
  for split in train val; do
    echo "===== [contrastive] mining $split ====="
    python3 -m src.classifier.crossencoder.build_contrastive_data \
      --source-dir "$SRC" \
      --out-dir    "$CDATA" \
      --split      "$split" \
      --model      BAAI/bge-large-en-v1.5 \
      --topk-hard  2 \
      --n-easy     2 \
      --seed       42
  done
fi

# ---------------------------------------------------------------------------
# Step 2: copy UNTOUCHED test split (original binary labels for eval)
# ---------------------------------------------------------------------------
cp -v "$SRC/directional_test.jsonl" "$CDATA/directional_test.jsonl"
cmp "$SRC/directional_test.jsonl" "$CDATA/directional_test.jsonl" \
  && echo "test split copy OK"

echo "===== GPU / CPU memory before training ====="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv || true
free -g | head -2

# ---------------------------------------------------------------------------
# Step 3: train with InfoNCE contrastive loss
# ---------------------------------------------------------------------------
CE_ARGS="--base-model bert-base-uncased --epochs 3 --batch-size 32 \
  --eval-batch-size 64 --lr-encoder 2e-5 --lr-head 1e-3 --warmup-frac 0.1 \
  --weight-decay 0.01 --grad-clip 1.0 --dropout 0.1 --temperature 0.07 --seed 42"

echo "===== training CE-contrastive ====="
python3 -m src.classifier.crossencoder.train_contrastive \
  --contrastive-data-dir "$CDATA" \
  --eval-data-dir        "$SRC" \
  --out-dir "$RUNS" --run-name ce_contrastive \
  $CE_ARGS \
  2>&1 | tee "$RUNS/ce_contrastive_stdout.log"

echo "===== DONE. Results in $RUNS/ce_contrastive/ ====="
