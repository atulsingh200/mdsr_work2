#!/usr/bin/env bash
# End-to-end ensemble cross-encoder pipeline:
#   1. mine semantic hard negs (train+val)        -> data/...hard_neg_semantic
#   2. mine in-domain hard negs (train+val)        -> data/...hard_neg_indomain
#   3. copy untouched test split into both dirs
#   4. train CE-semantic and CE-indomain (BERT-base)
#   5. ensemble blend + eval on the untouched test set
#
# Usage:
#   bash src/classifier/crossencoder/run_all.sh
#   GPU=1 bash src/classifier/crossencoder/run_all.sh          # pick GPU
#   SKIP_MINING=1 bash src/classifier/crossencoder/run_all.sh  # reuse mined data
set -euo pipefail

GPU="${GPU:-0}"                     # which CUDA device to use
SKIP_MINING="${SKIP_MINING:-0}"     # set 1 if mined data already exists
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

SRC=data/aep_causal_classification_34
SEM=data/aep_causal_classification_hard_neg_semantic
IND=data/aep_causal_classification_hard_neg_indomain
RUNS=runs/crossencoder
mkdir -p "$SEM" "$IND" "$RUNS"

echo "===== GPU / CPU memory before start ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
free -g | head -2

# Safety net: abort if available RAM is dangerously low (CPU OOM crashes the box).
# Training peak is ~10 GB; require a comfortable 30 GB headroom before starting.
AVAIL_GB="$(free -g | awk '/^Mem:/{print $7}')"
MIN_GB=30
if [ "${AVAIL_GB:-0}" -lt "$MIN_GB" ]; then
  echo "ABORT: only ${AVAIL_GB} GB RAM available (< ${MIN_GB} GB safety floor)." >&2
  echo "Free memory before running to avoid a CPU-OOM system crash." >&2
  exit 1
fi
echo "RAM check OK: ${AVAIL_GB} GB available (need >= ${MIN_GB} GB)."

if [ "$SKIP_MINING" = "1" ]; then
  echo "===== SKIP_MINING=1: reusing existing mined data ====="
else
  # -------------------------------------------------------------------------
  # Step 1: semantic hard negatives (train + val)
  # -------------------------------------------------------------------------
  for split in train val; do
    echo "===== [semantic] mining $split ====="
    python3 -m src.classifier.build_hard_neg_semantic \
      --source-dir "$SRC" --out-dir "$SEM" --split "$split" --topk 3
  done

  # -------------------------------------------------------------------------
  # Step 2: in-domain hard negatives (train + val), via trained 2-BERT
  # -------------------------------------------------------------------------
  for split in train val; do
    echo "===== [indomain] mining $split ====="
    python3 -m src.classifier.build_hard_neg_indomain \
      --source-dir "$SRC" --out-dir "$IND" --split "$split" \
      --ckpt runs/classifier/best_mlp_bge_small/best.pt --topk 3
  done
fi

# ---------------------------------------------------------------------------
# Step 3: copy the UNTOUCHED test split into both mined dirs
# ---------------------------------------------------------------------------
cp -v "$SRC/directional_test.jsonl" "$SEM/directional_test.jsonl"
cp -v "$SRC/directional_test.jsonl" "$IND/directional_test.jsonl"
cmp "$SRC/directional_test.jsonl" "$SEM/directional_test.jsonl" && echo "test == semantic test OK"
cmp "$SRC/directional_test.jsonl" "$IND/directional_test.jsonl" && echo "test == indomain test OK"

echo "===== GPU / CPU memory before training ====="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv || true
free -g | head -2

# ---------------------------------------------------------------------------
# Step 4: train the two cross-encoders (BERT-base)
# ---------------------------------------------------------------------------
CE_ARGS="--base-model bert-base-uncased --epochs 3 --batch-size 32 \
  --eval-batch-size 64 --lr-encoder 2e-5 --lr-head 1e-3 --warmup-frac 0.1 \
  --weight-decay 0.01 --grad-clip 1.0 --dropout 0.1 --seed 42"

echo "===== training CE-semantic ====="
python3 -m src.classifier.crossencoder.train_ce \
  --data-dir "$SEM" --out-dir "$RUNS" --run-name ce_semantic $CE_ARGS \
  2>&1 | tee "$RUNS/ce_semantic_stdout.log"

echo "===== training CE-indomain ====="
python3 -m src.classifier.crossencoder.train_ce \
  --data-dir "$IND" --out-dir "$RUNS" --run-name ce_indomain $CE_ARGS \
  2>&1 | tee "$RUNS/ce_indomain_stdout.log"

# ---------------------------------------------------------------------------
# Step 5: ensemble blend + eval
# ---------------------------------------------------------------------------
echo "===== ensemble eval ====="
python3 -m src.classifier.crossencoder.ensemble_eval \
  --semantic-run "$RUNS/ce_semantic" \
  --indomain-run "$RUNS/ce_indomain" \
  --out "$RUNS/ensemble_metrics.json" \
  2>&1 | tee "$RUNS/ensemble_stdout.log"

echo "===== DONE. Results in $RUNS/ensemble_metrics.json ====="
