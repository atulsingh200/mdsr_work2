#!/usr/bin/env bash
# End-to-end CDE training + evaluation pipeline for aep_causal.
#
# Usage:
#   cd /mnt/localssd/base_line/internship-causal-embedding
#   bash ../my_work/kl/run_pipeline.sh          # default settings
#   bash ../my_work/kl/run_pipeline.sh --bf16   # A100 fast path
#
# All output goes to my_work/kl/results/aep_causal/.

set -euo pipefail

VENV=".venv/bin/python"
KL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Extra args forwarded to train_cde.py (e.g. --bf16)
EXTRA="${*:-}"

# ── Step 1: prepare data ──────────────────────────────────────────────────
log "Step 1/4: prepare data"
$VENV "$KL_DIR/prepare_data.py"

# ── Step 2: mine hard negatives (train + test) ────────────────────────────
log "Step 2/4: mine hard negatives"
$VENV "$KL_DIR/mine_hard_negatives.py" --split both --k 4

# ── Step 3: train CDE ─────────────────────────────────────────────────────
log "Step 3/4: train CDE"
$VENV "$KL_DIR/train_cde.py" \
    --backbone "google-bert/bert-base-uncased" \
    --proj-dim 128 \
    --batch-size 32 \
    --phase-a-steps 500 \
    --total-steps 2000 \
    --backbone-lr 2e-5 \
    --head-lr 1e-4 \
    --temperature-init 0.05 \
    --max-seq-length 256 \
    --val-every 200 \
    --early-stopping-patience 5 \
    $EXTRA 2>&1 | tee "$KL_DIR/results/aep_causal/train.log"

# ── Step 4: evaluate & compare with BiEncoder ─────────────────────────────
log "Step 4/4: evaluate"
$VENV "$KL_DIR/evaluate_cde.py" 2>&1 | tee "$KL_DIR/results/aep_causal/eval.log"

log "Done. Results in $KL_DIR/results/aep_causal/"
