#!/usr/bin/env bash
# CDEv2 full pipeline: mine → train → evaluate
#
# Run from the kl/ directory:
#   bash v2/run_v2.sh
#
# GPU requirements: single A100-80GB is sufficient.
# For a quick smoke-test: add --total-steps 500 --val-every 100 to train_v2.py
#
# Phases:
#   0. (skip if already done) prepare_data.py + original mine_hard_negatives.py
#   1. Mine K=8 hard negatives (train + test)
#   2. Train CDEv2 (5000 steps, 3-phase schedule, batch=64)
#   3. Evaluate CDEv2 vs BiEncoder — print side-by-side table + 10% target check
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$KL_DIR")")"

cd "$KL_DIR"

echo "============================================================"
echo "  CDEv2 pipeline"
echo "  KL dir   : $KL_DIR"
echo "  Project  : $PROJECT_ROOT"
echo "============================================================"

# ── Step 0: Ensure base data is prepared ────────────────────────────────────
TRAIN_PAIRS="results/aep_causal/train_pairs.jsonl"
if [ ! -f "$TRAIN_PAIRS" ]; then
    echo "[step 0] preparing data …"
    uv run python prepare_data.py
else
    echo "[step 0] data already prepared, skipping"
fi

# ── Step 1: Mine K=8 hard negatives ─────────────────────────────────────────
HARD_NEG_K8="results/aep_causal/hard_negatives_k8.npy"
if [ ! -f "$HARD_NEG_K8" ]; then
    echo ""
    echo "[step 1] mining K=8 hard negatives …"
    uv run python v2/mine_hard_neg_v2.py --k 8 --split both
else
    echo "[step 1] hard_negatives_k8.npy already exists, skipping"
fi

# ── Step 2: Train CDEv2 ──────────────────────────────────────────────────────
echo ""
echo "[step 2] training CDEv2 …"
uv run python v2/train_v2.py \
    --backbone "google-bert/bert-base-uncased" \
    --proj-dim 256 \
    --batch-size 64 \
    --phase-a-steps 1000 \
    --phase-b-steps 2000 \
    --total-steps 5000 \
    --backbone-lr 2e-5 \
    --head-lr 1e-4 \
    --temperature-lr 1e-3 \
    --lambda-entropy 0.1 \
    --lambda-antisym 0.5 \
    --lambda-bpr 0.3 \
    --lambda-sigma-lb 0.05 \
    --sigma-lb -3.0 \
    --val-every 200 \
    --early-stopping-patience 6 \
    --bf16 \
    --seed 42

# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
echo ""
echo "[step 3] evaluating CDEv2 …"
uv run python v2/evaluate_v2.py --also-v1

echo ""
echo "============================================================"
echo "  Pipeline complete. Results in results/aep_causal/cde_v2_eval.json"
echo "============================================================"
