#!/usr/bin/env bash
# =============================================================================
# launch_all_runs.sh  —  9 runs across 8 GPUs (GPU 7 runs 2 sequentially)
#
# GPU 0  v2_res_baseline       no decoder  ep=5   data=v2_res
# GPU 1  v2_res_alpha07_ep12   alpha=0.7   ep=12  data=v2_res
# GPU 2  v2_fixed_alpha07_ep12 alpha=0.7   ep=12  data=v2_fixed
# GPU 3  v2_fixed_alpha02_ep05 alpha=0.2   ep=5   data=v2_fixed
# GPU 4  v2_fixed_alpha02_ep10 alpha=0.2   ep=10  data=v2_fixed
# GPU 5  v2_fixed_alpha05_ep05 alpha=0.5   ep=5   data=v2_fixed
# GPU 6  v2_fixed_alpha05_ep10 alpha=0.5   ep=10  data=v2_fixed
# GPU 7  v2_fixed_alpha07_ep05 alpha=0.7   ep=5   data=v2_fixed  (then ep=10 below)
# GPU 7  v2_fixed_alpha07_ep10 alpha=0.7   ep=10  data=v2_fixed  (sequential after ep=5)
#
# Usage:
#   bash launch_all_runs.sh
#
# Monitor a run:
#   tail -f /tmp/opencode/train_<run_name>.log
#   tail -f runs/crossencoder2x_deberta_reasoning/<run_name>/train.log
#
# Results when done:
#   for f in runs/crossencoder2x_deberta_reasoning/*/test_metrics.json; do
#     n=$(basename $(dirname $f))
#     python3 -c "import json; d=json.load(open('$f')); print(f'$n  acc={d[\"acc\"]:.4f}  auc={d[\"auc\"]:.4f}')"
#   done
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python3"
MODEL_DIR="/mnt/localssd/causal-embedding-research/new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x"
export PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}"

LOGS="/tmp/opencode"
mkdir -p "$LOGS"

DATA_RES="$REPO_ROOT/data/aep_ajo_procedural_workflow_v2_res"
DATA_FIXED="$REPO_ROOT/data/aep_ajo_procedural_workflow_v2_fixed_reas"
OUT_R="$REPO_ROOT/runs/crossencoder2x_deberta_reasoning"
OUT_B="$REPO_ROOT/runs/crossencoder2x_deberta"
mkdir -p "$OUT_R" "$OUT_B"

# Shared DeBERTa args
DEBERTA="--backbone microsoft/deberta-v3-large --native-backbone"
OPT="--batch-size 16 --eval-batch-size 16 --grad-accum 4 --lr 1e-5 --weight-decay 0.01 --warmup-frac 0.1 --grad-clip 1.0 --dropout 0.1 --seed 42 --step-key step"

echo "======================================================================"
echo " Launching 9 runs across GPUs 0-7"
echo "======================================================================"

# ── GPU 0 ── baseline (no reasoning decoder, alpha=0, ep=5) ───────────────
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m src.classifier.crossencoder.train_ce2x \
    --data-dir "$DATA_RES" --split-prefix "" \
    --out-dir "$OUT_B" --run-name "v2_res_baseline" \
    $DEBERTA $OPT \
    --epochs 5 \
    > "$LOGS/train_v2_res_baseline.log" 2>&1 &
echo "[GPU 0] PID=$!  v2_res_baseline          (no decoder, ep=5)"

# ── GPU 1 ── v2_res  alpha=0.7  ep=12 ────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_RES" --split-prefix "" \
    --out-dir "$OUT_R" --run-name "v2_res_alpha07_ep12" \
    $DEBERTA $OPT \
    --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.7 \
    --epochs 12 \
    > "$LOGS/train_v2_res_alpha07_ep12.log" 2>&1 &
echo "[GPU 1] PID=$!  v2_res_alpha07_ep12       (alpha=0.7, ep=12)"

# ── GPU 2 ── v2_fixed  alpha=0.7  ep=12 ──────────────────────────────────
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_FIXED" --split-prefix "" \
    --out-dir "$OUT_R" --run-name "v2_fixed_alpha07_ep12" \
    $DEBERTA $OPT \
    --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.7 \
    --epochs 12 \
    > "$LOGS/train_v2_fixed_alpha07_ep12.log" 2>&1 &
echo "[GPU 2] PID=$!  v2_fixed_alpha07_ep12     (alpha=0.7, ep=12)"

# ── GPU 3 ── v2_fixed  alpha=0.2  ep=5 ───────────────────────────────────
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_FIXED" --split-prefix "" \
    --out-dir "$OUT_R" --run-name "v2_fixed_alpha02_ep05" \
    $DEBERTA $OPT \
    --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.2 \
    --epochs 5 \
    > "$LOGS/train_v2_fixed_alpha02_ep05.log" 2>&1 &
echo "[GPU 3] PID=$!  v2_fixed_alpha02_ep05     (alpha=0.2, ep=5)"

# ── GPU 4 ── v2_fixed  alpha=0.2  ep=10 ──────────────────────────────────
CUDA_VISIBLE_DEVICES=4 "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_FIXED" --split-prefix "" \
    --out-dir "$OUT_R" --run-name "v2_fixed_alpha02_ep10" \
    $DEBERTA $OPT \
    --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.2 \
    --epochs 10 \
    > "$LOGS/train_v2_fixed_alpha02_ep10.log" 2>&1 &
echo "[GPU 4] PID=$!  v2_fixed_alpha02_ep10     (alpha=0.2, ep=10)"

# ── GPU 5 ── v2_fixed  alpha=0.5  ep=5 ───────────────────────────────────
CUDA_VISIBLE_DEVICES=5 "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_FIXED" --split-prefix "" \
    --out-dir "$OUT_R" --run-name "v2_fixed_alpha05_ep05" \
    $DEBERTA $OPT \
    --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.5 \
    --epochs 5 \
    > "$LOGS/train_v2_fixed_alpha05_ep05.log" 2>&1 &
echo "[GPU 5] PID=$!  v2_fixed_alpha05_ep05     (alpha=0.5, ep=5)"

# ── GPU 6 ── v2_fixed  alpha=0.5  ep=10 ──────────────────────────────────
CUDA_VISIBLE_DEVICES=6 "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
    --data-dir "$DATA_FIXED" --split-prefix "" \
    --out-dir "$OUT_R" --run-name "v2_fixed_alpha05_ep10" \
    $DEBERTA $OPT \
    --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.5 \
    --epochs 10 \
    > "$LOGS/train_v2_fixed_alpha05_ep10.log" 2>&1 &
echo "[GPU 6] PID=$!  v2_fixed_alpha05_ep10     (alpha=0.5, ep=10)"

# ── GPU 7 ── v2_fixed  alpha=0.7  ep=5  then  ep=10  (sequential) ─────────
(
    export CUDA_VISIBLE_DEVICES=7

    "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
        --data-dir "$DATA_FIXED" --split-prefix "" \
        --out-dir "$OUT_R" --run-name "v2_fixed_alpha07_ep05" \
        $DEBERTA $OPT \
        --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.7 \
        --epochs 5 \
        > "$LOGS/train_v2_fixed_alpha07_ep05.log" 2>&1

    "$PYTHON" -m src.classifier.crossencoder.train_ce2x_reasoning \
        --data-dir "$DATA_FIXED" --split-prefix "" \
        --out-dir "$OUT_R" --run-name "v2_fixed_alpha07_ep10" \
        $DEBERTA $OPT \
        --decoder-layers 4 --cls-loss-weight 1.0 --decoder-loss-weight 0.7 \
        --epochs 10 \
        > "$LOGS/train_v2_fixed_alpha07_ep10.log" 2>&1
) &
echo "[GPU 7] PID=$!  v2_fixed_alpha07_ep05 -> ep10  (alpha=0.7, sequential)"

echo ""
echo "======================================================================"
echo " All 9 runs launched."
echo " Logs : /tmp/opencode/train_<run_name>.log"
echo " Ckpts: runs/crossencoder2x_deberta_reasoning/<run_name>/"
echo "======================================================================"
