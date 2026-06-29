#!/usr/bin/env bash
# =============================================================================
# run_ce2x_full.sh
#
# Full pipeline for CrossEncoder-2x on 5 datasets:
#   1. Prepare capped splits (200k train / 20k val / 20k test) for every dataset
#   2. Mine semantic hard negatives on train + val + test splits
#   3. Generate RANDOM negative index files (same shape, random indices)
#   4. Train on RANDOM negatives
#   5. Train on SEMANTIC HARD negatives
#   6. Evaluate both checkpoints on random-neg + hard-neg metrics + pool retrieval
#
# Cap = ceiling: datasets smaller than the cap use ALL their rows (no upsampling).
# Only workflow (2.1M train / 260k val / 260k test) is actually capped.
# aep_causal, followupqg, multiwoz_v24, qrecc are all under the ceiling and
# are trained / evaluated on their FULL splits.
#   train ceiling: 200,000  |  val ceiling: 20,000  |  test ceiling: 20,000
#
# Usage (from repo root):
#   bash scripts/run_ce2x_full.sh 2>&1 | tee ce2x_results/logs/master.log
#
# Output layout:
#   ce2x_data/<dataset>/          — capped JSONL splits + hard_negatives.npy
#   ce2x_data_random/<dataset>/   — capped JSONL splits + random_negatives.npy
#   ce2x_results/random/<dataset>/  — checkpoints + eval from random-neg training
#   ce2x_results/hardneg/<dataset>/ — checkpoints + eval from hard-neg training
#   ce2x_results/logs/            — per-step log files
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UV="/opt/conda/bin/uv"
PYTHON="$UV run python"

# ── Caps (applied uniformly to every dataset) ─────────────────────────────────
TRAIN_CAP=200000
VAL_CAP=20000
TEST_CAP=20000

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR="$REPO_ROOT/ce2x_data"                  # semantic-hard-neg data dir
RANDOM_DATA_DIR="$REPO_ROOT/ce2x_data_random"    # random-neg data dir
RESULTS_RANDOM="$REPO_ROOT/ce2x_results/random"
RESULTS_HARDNEG="$REPO_ROOT/ce2x_results/hardneg"
LOG_DIR="$REPO_ROOT/ce2x_results/logs"

# Source for workflow (too large to live in this repo)
WORKFLOW_SRC="/mnt/localssd/internship-causal-embedding-merge-followup/data_6/workflow"

# ── Training hyperparams ──────────────────────────────────────────────────────
BACKBONE="google-bert/bert-base-uncased"
N_LAYERS=24
EPOCHS=10
BATCH_SIZE=64
GRAD_ACCUM=2          # effective batch = 128
LR=2e-5
MAX_LENGTH=256
SEED=42
EARLY_STOP_PATIENCE=2
K_NEG=4               # negatives per anchor

# ── Eval hyperparams ─────────────────────────────────────────────────────────
EVAL_BATCH=32         # smaller — cross-encoder pairs cost 2× vs bi-encoder
POOL_SIZE=1000

DATASETS=(aep_causal followupqg multiwoz_v24 qrecc workflow)

mkdir -p "$DATA_DIR" "$RANDOM_DATA_DIR" "$RESULTS_RANDOM" "$RESULTS_HARDNEG" "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# STEP 1 — PREPARE CAPPED SPLITS
# =============================================================================

log "========== STEP 1: Prepare capped splits =========="

prepare_dataset() {
    local ds="$1"
    local src_train src_val src_test

    case "$ds" in
        aep_causal)
            src_train="$REPO_ROOT/data_6/aep_causal/train.jsonl"
            src_val="$REPO_ROOT/data_6/aep_causal/val.jsonl"
            src_test="$REPO_ROOT/data_6/aep_causal/test.jsonl"
            ;;
        followupqg)
            src_train="$REPO_ROOT/data_6/followupqg/train.jsonl"
            src_val="$REPO_ROOT/data_6/followupqg/valid.jsonl"
            src_test="$REPO_ROOT/data_6/followupqg/test.jsonl"
            ;;
        multiwoz_v24)
            src_train="$REPO_ROOT/data_6/multiwoz_v24/train.jsonl"
            src_val="$REPO_ROOT/data_6/multiwoz_v24/val.jsonl"
            src_test="$REPO_ROOT/data_6/multiwoz_v24/test.jsonl"
            ;;
        qrecc)
            src_train="$REPO_ROOT/data_6/qrecc/train.jsonl"
            src_val=""   # no val source; train script will hold out 5%
            src_test="$REPO_ROOT/data_6/qrecc/test.jsonl"
            ;;
        workflow)
            src_train="$WORKFLOW_SRC/train_pairs.jsonl"
            src_val="$WORKFLOW_SRC/dev_pairs.jsonl"
            src_test="$WORKFLOW_SRC/test_pairs.jsonl"
            ;;
    esac

    local out="$DATA_DIR/$ds"
    mkdir -p "$out"

    log "  [$ds] train"
    $PYTHON scripts/_sample_jsonl.py "$src_train" "$out/train_pairs.jsonl" "$TRAIN_CAP" "$SEED"

    if [ -n "$src_val" ]; then
        log "  [$ds] val"
        $PYTHON scripts/_sample_jsonl.py "$src_val" "$out/val_pairs.jsonl" "$VAL_CAP" "$SEED"
    else
        log "  [$ds] val — no source; train script will hold out 5% of train"
    fi

    log "  [$ds] test"
    $PYTHON scripts/_sample_jsonl.py "$src_test" "$out/test_pairs.jsonl" "$TEST_CAP" "$SEED"
}

for ds in "${DATASETS[@]}"; do
    log "--- Preparing $ds ---"
    prepare_dataset "$ds"
done

log "Step 1 done."

# =============================================================================
# STEP 2 — MINE SEMANTIC HARD NEGATIVES
# Runs on train, val, test for each dataset.
# Outputs: hard_negatives.npy, val_hard_negatives.npy, test_hard_negatives.npy
# =============================================================================

log "========== STEP 2: Mine semantic hard negatives =========="

mine_split() {
    local ds="$1" split="$2" pairs_file="$3"

    [ -f "$pairs_file" ] || { log "  [$ds/$split] no file — skip"; return; }

    local hn_out
    if [ "$split" = "train" ]; then
        hn_out="$DATA_DIR/$ds/hard_negatives.npy"
    else
        hn_out="$DATA_DIR/$ds/${split}_hard_negatives.npy"
    fi

    if [ -f "$hn_out" ]; then
        log "  [$ds/$split] already mined — skip"
        return
    fi

    log "  [$ds/$split] mining → $hn_out"
    $PYTHON scripts/_mine_negatives.py "$pairs_file" "$hn_out" "$K_NEG" \
        2>&1 | tee -a "$LOG_DIR/mine_${ds}_${split}.log"
}

for ds in "${DATASETS[@]}"; do
    log "--- Mining $ds ---"
    mine_split "$ds" "train" "$DATA_DIR/$ds/train_pairs.jsonl"
    mine_split "$ds" "val"   "$DATA_DIR/$ds/val_pairs.jsonl"
    mine_split "$ds" "test"  "$DATA_DIR/$ds/test_pairs.jsonl"
done

log "Step 2 done."

# =============================================================================
# STEP 3 — GENERATE RANDOM NEGATIVE INDEX FILES
# train_ce2x.py always requires a hard_negatives.npy.
# For random-neg training we create a .npy of the same shape filled with
# randomly sampled indices — training code is identical, only quality differs.
# =============================================================================

log "========== STEP 3: Generate random negative index files =========="

make_random_npy() {
    local ds="$1" split="$2" pairs_file="$3"
    [ -f "$pairs_file" ] || return

    local out_dir="$RANDOM_DATA_DIR/$ds"
    mkdir -p "$out_dir"

    # Mirror JSONL files into random data dir
    cp "$pairs_file" "$out_dir/$(basename "$pairs_file")"

    local hn_out
    if [ "$split" = "train" ]; then
        hn_out="$out_dir/hard_negatives.npy"
    else
        hn_out="$out_dir/${split}_hard_negatives.npy"
    fi

    if [ -f "$hn_out" ]; then
        log "  [$ds/$split] random .npy exists — skip"
        return
    fi

    log "  [$ds/$split] generating random negative indices → $hn_out"
    $PYTHON scripts/_make_random_npy.py "$pairs_file" "$hn_out" "$K_NEG" "$SEED"
}

for ds in "${DATASETS[@]}"; do
    log "--- Random indices: $ds ---"
    make_random_npy "$ds" "train" "$DATA_DIR/$ds/train_pairs.jsonl"
    make_random_npy "$ds" "val"   "$DATA_DIR/$ds/val_pairs.jsonl"
    make_random_npy "$ds" "test"  "$DATA_DIR/$ds/test_pairs.jsonl"
done

log "Step 3 done."

# =============================================================================
# STEP 4 — TRAIN ON RANDOM NEGATIVES
# =============================================================================

log "========== STEP 4: Train on RANDOM negatives =========="

train_model() {
    local ds="$1" data_dir="$2" out_dir="$3" logfile="$4"
    mkdir -p "$out_dir"

    if [ -f "$out_dir/checkpoint_best.pt" ]; then
        log "  [$ds] checkpoint exists — skip"
        return
    fi

    $PYTHON my_work/kl/cross_encoder_2x/train_ce2x.py \
        --dataset        "$ds" \
        --data-dir       "$data_dir" \
        --out-dir        "$out_dir" \
        --backbone       "$BACKBONE" \
        --n-layers       "$N_LAYERS" \
        --epochs         "$EPOCHS" \
        --batch-size     "$BATCH_SIZE" \
        --grad-accum     "$GRAD_ACCUM" \
        --lr             "$LR" \
        --max-length     "$MAX_LENGTH" \
        --seed           "$SEED" \
        --early-stop-patience "$EARLY_STOP_PATIENCE" \
        --num-workers    2 \
        2>&1 | tee "$logfile"
}

for ds in "${DATASETS[@]}"; do
    log "--- Training random: $ds ---"
    train_model "$ds" \
        "$RANDOM_DATA_DIR" \
        "$RESULTS_RANDOM/$ds" \
        "$LOG_DIR/train_random_${ds}.log"
done

log "Step 4 done."

# =============================================================================
# STEP 5 — TRAIN ON SEMANTIC HARD NEGATIVES
# =============================================================================

log "========== STEP 5: Train on HARD negatives =========="

for ds in "${DATASETS[@]}"; do
    log "--- Training hardneg: $ds ---"
    train_model "$ds" \
        "$DATA_DIR" \
        "$RESULTS_HARDNEG/$ds" \
        "$LOG_DIR/train_hardneg_${ds}.log"
done

log "Step 5 done."

# =============================================================================
# STEP 6 — EVALUATE both checkpoints
# =============================================================================

log "========== STEP 6: Evaluate =========="

evaluate() {
    local ds="$1" ckpt="$2" out_dir="$3" logfile="$4"

    if [ ! -f "$ckpt" ]; then
        log "  [$ds] SKIP — checkpoint not found: $ckpt"
        return
    fi

    log "  [$ds] evaluating checkpoint: $ckpt"
    mkdir -p "$out_dir"

    $PYTHON my_work/kl/cross_encoder_2x/evaluate_ce2x.py \
        --dataset     "$ds" \
        --data-dir    "$DATA_DIR" \
        --checkpoint  "$ckpt" \
        --out-dir     "$out_dir" \
        --pool-size   "$POOL_SIZE" \
        --n-negatives "$K_NEG" \
        --batch-size  "$EVAL_BATCH" \
        --max-length  "$MAX_LENGTH" \
        --seed        "$SEED" \
        2>&1 | tee "$logfile"
}

for ds in "${DATASETS[@]}"; do
    log "--- Evaluating $ds (random-neg model) ---"
    evaluate "$ds" \
        "$RESULTS_RANDOM/$ds/checkpoint_best.pt" \
        "$RESULTS_RANDOM/$ds" \
        "$LOG_DIR/eval_random_${ds}.log"

    log "--- Evaluating $ds (hardneg model) ---"
    evaluate "$ds" \
        "$RESULTS_HARDNEG/$ds/checkpoint_best.pt" \
        "$RESULTS_HARDNEG/$ds" \
        "$LOG_DIR/eval_hardneg_${ds}.log"
done

log "Step 6 done."

# =============================================================================
# SUMMARY TABLE
# =============================================================================

log "========== FINAL SUMMARY =========="

$PYTHON - "$RESULTS_RANDOM" "$RESULTS_HARDNEG" <<'PYEOF'
import json, sys
from pathlib import Path

results_random  = Path(sys.argv[1])
results_hardneg = Path(sys.argv[2])
datasets = ["aep_causal", "followupqg", "multiwoz_v24", "qrecc", "workflow"]

def load(p):
    return json.loads(p.read_text()) if p.exists() else {}

H = f"{'Dataset':<16} {'Train_neg':<10} {'Eval_neg':<12} {'AUC':>7} {'P@1':>7} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}"
print("\n" + "=" * len(H))
print(H)
print("-" * len(H))

for ds in datasets:
    for train_label, base in [("random", results_random), ("hardneg", results_hardneg)]:
        for eval_neg, fname in [("random", "eval.json"), ("hard_neg", "eval_hardneg.json")]:
            r = load(base / ds / fname)
            if not r:
                continue
            auc = r.get("auc", 0)
            p1  = r.get("precision_at_1", r.get("p_at_1", 0))
            mrr = r.get("mrr", 0)
            rk  = r.get("recall_at_k", {})
            print(f"{ds:<16} {train_label:<10} {eval_neg:<12} "
                  f"{auc:>7.4f} {p1:>7.4f} {mrr:>7.4f} "
                  f"{rk.get(1,0):>7.4f} {rk.get(5,0):>7.4f} {rk.get(10,0):>7.4f}")

print("=" * len(H))
PYEOF

log "All done."
log "  Hard-neg results : $RESULTS_HARDNEG"
log "  Random results   : $RESULTS_RANDOM"
log "  Logs             : $LOG_DIR"
