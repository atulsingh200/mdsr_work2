#!/usr/bin/env bash
# run_baselines.sh
#
# Baseline sweep: 7 datasets × 6 models = 42 training runs, distributed
# across 4 A100 GPUs.  At most 1 training job runs per GPU at a time.
# After all training completes, OOD eval runs in parallel (all 42 jobs),
# then results are aggregated into final_results.json.
#
# Usage:
#   bash /mnt/localssd/run_baselines.sh
#
# To resume after a partial run, re-run the same command; completed runs
# (run_summary.json present) and OOD evals (ood_results.json present) are
# skipped automatically.

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
REPO=/mnt/localssd/automation/internship-causal-embedding
TRAIN_SCRIPT=$REPO/src/classifier/training/train_baselines.py
EVAL_SCRIPT=$REPO/src/classifier/training/eval_ood.py
AGG_SCRIPT=/mnt/localssd/aggregate_results.py
VENV=$REPO/.venv
PYTHON=$VENV/bin/python

DATA_BASE=$REPO/data

# Dedicated sweep home: everything (models + results + logs) lives here.
SWEEP_HOME=/mnt/localssd/baseline_sweep
OUT_ROOT=$SWEEP_HOME/runs           # per-run model checkpoints + per-run JSONs
RESULTS_DIR=$SWEEP_HOME/results     # final aggregated JSON
LOGS_DIR=$SWEEP_HOME/logs           # sweep-level log

MILAN_JSON=/mnt/localssd/test_samples_milan.json
AJO_JSON=/mnt/localssd/ajo_orchestrated_workflows_flat.json

mkdir -p "$OUT_ROOT" "$RESULTS_DIR" "$LOGS_DIR"
LOG_FILE=$LOGS_DIR/sweep.log

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

# ---- Dataset registry -----------------------------------------------------
declare -A DS_PATHS
DS_PATHS[aep_causal_cls34]=$DATA_BASE/aep_causal_classification_34
DS_PATHS[aep_dataset]=$DATA_BASE/aep_dataset
DS_PATHS[aep_causal_wf_v3]=$DATA_BASE/aep_causal_workflow_v3
DS_PATHS[ajo_doc_not_tier1]=$DATA_BASE/ajo_doc_dataset_not_tier1
DS_PATHS[ajo_newstyle]=$DATA_BASE/ajo_newstyle
DS_PATHS[new_aep_wf_scrap]=$DATA_BASE/new_aep_workflow_scrap
DS_PATHS[new_ajo_workflows]=$DATA_BASE/new_ajo_workflows

# ---- Model registry -------------------------------------------------------
declare -A MODEL_IDS
MODEL_IDS[deberta-v3-large]=microsoft/deberta-v3-large
MODEL_IDS[all-mpnet-base-v2]=sentence-transformers/all-mpnet-base-v2
MODEL_IDS[all-MiniLM-L6-v2]=sentence-transformers/all-MiniLM-L6-v2
MODEL_IDS[e5-large-v2]=intfloat/e5-large-v2
MODEL_IDS[bge-base-en-v1.5]=BAAI/bge-base-en-v1.5
MODEL_IDS[bge-large-en-v1.5]=BAAI/bge-large-en-v1.5

# Micro-batch (× grad-accum below = effective batch 64 for every model). The
# large models (deberta/e5-large/bge-large) keep a small memory-safe micro-batch
# (dual encoders on A100 80GB) and accumulate to reach 64; base models fit 64.
declare -A MODEL_BATCH
MODEL_BATCH[deberta-v3-large]=8
MODEL_BATCH[all-mpnet-base-v2]=64
MODEL_BATCH[all-MiniLM-L6-v2]=64
MODEL_BATCH[e5-large-v2]=16
MODEL_BATCH[bge-base-en-v1.5]=64
MODEL_BATCH[bge-large-en-v1.5]=16

# Gradient-accumulation steps: micro-batch × accum = 64 (effective batch) for all.
declare -A MODEL_ACCUM
MODEL_ACCUM[deberta-v3-large]=8
MODEL_ACCUM[all-mpnet-base-v2]=1
MODEL_ACCUM[all-MiniLM-L6-v2]=1
MODEL_ACCUM[e5-large-v2]=4
MODEL_ACCUM[bge-base-en-v1.5]=1
MODEL_ACCUM[bge-large-en-v1.5]=4

# Extra per-model train flags. The large models (deberta/e5-large/bge-large)
# collapsed to constant output (test_acc~=0.50) under the default recipe: the
# real cause was the tiny effective batch (8/16) combined with the default LR,
# which pinned them at the symmetric 0.5 saddle on this 50/50 directional task.
# Fix (validated): effective batch 64 (via MODEL_ACCUM above) + encoder LR 1e-5
# + 0.06 warmup; the randomly-initialised MLP head keeps its 1e-3 LR. deberta
# keeps --no-amp (fp32). Base models are fine on the defaults and left unchanged.
declare -A MODEL_EXTRA
MODEL_EXTRA[deberta-v3-large]="--no-amp --lr-encoder 1e-5 --lr-head 1e-3 --warmup-frac 0.06"
MODEL_EXTRA[e5-large-v2]="--lr-encoder 1e-5 --lr-head 1e-3 --warmup-frac 0.06"
MODEL_EXTRA[bge-large-en-v1.5]="--lr-encoder 1e-5 --lr-head 1e-3 --warmup-frac 0.06"

EPOCHS=4

# ---- Run order (datasets × models) ----------------------------------------
DATASETS=(
    aep_causal_cls34
    aep_dataset
    aep_causal_wf_v3
    ajo_doc_not_tier1
    ajo_newstyle
    new_aep_wf_scrap
    new_ajo_workflows
)
MODELS=(
    deberta-v3-large
    all-mpnet-base-v2
    all-MiniLM-L6-v2
    e5-large-v2
    bge-base-en-v1.5
    bge-large-en-v1.5
)

GPU_COUNT=4

# ---- GPU queue helpers -----------------------------------------------------
declare -a GPU_PIDS
for _g in 0 1 2 3; do GPU_PIDS[$_g]=-1; done

# Print index of the first GPU whose job has finished (blocks until one is free)
wait_for_free_gpu() {
    while true; do
        for g in 0 1 2 3; do
            local pid=${GPU_PIDS[$g]:-1}
            if [[ $pid -eq -1 ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$g"
                return 0
            fi
        done
        sleep 30
    done
}

# ---- Phase 0: Pre-download all backbones (serial) --------------------------
# Downloading once up front avoids multiple parallel training jobs racing to
# fetch the same model into the HF cache, and fails fast on a bad model id.
log "=== PHASE 0: Pre-downloading ${#MODELS[@]} backbones (serial) ==="
for model in "${MODELS[@]}"; do
    log "  fetch ${MODEL_IDS[$model]}"
    "$PYTHON" - "${MODEL_IDS[$model]}" <<'PYEOF' >> "$LOG_FILE" 2>&1
import sys
from transformers import AutoModel, AutoTokenizer
mid = sys.argv[1]
AutoTokenizer.from_pretrained(mid)
AutoModel.from_pretrained(mid)
print(f"cached {mid}")
PYEOF
done
log "=== PHASE 0 complete ==="

# ---- Phase 1: Training runs ------------------------------------------------
log "=== PHASE 1: Training (${#DATASETS[@]} datasets × ${#MODELS[@]} models = $(( ${#DATASETS[@]} * ${#MODELS[@]} )) runs) ==="

for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_dir=$OUT_ROOT/$ds/$model
        if [[ -f "$run_dir/run_summary.json" ]]; then
            log "SKIP (already done): $ds/$model"
            continue
        fi

        g=$(wait_for_free_gpu)
        mkdir -p "$run_dir"
        log "START GPU${g}: $ds/$model  batch=${MODEL_BATCH[$model]}"

        CUDA_VISIBLE_DEVICES=$g "$PYTHON" "$TRAIN_SCRIPT" \
            --data-dir    "${DS_PATHS[$ds]}" \
            --base-model  "${MODEL_IDS[$model]}" \
            --out-dir     "$run_dir" \
            --dataset-slug "$ds" \
            --model-slug   "$model" \
            --gpu-id       "$g" \
            --batch-size   "${MODEL_BATCH[$model]}" \
            --grad-accum   "${MODEL_ACCUM[$model]:-1}" \
            --eval-batch-size 64 \
            --epochs       $EPOCHS \
            --head-type    mlp \
            --head-hidden-dims 512,128 \
            ${MODEL_EXTRA[$model]:-} \
            > "$run_dir/stdout.log" 2>&1 &

        GPU_PIDS[$g]=$!
    done
done

log "All jobs submitted — waiting for completion ..."
wait
log "=== PHASE 1 complete ==="

# ---- Phase 2: OOD evaluation -----------------------------------------------
# Bounded to one eval per GPU (max 4 concurrent), reusing the GPU queue.
# This avoids the CPU/GPU memory spike of launching all 42 evals at once.
log "=== PHASE 2: OOD evaluation (max ${GPU_COUNT} concurrent, 1 per GPU) ==="

for _g in 0 1 2 3; do GPU_PIDS[$_g]=-1; done

for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_dir=$OUT_ROOT/$ds/$model
        if [[ ! -f "$run_dir/best.pt" ]]; then
            log "WARN: no best.pt for $ds/$model — skipping OOD eval"
            continue
        fi
        if [[ -f "$run_dir/ood_results.json" ]]; then
            log "SKIP OOD (already done): $ds/$model"
            continue
        fi

        g=$(wait_for_free_gpu)
        log "OOD eval GPU${g}: $ds/$model"
        CUDA_VISIBLE_DEVICES=$g "$PYTHON" "$EVAL_SCRIPT" \
            --run-dir     "$run_dir" \
            --milan-json  "$MILAN_JSON" \
            --ajo-json    "$AJO_JSON" \
            > "$run_dir/ood_stdout.log" 2>&1 &
        GPU_PIDS[$g]=$!
    done
done

log "All OOD evals submitted — waiting for completion ..."
wait
log "=== PHASE 2 complete ==="

# ---- Phase 3: Aggregation --------------------------------------------------
log "=== PHASE 3: Aggregating results ==="
"$PYTHON" "$AGG_SCRIPT" \
    --root "$OUT_ROOT" \
    --out  "$RESULTS_DIR/final_results.json"
log "Done. Final results at $RESULTS_DIR/final_results.json"
