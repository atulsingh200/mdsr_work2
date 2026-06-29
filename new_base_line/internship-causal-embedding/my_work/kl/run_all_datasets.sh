#!/usr/bin/env bash
# Run the full CDEv2 + CE-rerank pipeline on all 5 datasets in PARALLEL,
# one dataset per GPU (4x A100 available; aep_causal and followupqg share GPU 0).
#
# Each dataset is pinned to a single GPU via CUDA_VISIBLE_DEVICES so
# DataParallel is never triggered (it would hide custom model methods).
#
# Logs go to my_work/kl/logs/<dataset>.log
#
# Usage (from project root):
#   bash my_work/kl/run_all_datasets.sh
#
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root
PY=.venv/bin/python
LOG_DIR="my_work/kl/logs"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_dataset() {
    local ds=$1
    local gpu=$2
    local cpool=$3
    local ce_epochs=$4
    local logfile="${LOG_DIR}/${ds}.log"

    CKPT="finetune_eval/results/${ds}/checkpoint_best.pt"
    if [ ! -f "${CKPT}" ]; then
        echo "[SKIP] ${ds}: missing BiEncoder checkpoint at ${CKPT}" | tee -a "${logfile}"
        return
    fi

    KL_RESULTS="my_work/kl/results/${ds}"
    mkdir -p "${KL_RESULTS}"

    HN_FILE="${KL_RESULTS}/hard_negatives_k8.npy"
    # CE dirs passed to rerank_eval must be relative to KL_DIR (my_work/kl/)
    CE_DIR="my_work/kl/cross_encoder/results/${ds}_rerank"
    CE_DIR_REL="cross_encoder/results/${ds}_rerank"

    CPOOL_ARG=""
    if [ "${cpool}" -gt 0 ]; then
        CPOOL_ARG="--candidate-pool ${cpool}"
    fi

    {
        echo "[$(date +%H:%M:%S)] ===== START ${ds} on GPU ${gpu} ====="

        # Step 0: Stage data — copy train/val/test pairs + hard_negatives from
        # finetune_eval/results/<ds>/ into kl/results/<ds>/
        echo "[$(date +%H:%M:%S)] [${ds}] Step 0: staging data"
        SRC="finetune_eval/results/${ds}"
        for fn in train_pairs.jsonl val_pairs.jsonl test_pairs.jsonl \
                  hard_negatives.npy val_hard_negatives.npy test_hard_negatives.npy; do
            [ -f "${SRC}/${fn}" ] && cp -u "${SRC}/${fn}" "${KL_RESULTS}/${fn}"
        done
        # qrecc has no val split — hold out 10% of train as val
        if [ ! -f "${KL_RESULTS}/val_pairs.jsonl" ]; then
            $PY - "${KL_RESULTS}" <<'PYEOF'
import sys, random, pathlib
d = pathlib.Path(sys.argv[1])
lines = (d / "train_pairs.jsonl").read_text().splitlines()
rng = random.Random(42)
idx = list(range(len(lines))); rng.shuffle(idx)
n_val = max(1, int(len(lines) * 0.10))
val_idx = set(idx[:n_val])
(d / "val_pairs.jsonl").write_text("\n".join(lines[i] for i in sorted(val_idx)) + "\n")
(d / "train_pairs.jsonl").write_text("\n".join(lines[i] for i in range(len(lines)) if i not in val_idx) + "\n")
print(f"[stage] held out {n_val} of {len(lines)} pairs as val")
PYEOF
        fi

        # Step 1: Mine K=8 hard negatives
        echo "[$(date +%H:%M:%S)] [${ds}] Step 1: mining K=8 hard negatives"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/v2/mine_hard_neg_v2.py \
            --dataset "${ds}" --split both --k 8

        # Step 2: Train CDEv2 retriever (single-GPU, run5c config)
        echo "[$(date +%H:%M:%S)] [${ds}] Step 2: training CDEv2 retriever"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/v2/train_v2.py \
            --dataset "${ds}" \
            --init-from-biencoder "${CKPT}" \
            --proj-dim 768 \
            --pooling cls \
            --mu-identity \
            --log-sigma-init 3.0 \
            --kl-scale dim \
            --init-cos-weight 1.0 \
            --backbone-lr 2e-5 \
            --batch-size 64 \
            --max-seq-length 256 \
            --phase-a-steps 3000 \
            --phase-b-steps 0 \
            --total-steps 3000 \
            --lambda-entropy 0 \
            --lambda-antisym 0 \
            --lambda-bpr 0 \
            --hard-neg-file "${HN_FILE}" \
            --out-suffix _run5c

        # Step 3a: Train CE v1 — MiniLM semantic hard negatives (K=4)
        echo "[$(date +%H:%M:%S)] [${ds}] Step 3a: training CE v1 (MiniLM negs)"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/cross_encoder/train_ce.py \
            --dataset "${ds}" \
            --data-dir "finetune_eval/results" \
            --epochs "${ce_epochs}" \
            --out-dir "${CE_DIR}"

        # Step 3b: Mine in-domain hard negatives from the trained CDEv2 retriever
        CE_DIR_V2="my_work/kl/cross_encoder/results/${ds}_rerank_v2"
        CE_DIR_V2_REL="cross_encoder/results/${ds}_rerank_v2"
        INDOMAIN_DIR="my_work/kl/cross_encoder/data_indomain/${ds}"
        echo "[$(date +%H:%M:%S)] [${ds}] Step 3b: mining in-domain negatives from CDEv2"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/v2/mine_indomain_negs.py \
            --dataset "${ds}" \
            --cde-suffix _run5c \
            --k 4

        # Step 3c: Train CE v2 — in-domain CDEv2-retrieved hard negatives (6 epochs)
        echo "[$(date +%H:%M:%S)] [${ds}] Step 3c: training CE v2 (in-domain negs)"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/cross_encoder/train_ce.py \
            --dataset "${ds}" \
            --data-dir "my_work/kl/cross_encoder/data_indomain" \
            --epochs 6 \
            --out-dir "${CE_DIR_V2}"

        # Step 4a: Tune on VAL — ensemble of both CEs
        echo "[$(date +%H:%M:%S)] [${ds}] Step 4a: rerank eval on VAL (CE v1 + v2 ensemble)"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/v2/rerank_eval.py \
            --dataset "${ds}" --split val \
            --cde-suffix _run5c \
            --ce-dir "${CE_DIR_REL}" "${CE_DIR_V2_REL}" \
            --pool 20 50 --blend 0.3 0.5 0.7 \
            ${CPOOL_ARG}

        # Step 4b: Report on TEST — ensemble of both CEs
        echo "[$(date +%H:%M:%S)] [${ds}] Step 4b: rerank eval on TEST (CE v1 + v2 ensemble)"
        CUDA_VISIBLE_DEVICES=${gpu} $PY my_work/kl/v2/rerank_eval.py \
            --dataset "${ds}" --split test \
            --cde-suffix _run5c \
            --ce-dir "${CE_DIR_REL}" "${CE_DIR_V2_REL}" \
            --pool 20 50 --blend 0.3 0.5 0.7 \
            ${CPOOL_ARG}

        echo "[$(date +%H:%M:%S)] ===== DONE ${ds} ====="
    } > "${logfile}" 2>&1
}

# Launch all 5 datasets in parallel, pinned to specific GPUs.
# GPU assignment: 0=aep_causal, 1=followupqg, 2=multiwoz_v24, 3=qrecc, 3=workflow
# (workflow and qrecc share GPU 3 — they run sequentially via subshell ordering)
#                ds              gpu  cpool  ce_epochs
run_dataset  aep_causal           0    0      4  &
run_dataset  followupqg           1    0      4  &
run_dataset  multiwoz_v24         2    0      4  &
run_dataset  qrecc                3    0      4  &
run_dataset  workflow             3    1000   4  &

log "All 5 jobs launched. Tailing logs..."
log "  aep_causal   → ${LOG_DIR}/aep_causal.log"
log "  followupqg   → ${LOG_DIR}/followupqg.log"
log "  multiwoz_v24 → ${LOG_DIR}/multiwoz_v24.log"
log "  qrecc        → ${LOG_DIR}/qrecc.log"
log "  workflow     → ${LOG_DIR}/workflow.log"

# Wait for all background jobs and report exit codes
FAILED=0
for job in $(jobs -p); do
    if ! wait "${job}"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
log "===== ALL DONE (${FAILED} failed) ====="
for ds in aep_causal followupqg multiwoz_v24 qrecc workflow; do
    echo "--- ${ds} last lines ---"
    tail -3 "${LOG_DIR}/${ds}.log" 2>/dev/null || echo "(no log)"
done
