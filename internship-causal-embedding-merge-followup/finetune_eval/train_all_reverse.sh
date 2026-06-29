#!/usr/bin/env bash
# Full reverse-direction finetuning pipeline for all 5 datasets.
# Steps:
#   1. Mine reverse pairs (train + test splits) — no kNN, just swap anchor as hard neg
#   2. Train BiEncoder with reverse hard negatives for each dataset
#   3. Evaluate: AUC/P@1 with 2 candidates (forward a->b vs reverse b->a)
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
RDIR="finetune_eval/results_reverse"

echo "=== Step 1: mine reverse pairs (train + test) ==="
$PY finetune_eval/mine_hard_negatives_reverse.py --dataset all --splits train test

echo ""
echo "=== Step 2: train ==="

declare -a JOBS=(
    "aep_causal:3"
    "followupqg:3"
    "multiwoz_v24:2"
    "qrecc:2"
    "workflow:2"
)

for job in "${JOBS[@]}"; do
    ds="${job%%:*}"
    ep="${job##*:}"
    log="${RDIR}/${ds}/train.log"
    mkdir -p "${RDIR}/${ds}"
    echo "[$(date +%H:%M:%S)] training ${ds} for ${ep} epoch(s) — log: ${log}"
    $PY finetune_eval/train_finetune_reverse.py --dataset "${ds}" --epochs "${ep}" --batch-size 32 \
        > "${log}" 2>&1
    echo "[$(date +%H:%M:%S)] done ${ds}"
done

echo ""
echo "=== Step 3: evaluate (forward vs reverse, 2 candidates) ==="
$PY finetune_eval/evaluate_finetune_reverse.py --dataset all

echo ""
echo "all done — results in ${RDIR}/"
