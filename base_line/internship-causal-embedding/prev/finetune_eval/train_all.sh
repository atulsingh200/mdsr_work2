#!/usr/bin/env bash
# Train all 5 fine-tuning runs sequentially. Logs go to finetune_eval/results/<ds>/train.log.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

# (dataset, epochs) — small datasets get more epochs since each epoch is cheap.
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
    log="finetune_eval/results/${ds}/train.log"
    echo "[$(date +%H:%M:%S)] training ${ds} for ${ep} epoch(s) — log: ${log}"
    $PY finetune_eval/train_finetune.py --dataset "${ds}" --epochs "${ep}" --batch-size 32 \
        > "${log}" 2>&1
    echo "[$(date +%H:%M:%S)] done ${ds}"
done

echo "all training runs finished"
