#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for ds in qrecc workflow; do
    log="finetune_eval/results/${ds}/train.log"
    echo "[$(date +%H:%M:%S)] training ${ds} — log: ${log}"
    $PY finetune_eval/train_finetune.py --dataset "${ds}" --epochs 2 --batch-size 32 \
        > "${log}" 2>&1
    echo "[$(date +%H:%M:%S)] done ${ds}"
done
echo "remaining training runs finished"
