#!/usr/bin/env bash
# Train BiEncoder with hard negatives ONLY (no in-batch negatives).
# Ablation to isolate whether the BiEncoder vs CrossEncoder performance
# difference is architectural or due to the extra in-batch negatives.
#
# Mining artifacts (train_pairs.jsonl, hard_negatives.npy) are reused from
# finetune_eval/results/<dataset>/ — no need to re-mine.
# Checkpoints go to finetune_eval/results/<dataset>_no_inbatch/.
#
# Usage:
#   bash finetune_eval/train_all_no_inbatch.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
RESULTS=finetune_eval/results

# (dataset, epochs) — same as train_all.sh
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
    log="${RESULTS}/${ds}_no_inbatch/train.log"
    mkdir -p "${RESULTS}/${ds}_no_inbatch"
    echo "[$(date +%H:%M:%S)] training ${ds} (no in-batch) for ${ep} epoch(s) — log: ${log}"
    $PY finetune_eval/train_finetune.py \
        --dataset "${ds}" \
        --epochs "${ep}" \
        --batch-size 32 \
        --results-dir "${RESULTS}" \
        --no-inbatch \
        --results-suffix "_no_inbatch" \
        > "${log}" 2>&1
    echo "[$(date +%H:%M:%S)] done ${ds}"
done

echo "all no-inbatch training runs finished"
