#!/usr/bin/env bash
# Evaluate the no-inbatch BiEncoder checkpoints.
# Runs both eval scripts (random-neg AUC/P@1/MRR and hard-neg AUC/P@1).
#
# Usage:
#   bash finetune_eval/evaluate_no_inbatch.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
RESULTS=finetune_eval/results
RESULTS_NI="${RESULTS}_no_inbatch"   # convenience alias (not used; suffix is per-dataset)

echo "[$(date +%H:%M:%S)] evaluating no-inbatch checkpoints (random-neg AUC + retrieval)"
$PY finetune_eval/evaluate_finetune.py \
    --dataset all \
    --results-dir "${RESULTS}" \
    --results-suffix "_no_inbatch"

echo
echo "[$(date +%H:%M:%S)] evaluating no-inbatch checkpoints (hard-neg AUC)"
$PY finetune_eval/evaluate_finetune.py \
    --dataset all \
    --results-dir "${RESULTS}" \
    --results-suffix "_no_inbatch"

echo
echo "[$(date +%H:%M:%S)] done"
