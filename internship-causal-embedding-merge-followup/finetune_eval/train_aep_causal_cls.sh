#!/usr/bin/env bash
# Reverse-direction finetuning pipeline for the directional AEP causal
# classification dataset (aep_causal_cls), trained on the ENTIRE dataset.
#
# Steps:
#   1. Mine reverse pairs (train split only) — hard_neg = anchor (swap direction)
#   2. Train BiEncoder with the pairwise DPO loss on the full train set
#   3. Evaluate as binary classification: score s(text_1 -> text_2), tune a
#      threshold on the raw labeled val, report accuracy/F1/AUC on raw test
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
DS=aep_causal_cls
RDIR="finetune_eval/results_reverse"
EPOCHS="${1:-3}"
BATCH="${2:-32}"

echo "=== Step 0: build all-label-1 train/val files (idempotent) ==="
$PY data_6/aep_causal_classification/make_all_label1.py \
    --in directional_train.jsonl --out directional_train_all_label1.jsonl
$PY data_6/aep_causal_classification/make_all_label1.py \
    --in directional_val.jsonl --out directional_val_all_label1.jsonl

echo ""
echo "=== Step 1: mine reverse pairs (train split, full dataset) ==="
$PY finetune_eval/mine_hard_negatives_reverse.py --dataset "$DS" --splits train

echo ""
echo "=== Step 2: train on the entire dataset (${EPOCHS} epoch(s), batch ${BATCH}) ==="
mkdir -p "${RDIR}/${DS}"
$PY finetune_eval/train_finetune_reverse.py --dataset "$DS" \
    --epochs "$EPOCHS" --batch-size "$BATCH" 2>&1 | tee "${RDIR}/${DS}/train.log"

echo ""
echo "=== Step 3: classification eval (thresholded accuracy on raw test) ==="
$PY finetune_eval/evaluate_finetune_reverse_cls.py --dataset "$DS"

echo ""
echo "all done — results in ${RDIR}/${DS}/"
