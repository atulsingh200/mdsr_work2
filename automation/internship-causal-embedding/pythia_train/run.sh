#!/usr/bin/env bash
# Full pipeline:
#   Step 1 — build merged dataset (transcripts + C4), save to DATASET_DIR
#   Step 2 — train Pythia-410M on that dataset on a single GPU
#
# Usage: bash run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python3

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME="EleutherAI/pythia-410m"
TRANSCRIPT_PATH="${SCRIPT_DIR}/../data/scrap_transcript/aep_transcripts.json"
DATASET_DIR="${SCRIPT_DIR}/merged_dataset"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"   # override with: OUTPUT_DIR=... bash run.sh
GPU=0                 # which single GPU to use

C4_RATIO=0.30         # 70% transcript tokens, 30% C4 tokens
MAX_LENGTH=1024
NUM_EPOCHS=40
BATCH_SIZE=4          # single GPU  →  4 × grad_accum=16 = effective batch 64
GRAD_ACCUM=16         # large batch → fewer, smoother updates (less overfitting)
LR=2e-5               # conservative LR for stable domain adaptation
WARMUP_STEPS=20
SAVE_STEPS=200
EVAL_STEPS=20
LOGGING_STEPS=5       # fine-grained logging for a usable loss curve
SEED=42
# ──────────────────────────────────────────────────────────────────────────────

# ── Step 1: prepare data (skip if already done) ───────────────────────────────
if [ -d "${DATASET_DIR}/train" ] && [ -d "${DATASET_DIR}/eval" ]; then
    echo "Dataset already exists at ${DATASET_DIR} — skipping data preparation."
    echo "  (delete ${DATASET_DIR} to rebuild)"
else
    echo "=================================================="
    echo " Step 1 — Build merged dataset"
    echo "  transcripts : ${TRANSCRIPT_PATH}"
    echo "  c4 ratio    : ${C4_RATIO}"
    echo "  output      : ${DATASET_DIR}"
    echo "=================================================="
    $PYTHON "${SCRIPT_DIR}/prepare_data.py" \
        --transcript_path "${TRANSCRIPT_PATH}" \
        --output_dir      "${DATASET_DIR}" \
        --model_name      "${MODEL_NAME}" \
        --c4_ratio        "${C4_RATIO}" \
        --max_length      "${MAX_LENGTH}" \
        --seed            "${SEED}"
    echo "Data preparation complete."
fi

# ── Step 2: train ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Step 2 — Train Pythia-410M (1 GPU)"
echo "  dataset : ${DATASET_DIR}"
echo "  output  : ${OUTPUT_DIR}"
echo "  gpu     : ${GPU}"
echo "  epochs  : ${NUM_EPOCHS}  |  lr: ${LR}"
echo "=================================================="

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" $PYTHON \
    "${SCRIPT_DIR}/train.py" \
    --model_name                  "${MODEL_NAME}" \
    --dataset_dir                 "${DATASET_DIR}" \
    --output_dir                  "${OUTPUT_DIR}" \
    --num_train_epochs            "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate               "${LR}" \
    --warmup_steps                "${WARMUP_STEPS}" \
    --save_steps                  "${SAVE_STEPS}" \
    --eval_steps                  "${EVAL_STEPS}" \
    --logging_steps               "${LOGGING_STEPS}" \
    --seed                        "${SEED}"

echo "Training complete. Model saved to: ${OUTPUT_DIR}"

# ── Step 3: plot training curves ──────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Step 3 — Plot training curves"
echo "=================================================="
$PYTHON "${SCRIPT_DIR}/plot_training.py" \
    --log_history "${OUTPUT_DIR}/log_history.json" \
    --output_dir  "${OUTPUT_DIR}"
echo "Plots saved to: ${OUTPUT_DIR}/training_curves.png"
