#!/usr/bin/env bash
# Full pipeline — GPT-2 medium (355M) counterpart of run.sh.
#   Step 1 — build merged dataset, re-tokenised with the GPT-2 tokenizer,
#            reusing the EXACT same documents as the Pythia run (fair comparison)
#   Step 2 — train gpt2-medium on that dataset on a single GPU
#   Step 3 — plot training curves
#
# Usage: bash run_gpt2.sh
#        OUTPUT_DIR=/some/dir bash run_gpt2.sh   # to override output location
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python3

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME="gpt2-medium"                                   # 355M params (closest GPT-2 to Pythia-410M)
TRANSCRIPT_PATH="${SCRIPT_DIR}/../data/scrap_transcript/aep_transcripts.json"
REUSE_DOCS="${SCRIPT_DIR}/merged_dataset/merged_docs.jsonl"  # identical text to the Pythia run
DATASET_DIR="${SCRIPT_DIR}/merged_dataset_gpt2"            # GPT-2-tokenised (different vocab → own dir)
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_gpt2}"      # override with: OUTPUT_DIR=... bash run_gpt2.sh
GPU=0                 # which single GPU to use

C4_RATIO=0.30         # only used if REUSE_DOCS is missing
MAX_LENGTH=1024       # gpt2-medium max context is exactly 1024
NUM_EPOCHS=40
BATCH_SIZE=4          # single GPU  →  4 × grad_accum=4 = effective batch 16
GRAD_ACCUM=4          # SMALL batch → ~16 steps/epoch → many smooth gradient updates
LR=1e-5               # GPT-2 is LR-sensitive; lower LR keeps steps stable
WARMUP_STEPS=50       # ramp over the first ~3 epochs
WEIGHT_DECAY=0.01     # mild regularisation, standard for GPT-2 fine-tuning
SAVE_STEPS=320
EVAL_STEPS=40
LOGGING_STEPS=20      # each plotted point averages ~20 steps → smooth curve
SEED=42
# ──────────────────────────────────────────────────────────────────────────────

# ── Step 1: prepare data (skip if already done) ───────────────────────────────
if [ -d "${DATASET_DIR}/train" ] && [ -d "${DATASET_DIR}/eval" ]; then
    echo "Dataset already exists at ${DATASET_DIR} — skipping data preparation."
    echo "  (delete ${DATASET_DIR} to rebuild)"
else
    echo "=================================================="
    echo " Step 1 — Build merged dataset (GPT-2 tokenizer)"
    echo "  output : ${DATASET_DIR}"
    echo "=================================================="
    if [ -f "${REUSE_DOCS}" ]; then
        echo "  reusing identical documents from: ${REUSE_DOCS}"
        $PYTHON "${SCRIPT_DIR}/prepare_data.py" \
            --transcript_path  "${TRANSCRIPT_PATH}" \
            --output_dir       "${DATASET_DIR}" \
            --model_name       "${MODEL_NAME}" \
            --reuse_docs_jsonl "${REUSE_DOCS}" \
            --max_length       "${MAX_LENGTH}" \
            --seed             "${SEED}"
    else
        echo "  (${REUSE_DOCS} not found — building fresh from transcripts + C4)"
        $PYTHON "${SCRIPT_DIR}/prepare_data.py" \
            --transcript_path "${TRANSCRIPT_PATH}" \
            --output_dir      "${DATASET_DIR}" \
            --model_name      "${MODEL_NAME}" \
            --c4_ratio        "${C4_RATIO}" \
            --max_length      "${MAX_LENGTH}" \
            --seed            "${SEED}"
    fi
    echo "Data preparation complete."
fi

# ── Step 2: train ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Step 2 — Train gpt2-medium 355M (1 GPU)"
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
    --weight_decay                "${WEIGHT_DECAY}" \
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
