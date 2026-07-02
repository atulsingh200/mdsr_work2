#!/usr/bin/env bash
# Sweep EVERY causal-LM model under pythia_train, rerank follow-ups by
# full-sequence perplexity, and report the best hit@k / recall@k (k=1,3,5).
set -euo pipefail

REPO=/mnt/localssd/automation/internship-causal-embedding
SCRIPT=/mnt/localssd/z_followupq/sweep_ppl_models.py
INPUT=/mnt/localssd/z_followupq/match_results.jsonl
MODELS_ROOT="$REPO/pythia_train"
OUT_DIR=/mnt/localssd/z_followupq/sweep_ppl

# Auto-select GPU with most free memory
if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -n1 | awk -F, '{gsub(/ /,"",$1); print $1}')
  echo "[run] auto-selected GPU $GPU"
else
  echo "[run] using explicitly set GPU $GPU"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

source "$REPO/.venv/bin/activate"

mkdir -p "$OUT_DIR"

echo ""
echo "===== Perplexity sweep over ALL models in $MODELS_ROOT ====="
# --include-checkpoints also evaluates intermediate checkpoint-* dirs so the
# reported "best possible" covers every saved model state. Drop the flag to
# score only the 5 final model directories.
python3 "$SCRIPT" \
  --input        "$INPUT" \
  --models-root  "$MODELS_ROOT" \
  --out-dir      "$OUT_DIR" \
  --include-checkpoints \
  --batch-size 32 \
  --sort-metric hit@1 \
  2>&1 | tee "$OUT_DIR/sweep.log"

echo ""
echo "===== DONE. Per-model outputs + summary in $OUT_DIR ====="
