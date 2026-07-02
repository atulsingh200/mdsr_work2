#!/usr/bin/env bash
# Rerank follow-up questions by GPT-2 full-sequence perplexity of
# "question + space + follow_up", then evaluate hit@k / recall@k (k=1,3,5).
set -euo pipefail

REPO=/mnt/localssd/automation/internship-causal-embedding
SCRIPT=/mnt/localssd/z_followupq/rerank_gpt2_ppl.py
INPUT=/mnt/localssd/z_followupq/match_results.jsonl
OUT_DIR=/mnt/localssd/z_followupq
MODEL_DIR="$REPO/pythia_train/output_gpt2_smooth"

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

echo ""
echo "===== GPT-2 perplexity rerank | output_gpt2_smooth ====="
python3 "$SCRIPT" \
  --input     "$INPUT" \
  --model-dir "$MODEL_DIR" \
  --output    "$OUT_DIR/reranked_gpt2_ppl.jsonl" \
  --batch-size 32

echo ""
echo "===== DONE. Output in $OUT_DIR/reranked_gpt2_ppl.jsonl ====="
