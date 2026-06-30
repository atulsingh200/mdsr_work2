#!/usr/bin/env bash
# Rerank follow-up questions using both CE2x DeBERTa checkpoints and
# evaluate hit@k / recall@k for k in {1, 3, 5}.
set -euo pipefail

REPO=/mnt/localssd/automation/internship-causal-embedding
SCRIPT=/mnt/localssd/z_followupq/rerank_ce2x.py
INPUT=/mnt/localssd/z_followupq/match_results.jsonl
OUT_DIR=/mnt/localssd/z_followupq
RUNS="$REPO/runs/crossencoder2x_deberta"

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
echo "===== CE2x DeBERTa | new_aep_workflow_scrap ====="
python3 "$SCRIPT" \
  --input   "$INPUT" \
  --model   "$RUNS/new_aep_workflow_scrap/best.pt" \
  --config  "$RUNS/new_aep_workflow_scrap/config.json" \
  --output  "$OUT_DIR/reranked_ce2x_aep.jsonl" \
  --batch-size 32

echo ""
echo "===== CE2x DeBERTa | new_ajo_workflows ====="
python3 "$SCRIPT" \
  --input   "$INPUT" \
  --model   "$RUNS/new_ajo_workflows/best.pt" \
  --config  "$RUNS/new_ajo_workflows/config.json" \
  --output  "$OUT_DIR/reranked_ce2x_ajo.jsonl" \
  --batch-size 32

echo ""
echo "===== DONE. Outputs in $OUT_DIR ====="
