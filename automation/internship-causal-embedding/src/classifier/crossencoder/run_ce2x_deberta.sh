#!/usr/bin/env bash
# DeBERTa-v3-large cross-encoder (24 layers, ~304M params, native backbone).
# Replaces the stacked BERT-base setup; uses --native-backbone to skip layer-repeat logic.
#
# Key tuning differences from run_ce2x_org.sh (bert-base):
#   --lr 1e-5          (bert-base used 2e-5; DeBERTa is sensitive to high LRs)
#   --batch-size 8     (larger model; grad-accum 4 keeps effective batch = 32)
#   --grad-accum 4
#   --warmup-frac 0.1  (slightly longer warmup helps DeBERTa fine-tuning)
#   --native-backbone  (bypasses BERT-stacking, uses AutoModel.from_pretrained)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

# Pick the GPU with the most free memory unless one is set explicitly via GPU=N.
# (GPU 0 on this box is frequently occupied by other processes.)
if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -n1 | awk -F, '{gsub(/ /,"",$1); print $1}')
  echo "[run] auto-selected GPU $GPU (most free memory)"
else
  echo "[run] using explicitly set GPU $GPU"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

RUNS=runs/crossencoder2x_deberta
mkdir -p "$RUNS"

CE2X_ARGS="
  --backbone microsoft/deberta-v3-large
  --native-backbone
  --epochs 3
  --batch-size 16
  --eval-batch-size 16
  --grad-accum 4
  --lr 1e-5
  --weight-decay 0.01
  --warmup-frac 0.1
  --grad-clip 1.0
  --dropout 0.1
  --seed 42
"

echo "===== training CE2x-deberta-v3-large ====="
python3 -m src.classifier.crossencoder.train_ce2x \
    --data-dir /mnt/localssd/automation/internship-causal-embedding/data/aep_causal_workflow_v3 \
    --split-prefix "directional_" \
    --out-dir "$RUNS" --run-name aep_causal_workflow_v3  \
    $CE2X_ARGS \
    2>&1 | tee "$RUNS/ce2x_deberta_aep_causal_workflow_v3.log"

echo "===== DONE. Results in $RUNS/ ====="
