#!/usr/bin/env bash
# =============================================================================
# run.sh — Master evaluation launcher
#
# Evaluations are organized in 3 groups:
#
#   1. Encoder models  (CrossEncoder2x) on 697-workflow ordering task
#   2. LLM fine-tuned models (LoRA) on 697-workflow ordering task (logit-based)
#   3. Claude Sonnet 4.6 (2-shot ICL) on pair classification test set
#
# Usage:
#   bash run.sh [--group 1|2|3|all] [--gpus "0,1,2,3"] [--tp 4]
#
# Requirements:
#   - venv at /mnt/localssd/causal-embedding-research/automation/
#             internship-causal-embedding/.venv   (encoder models)
#   - venv at /mnt/localssd/causal-embedding-research/llm_trc_baseline/.venv
#             (LLM models, vllm 0.6.6)
#   - HF models cached at /mnt/localssd/.cache/huggingface
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CER="/mnt/localssd/causal-embedding-research"
AUTOMATION="$CER/automation/internship-causal-embedding"
LLM_BASE="$CER/llm_trc_baseline"

# Defaults
GROUP="all"
GPUS="0,1,2,3"
TP=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --gpus)  GPUS="$2";  shift 2 ;;
    --tp)    TP="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="/mnt/localssd/.cache/huggingface"

ENC_VENV="$AUTOMATION/.venv/bin/python"
LLM_VENV="$LLM_BASE/.venv/bin/python"

SEP="======================================================================"

run_group() { echo -e "\n$SEP\nGROUP $1: $2\n$SEP"; }

# =============================================================================
# GROUP 1 — Encoder models on 697-workflow ordering
# =============================================================================
if [[ "$GROUP" == "1" || "$GROUP" == "all" ]]; then
  run_group 1 "Encoder models — 697-workflow ordering (CrossEncoder2x)"

  cd "$AUTOMATION"
  source .venv/bin/activate

  echo "[1] v2_res_baseline + v2_fixed_alpha02_ep10 on 697 workflows..."
  "$ENC_VENV" "$HERE/code/eval_encoder_workflows.py"

  deactivate
fi

# =============================================================================
# GROUP 2 — LLM fine-tuned models on 697-workflow ordering (logit YES/NO)
# =============================================================================
if [[ "$GROUP" == "2" || "$GROUP" == "all" ]]; then
  run_group 2 "LLM fine-tuned models — 697-workflow ordering (logit YES/NO)"

  cd "$LLM_BASE"

  LORA="outputs/lora"
  LLMERE="outputs/llmere"
  LLMERE_LR1E4="outputs/llmere_lr1e4"
  LLMERE_R32="outputs/llmere_r32"
  OUT="outputs/workflow_ordering"

  run_wf() {
    echo -e "\n--- $1 ---"
    "$LLM_VENV" "$HERE/code/eval_workflows_llm.py" "${@:2}" --tp $TP --out-dir $OUT
  }

  # Finetuned LoRA models
  run_wf "Qwen2.5-14B QA1" \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapter $LORA/Qwen2.5-14B-Instruct__QA1__lora_adapter --prompt-type QA1

  run_wf "Qwen2.5-7B QA1" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LORA/Qwen2.5-7B-Instruct__QA1__lora_adapter --prompt-type QA1

  run_wf "Qwen2.5-7B P" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LORA/Qwen2.5-7B-Instruct__P__lora_adapter --prompt-type P

  run_wf "Phi-3.5-mini QA1" \
    --base-model microsoft/Phi-3.5-mini-instruct \
    --adapter $LORA/Phi-3.5-mini-instruct__QA1__lora_adapter --prompt-type QA1

  run_wf "Qwen2.5-7B rationale @1e-4" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE_LR1E4/Qwen2.5-7B-Instruct__rationale__lora_adapter \
    --prompt-type QA1 --tag Qwen2.5-7B-Instruct__rationale_lr1e4__wf_order

  run_wf "Qwen2.5-7B rationale @2e-4" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE/Qwen2.5-7B-Instruct__rationale__lora_adapter \
    --prompt-type QA1 --tag Qwen2.5-7B-Instruct__rationale_lr2e4__wf_order

  run_wf "Qwen2.5-7B rationale @r32" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE_R32/Qwen2.5-7B-Instruct__rationale__lora_adapter \
    --prompt-type QA1 --tag Qwen2.5-7B-Instruct__rationale_r32__wf_order

  run_wf "Qwen2.5-7B per-event" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE/Qwen2.5-7B-Instruct__perevent__lora_adapter \
    --prompt-type QA1 --tag Qwen2.5-7B-Instruct__perevent__wf_order

  # ICL baselines
  run_wf "Qwen2.5-7B ICL 0-shot" \
    --base-model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --shots 0

  run_wf "Qwen2.5-7B ICL 2-shot" \
    --base-model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --shots 2

  run_wf "Qwen2.5-14B ICL 0-shot" \
    --base-model Qwen/Qwen2.5-14B-Instruct --prompt-type QA1 --shots 0
fi

# =============================================================================
# GROUP 3 — Claude Sonnet 4.6 pair classification
# =============================================================================
if [[ "$GROUP" == "3" || "$GROUP" == "all" ]]; then
  run_group 3 "Claude Sonnet 4.6 — pair classification (2-shot ICL)"

  cd "$AUTOMATION"
  source .venv/bin/activate

  echo "[3] Running Claude Sonnet 4.6 on directional test set..."
  echo "    Note: requires Azure Anthropic API access. Takes ~60-90 min."
  "$ENC_VENV" "$HERE/code/eval_claude_sonnet.py"

  deactivate
fi

echo -e "\n$SEP"
echo "ALL EVALUATIONS COMPLETE"
echo "Results saved in: $CER/project_final/eval_results/"
echo "$SEP"
