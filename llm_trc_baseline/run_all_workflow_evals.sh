#!/usr/bin/env bash
# Run all LLM workflow ordering evaluations sequentially.
# Uses GPUs 0-3 (4x tensor parallel) for each run.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv/bin/python"
SCRIPT="eval_workflows_llm.py"
OUT="outputs/workflow_ordering"
LORA="outputs/lora"
LLMERE="outputs/llmere"
LLMERE_LR1E4="outputs/llmere_lr1e4"
LLMERE_R32="outputs/llmere_r32"
GPUS="0,1,2,3"
TP=4

export HF_HOME="/mnt/localssd/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="$GPUS"

run() {
    echo ""
    echo "======================================================================"
    echo "RUNNING: $*"
    echo "======================================================================"
    $VENV $SCRIPT "$@" 2>&1
}

# ── Finetuned LoRA models ──────────────────────────────────────────────────────

# 1. Qwen2.5-14B QA1
run --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapter $LORA/Qwen2.5-14B-Instruct__QA1__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT

# 2. Qwen2.5-7B QA1
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LORA/Qwen2.5-7B-Instruct__QA1__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT

# 3. Qwen2.5-7B P
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LORA/Qwen2.5-7B-Instruct__P__lora_adapter \
    --prompt-type P --tp $TP --out-dir $OUT

# 4. Phi-3.5-mini QA1
run --base-model microsoft/Phi-3.5-mini-instruct \
    --adapter $LORA/Phi-3.5-mini-instruct__QA1__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT

# 5. Qwen2.5-7B rationale @1e-4
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE_LR1E4/Qwen2.5-7B-Instruct__rationale__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT \
    --tag Qwen2.5-7B-Instruct__rationale_lr1e4__wf_order

# 6. Qwen2.5-7B rationale @2e-4
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE/Qwen2.5-7B-Instruct__rationale__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT \
    --tag Qwen2.5-7B-Instruct__rationale_lr2e4__wf_order

# 7. Qwen2.5-7B rationale @r32
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE_R32/Qwen2.5-7B-Instruct__rationale__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT \
    --tag Qwen2.5-7B-Instruct__rationale_r32__wf_order

# 8. Qwen2.5-7B per-event O(n)
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter $LLMERE/Qwen2.5-7B-Instruct__perevent__lora_adapter \
    --prompt-type QA1 --tp $TP --out-dir $OUT \
    --tag Qwen2.5-7B-Instruct__perevent__wf_order

# ── ICL baselines (no adapter) ─────────────────────────────────────────────────

# 9. Qwen2.5-7B ICL 0-shot QA1
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --prompt-type QA1 --shots 0 --tp $TP --out-dir $OUT

# 10. Qwen2.5-7B ICL 2-shot QA1
run --base-model Qwen/Qwen2.5-7B-Instruct \
    --prompt-type QA1 --shots 2 --tp $TP --out-dir $OUT

# 11. Qwen2.5-14B ICL 0-shot QA1
run --base-model Qwen/Qwen2.5-14B-Instruct \
    --prompt-type QA1 --shots 0 --tp $TP --out-dir $OUT

echo ""
echo "======================================================================"
echo "ALL DONE. Results in $OUT/"
echo "======================================================================"
