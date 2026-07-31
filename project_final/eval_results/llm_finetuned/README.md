# eval_results/llm_finetuned/

Workflow ordering accuracy for all LLM models on `aep_workflows_eval.json` (697 workflows, 2/3/4-step).

**Method:** P(YES) from YES-token logprob used as directional causal score.
YES and NO are single tokens for all models (Qwen2.5: YES=14004, NO=8996; Phi: YES=22483, NO=11698).

## Results

| Model | 2-step | 3-step | 4-step | **Total** |
|-------|:------:|:------:|:------:|:---------:|
| Qwen2.5-14B QA1 (FT) | 100.0% | 93.4% | 86.5% | **92.8%** |
| Qwen2.5-7B rationale @2e-4 (FT) | 100.0% | 94.3% | 79.4% | **91.7%** |
| Qwen2.5-7B rationale @1e-4 (FT) | 100.0% | 92.1% | 80.0% | **90.5%** |
| Qwen2.5-7B rationale @r32 (FT) | 100.0% | 92.6% | 78.2% | **90.4%** |
| Qwen2.5-14B ICL 0-shot | 100.0% | 89.9% | 75.3% | 88.1% |
| Qwen2.5-7B QA1 (FT) | 99.2% | 88.7% | 74.1% | 86.9% |
| Qwen2.5-7B ICL 0-shot | 100.0% | 87.7% | 64.7% | 84.2% |
| Qwen2.5-7B ICL 2-shot | 100.0% | 86.0% | 67.1% | 83.8% |
| Qwen2.5-7B per-event (FT) | 100.0% | 85.5% | 67.6% | 83.6% |
| Phi-3.5-mini QA1 (FT) | 99.2% | 85.0% | 68.8% | 83.5% |
| Qwen2.5-7B P prompt (FT) | 98.3% | 62.7% | 28.8% | 60.6% |

**Note:** P prompt collapses at 60.6% — BEFORE/NOT_BEFORE are multi-token outputs, making the first-token logprob unreliable.

## Re-run all models

```bash
cd /mnt/localssd/causal-embedding-research/llm_trc_baseline
source .venv/bin/activate

# Run all 11 models (finetuned + ICL)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash /mnt/localssd/causal-embedding-research/project_final/code/run_all_workflow_evals.sh
```

## Re-run a single model

```bash
cd /mnt/localssd/causal-embedding-research/llm_trc_baseline
source .venv/bin/activate

# Finetuned model (adapter path required)
CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/localssd/causal-embedding-research/project_final/code/eval_workflows_llm.py \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapter outputs/lora/Qwen2.5-14B-Instruct__QA1__lora_adapter \
    --prompt-type QA1 --tp 4 \
    --out-dir /mnt/localssd/causal-embedding-research/project_final/eval_results/llm_finetuned

# ICL model (no adapter)
CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/localssd/causal-embedding-research/project_final/code/eval_workflows_llm.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --prompt-type QA1 --shots 0 --tp 4 \
    --out-dir /mnt/localssd/causal-embedding-research/project_final/eval_results/llm_finetuned
```
