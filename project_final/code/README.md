# code/

Evaluation scripts. All scripts evaluate on the datasets in `../data/`.

## Requirements

```bash
# For encoder evals (Groups 1)
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

# For LLM evals (Group 2)
source /mnt/localssd/causal-embedding-research/llm_trc_baseline/.venv/bin/activate
# vllm==0.6.6, transformers==4.47.1, peft required

# For Claude eval (Group 3)
# Uses Azure Anthropic API — needs API key in eval_claude_sonnet.py
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate
```

## Files

| File | Description |
|------|-------------|
| `eval_encoder_workflows.py` | Encoder models on 697-workflow ordering |
| `eval_workflows_llm.py` | LLM models on 697-workflow ordering (logit YES/NO) |
| `eval_lora_vllm.py` | LLM pair classification on test set (vLLM, merged LoRA) |
| `eval_claude_sonnet.py` | Claude Sonnet 4.6 pair classification (2-shot ICL) |
| `run_all_workflow_evals.sh` | Batch runner for all LLM workflow ordering evals |
| `llm_prompts.py` | Prompt templates shared across eval scripts |

---

## Group 1 — Encoder models on 697-workflow ordering

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python code/eval_encoder_workflows.py
```

Evaluates both `v2_res_baseline` and `v2_fixed_alpha02_ep10` on `data/aep_workflows_eval.json`.

**Method:** Score all directed pairs (i→j) using the YES-token logit, rank steps by net score, check exact ordering.

**Expected output:**
```
v2_res_baseline     575/697 (82.5%)  [2-step:100%  3-step:83.8%  4-step:67.1%]
v2_fixed_α02_ep10   624/697 (89.5%)  [2-step:100%  3-step:91.2%  4-step:78.2%]
Δ (reasoning)       +7.0%
```

---

## Group 2 — LLM fine-tuned models on 697-workflow ordering

### Run all models at once:
```bash
cd /mnt/localssd/causal-embedding-research/llm_trc_baseline
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 bash /mnt/localssd/causal-embedding-research/project_final/code/run_all_workflow_evals.sh
```

### Run a single model:
```bash
cd /mnt/localssd/causal-embedding-research/llm_trc_baseline
source .venv/bin/activate

# Finetuned LoRA model (merges adapter then scores)
CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/localssd/causal-embedding-research/project_final/code/eval_workflows_llm.py \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapter outputs/lora/Qwen2.5-14B-Instruct__QA1__lora_adapter \
    --prompt-type QA1 --tp 4

# ICL (no adapter, 0-shot)
CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/localssd/causal-embedding-research/project_final/code/eval_workflows_llm.py \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --prompt-type QA1 --shots 0 --tp 4
```

**Method:** P(YES) from YES-token logprob used as directional score (YES/NO confirmed single tokens for all models).

**Expected results (697 workflows):**

| Model | Total |
|-------|------:|
| Qwen2.5-14B QA1 (FT) | 92.8% |
| Qwen2.5-7B rationale @2e-4 (FT) | 91.7% |
| Qwen2.5-14B ICL 0-shot | 88.1% |
| Qwen2.5-7B QA1 (FT) | 86.9% |

### Run LLM pair classification eval:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/eval_lora_vllm.py \
    --adapter /mnt/localssd/causal-embedding-research/llm_trc_baseline/outputs/lora/Qwen2.5-14B-Instruct__QA1__lora_adapter \
    --model Qwen/Qwen2.5-14B-Instruct \
    --prompt-type QA1 --tp 4
```

---

## Group 3 — Claude Sonnet 4.6 pair classification

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

python code/eval_claude_sonnet.py
```

Runs Claude Sonnet 4.6 with 2-shot ICL on `data/directional_test.jsonl` (4,801 pairs).

**API:** Azure Anthropic endpoint (key hardcoded in script). Takes ~60-90 min at ~4 req/s.

**Expected results:** acc=82.9%, F1=83.3% (n=4,801)

**Note:** The script saves a checkpoint every row to `final_data/claude_sonnet46_checkpoint.jsonl` so it can resume if interrupted.
