# Causal Step-Ordering for Adobe Experience Platform Workflows

## Overview

This project studies **causal/temporal ordering of procedural workflow steps** in Adobe Experience Platform (AEP) and Adobe Journey Optimizer (AJO) documentation. Given two workflow steps A and B, we predict whether A causally precedes B (`label=1`) or not (`label=0`). We also evaluate the ability of models to recover the full correct order of multi-step workflows.

Two model families are compared:
- **CrossEncoder2x (DeBERTa-v3-large)** — fine-tuned binary classifiers, with and without an auxiliary reasoning decoder
- **LLMs (Qwen2.5, Phi-3.5)** — fine-tuned via LoRA or used in-context (ICL)

---

## Repository Structure

```
project_final/
├── data/                              # All datasets
│   ├── directional_train.jsonl        # 38,731 training pairs (text_1, text_2, label)
│   ├── directional_val.jsonl          # 3,954 validation pairs
│   ├── directional_test.jsonl         # 4,801 test pairs
│   └── aep_workflows_eval.json        # 697 workflows for ordering eval (2/3/4-step)
│
├── training/                          # All training scripts
│   ├── train_encoder.py               # Train CrossEncoder2x (cls only)
│   ├── train_encoder_reasoning.py     # Train CrossEncoder2x + reasoning decoder
│   ├── lora_finetune.py              # LoRA fine-tune LLMs (QA1/P/rationale)
│   └── llm_prompts.py                # Prompt templates used during training
│
├── code/                              # All evaluation scripts
│   ├── eval_encoder_workflows.py      # Encoder model ordering eval on 697 workflows
│   ├── eval_workflows_llm.py          # LLM logit-based ordering eval on 697 workflows
│   ├── eval_lora_vllm.py             # LLM pair classification eval (vLLM, merged LoRA)
│   ├── eval_claude_sonnet.py         # Claude Sonnet 4.6 pair classification eval
│   ├── run_all_workflow_evals.sh      # Shell script: runs all LLM workflow ordering evals
│   └── llm_prompts.py                # Prompt templates (P / QA1 / QA2 / CoT)
│
├── eval_results/                      # All saved evaluation outputs
│   ├── encoder/                       # Encoder model results on pair classification
│   ├── llm_finetuned/                 # LLM workflow ordering results (per model JSON)
│   └── claude/                        # Claude Sonnet 4.6 results
│
├── model_checkpoints/                 # (Symlinks / notes on where checkpoints live)
│
├── run.sh                            # Master evaluation launcher
└── README.md                         # This file
```

---

## Datasets

### Pair Classification Dataset
| Split | File | Samples | Label dist |
|-------|------|--------:|:----------:|
| Train | `directional_train.jsonl` | 38,731 | ~50/50 |
| Val   | `directional_val.jsonl`   |  3,954 | ~50/50 |
| Test  | `directional_test.jsonl`  |  4,801 | 48.4% neg / 51.6% pos |

Each row: `{text_1, text_2, label, source, step_1, step_2, reasoning}`. `source` ∈ {procedural, aep, ajo}.

### Workflow Ordering Dataset
| File | Workflows | Step counts |
|------|----------:|:----------:|
| `aep_workflows_eval.json` | 697 | 120×2, 407×3, 170×4 |

The reasoning encoder model wins by **+7.0%** over the baseline on this eval set.

---

## How to Train

All models use the same training data in `data/`.

### CrossEncoder2x — Baseline (cls only)
```bash
CUDA_VISIBLE_DEVICES=0 bash training/train_baseline.sh
```
Exact config: deberta-v3-large, epochs=5, batch=16, grad_accum=4, lr=1e-5, seed=42.
Reproduces `v2_res_baseline` → val_acc=0.8225, val_auc=0.9027.

### CrossEncoder2x — Reasoning model (cls + decoder)
```bash
CUDA_VISIBLE_DEVICES=0 bash training/train_reasoning.sh
```
Exact config: deberta-v3-large, epochs=10, cls_w=1.0, dec_w=0.2, decoder_layers=4, lr=1e-5, seed=42.
Reproduces `v2_fixed_alpha02_ep10` → val_acc=0.8331, val_auc=0.9142.

### LLM LoRA fine-tuning
```bash
# QA1 prompt (Qwen2.5-7B)
CUDA_VISIBLE_DEVICES=0 python training/lora_finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 \
    --train data/directional_train.jsonl --val data/directional_val.jsonl \
    --out-dir outputs/lora --epochs 2 --lr 1e-4 --lora-r 32

# Rationale / LLMERE style
CUDA_VISIBLE_DEVICES=0 python training/lora_finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --rationale \
    --train data/directional_train.jsonl --val data/directional_val.jsonl \
    --out-dir outputs/llmere --epochs 2 --lr 2e-4 --lora-r 32
```

---

## How to Run

```bash
# Run all evaluations (groups 1–3)
bash run.sh --gpus "0,1,2,3" --tp 4

# Run only encoder model eval (Group 1)
bash run.sh --group 1

# Run only LLM fine-tuned workflow ordering (Group 2)
bash run.sh --group 2 --gpus "0,1,2,3" --tp 4

# Run only Claude Sonnet 4.6 pair classification (Group 3)
bash run.sh --group 3
```

**Requirements:**
- Encoder venv: `.../automation/internship-causal-embedding/.venv` (transformers, torch, sklearn)
- LLM venv: `.../llm_trc_baseline/.venv` (vllm==0.6.6, transformers==4.47.1, peft)
- GPU: 4× A100-80GB recommended (tp=4); single GPU works for smaller models
- HF cache: `/mnt/localssd/.cache/huggingface` (Qwen2.5-7B, 14B, Mistral-7B pre-downloaded)
- Azure Anthropic API key required for Group 4 (Claude)

---

## Results

### 1. Pair Classification — CrossEncoder2x (DeBERTa-v3-large)

Test set: `directional_test.jsonl` (4,801 pairs).

| Model | Type | Acc | F1 | AUC |
|-------|------|:---:|:--:|:---:|
| Reasoning model (v2_fixed_α02_ep10) | Finetuned + Reasoning | **86.1%** | **86.7%** | **91.5%** |
| Claude Sonnet 4.6 (2-shot ICL) | Similarity | 82.9% | 83.3% | — † |
| Baseline (v2_res_baseline) | Finetuned | 78.3% | 78.7% | 88.1% |

† AUC not computable — Claude outputs hard 0/1 labels, no probability scores.

**Claude Sonnet 4.6 (2-shot ICL) — full test set (4,801 samples):**

| Metric | Value |
|--------|------:|
| n | 4,801 |
| Acc | **82.9%** |
| F1 | **83.3%** |
| AUC | N/A † |

† Hard 0/1 text predictions — no probability scores available for AUC computation.

---

### 2. Pair Classification — LLM Fine-tuned (LoRA)

Test set: `directional_test.jsonl` (4,801 pairs).

| Model | Prompt | Acc | F1 |
|-------|--------|:---:|:--:|
| Qwen2.5-14B-Instruct QA1 | QA1 | **0.8569** | **0.8594** |
| Qwen2.5-7B-Instruct QA1 | QA1 | 0.8384 | 0.8418 |
| Qwen2.5-7B-Instruct P | P | 0.8280 | 0.8308 |
| Phi-3.5-mini-instruct QA1 | QA1 | 0.7638 | 0.7651 |
| Qwen2.5-7B rationale @2e-4 | rationale | 0.7682 | 0.7679 |
| Qwen2.5-7B rationale @1e-4 | rationale | 0.7703 | 0.7716 |
| Qwen2.5-7B rationale @r32 | rationale | 0.7617 | 0.7615 |

**With prompt variation (ICL k=2, procedural source, acc ~85%):**

| Model | Prompt | Shots | Acc |
|-------|--------|:-----:|:---:|
| Qwen2.5-14B-Instruct | QA1 | 2 | ~0.85 |
| Qwen2.5-7B-Instruct | QA1 | 2 | ~0.85 |
| Qwen2.5-14B-Instruct | P | 2 | ~0.85 |

---

### 3. Workflow Ordering — CrossEncoder2x on 697 Workflows

Method: net-score ranking of all directed pairs → exact ordering accuracy.

| Model | Type | 2-step | 3-step | 4-step | **TOTAL** |
|-------|------|:------:|:------:|:------:|:---------:|
| v2_res_baseline | Finetuned (cls only) | 100.0% | 83.8% | 67.1% | 575/697 **(82.5%)** |
| v2_fixed_α02_ep10 | Finetuned + Reasoning | 100.0% | 91.2% | 78.2% | 624/697 **(89.5%)** |
| **Δ (reasoning − baseline)** | | 0% | **+7.4%** | **+11.2%** | **+7.0%** |

---

### 4. Workflow Ordering — LLM Models on 697 Workflows

Method: P(YES) from YES-token logprob (single token, confirmed for all models) used as directional score.

| Model | Type | 2-step | 3-step | 4-step | **TOTAL** |
|-------|------|:------:|:------:|:------:|:---------:|
| Qwen2.5-14B QA1 (FT) | Finetuned | 100.0% | 93.4% | 86.5% | **92.8%** |
| Qwen2.5-7B rationale @2e-4 (FT) | Finetuned | 100.0% | 94.3% | 79.4% | **91.7%** |
| Qwen2.5-7B rationale @1e-4 (FT) | Finetuned | 100.0% | 92.1% | 80.0% | **90.5%** |
| Qwen2.5-7B rationale @r32 (FT) | Finetuned | 100.0% | 92.6% | 78.2% | **90.4%** |
| Qwen2.5-14B ICL 0-shot | ICL | 100.0% | 89.9% | 75.3% | 88.1% |
| Qwen2.5-7B QA1 (FT) | Finetuned | 99.2% | 88.7% | 74.1% | 86.9% |
| Qwen2.5-7B ICL 0-shot | ICL | 100.0% | 87.7% | 64.7% | 84.2% |
| Qwen2.5-7B ICL 2-shot | ICL | 100.0% | 86.0% | 67.1% | 83.8% |
| Qwen2.5-7B per-event (FT) | Finetuned | 100.0% | 85.5% | 67.6% | 83.6% |
| Phi-3.5-mini QA1 (FT) | Finetuned | 99.2% | 85.0% | 68.8% | 83.5% |
| Qwen2.5-7B P (FT) | Finetuned | 98.3% | 62.7% | 28.8% | 60.6% |

**Note on P prompt:** BEFORE/NOT_BEFORE are multi-token outputs; the first-token logit is unreliable for scoring, causing the 60.6% collapse.

---

## Key Findings

1. **Reasoning decoder helps on both classification and ordering:** The v2_fixed_α02 reasoning model achieves 86.1% acc vs 78.3% for the baseline on pair classification, and +7.0% on 697-workflow ordering (89.5% vs 82.5%).

2. **Finetuned LLMs beat reasoning encoder on ordering:** Qwen2.5-14B QA1 FT (92.8%) and Qwen2.5-7B rationale (91.7%) both outperform the CrossEncoder2x reasoning model (89.5%) on workflow ordering.

3. **Claude Sonnet 4.6 is strong ICL:** 82.9% acc (2-shot, 4,801 pairs) — between the baseline (78.3%) and reasoning model (86.1%) on the modified test set, with no training data at all.

4. **4-step is the hardest:** All models degrade significantly on 4-step workflows (encoder: 67–78%, LLMs: 29–87%).

5. **Prompt type matters critically:** QA1 (YES/NO single token) outperforms P (BEFORE/NOT_BEFORE multi-token) for workflow ordering because the YES/NO token logprob is a reliable directional score.

---

## Model Checkpoints

All CrossEncoder2x checkpoints are at:
```
.../automation/internship-causal-embedding/runs/
  crossencoder2x_deberta/         # baseline cls-only models
  crossencoder2x_deberta_reasoning/  # models with reasoning decoder
```

All LLM LoRA adapters are at:
```
.../llm_trc_baseline/outputs/
  lora/          # QA1 / P fine-tuned adapters
  llmere/        # rationale + per-event adapters (@2e-4)
  llmere_lr1e4/  # rationale adapter (@1e-4)
  llmere_r32/    # rationale adapter (rank 32)
```
