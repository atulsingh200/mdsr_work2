# training/

Scripts to train the two main models from scratch.

## Requirements

```bash
# Activate the encoder venv
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

# GPU: single A100-80GB is enough for both models
# HF cache: microsoft/deberta-v3-large will be downloaded automatically
```

## Files

| File | Description |
|------|-------------|
| `train_baseline.sh` | **Run this** to train the baseline encoder |
| `train_reasoning.sh` | **Run this** to train the reasoning encoder |
| `train_encoder.py` | Baseline training script (CrossEncoder2x, cls head only) |
| `train_encoder_reasoning.py` | Reasoning training script (CrossEncoder2x + decoder) |
| `model_ce2x.py` | Model definitions: `CrossEncoder2x`, `CrossEncoder2xWithReasoning` |
| `lora_finetune.py` | LoRA fine-tuning script for LLMs |
| `llm_prompts.py` | Prompt templates (P / QA1 / rationale) used during LLM training |

## 1. Train the Baseline Encoder

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 bash training/train_baseline.sh
```

**What it does:** Trains `CrossEncoder2x` (DeBERTa-v3-large, 24 layers) on binary pair classification using BCE loss only.

**Exact config:**
| Parameter | Value |
|-----------|-------|
| backbone | microsoft/deberta-v3-large |
| epochs | 5 |
| batch_size | 16 |
| grad_accum | 4 (effective batch = 64) |
| lr | 1e-5 |
| weight_decay | 0.01 |
| warmup_frac | 0.1 |
| dropout | 0.1 |
| seed | 42 |

**Expected results:** val_acc=0.8225, val_auc=0.9027 (best checkpoint at epoch 5)

**Output:** `runs/crossencoder2x_deberta/v2_res_baseline/`

---

## 2. Train the Reasoning Encoder

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 bash training/train_reasoning.sh
```

**What it does:** Trains `CrossEncoder2xWithReasoning` — same encoder as baseline plus a 4-layer Transformer decoder that reconstructs the reasoning trace. Loss = `1.0 × BCE + 0.2 × CrossEntropy(decoder)`.

**Exact config:**
| Parameter | Value |
|-----------|-------|
| backbone | microsoft/deberta-v3-large |
| decoder_layers | 4 |
| cls_loss_weight | 1.0 |
| decoder_loss_weight | 0.2 |
| epochs | 10 |
| batch_size | 16 |
| grad_accum | 4 (effective batch = 64) |
| lr | 1e-5 |
| weight_decay | 0.01 |
| warmup_frac | 0.1 |
| dropout | 0.1 |
| seed | 42 |

**Expected results:** val_acc=0.8331, val_auc=0.9142 (best checkpoint at epoch 3)

**Output:** `runs/crossencoder2x_deberta_reasoning/v2_fixed_alpha02_ep10/`

**Note:** The training data must have a `reasoning` field (the `directional_train.jsonl` already has it).

---

## 3. Fine-tune LLMs with LoRA

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

# Qwen2.5-7B, QA1 prompt
CUDA_VISIBLE_DEVICES=0 python training/lora_finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompt-type QA1 \
    --train data/directional_train.jsonl \
    --val   data/directional_val.jsonl \
    --test  data/directional_test.jsonl \
    --out-dir runs/lora \
    --epochs 2 --lr 1e-4 --lora-r 32

# Qwen2.5-14B, QA1 prompt (best LLM result)
CUDA_VISIBLE_DEVICES=0,1 python training/lora_finetune.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --prompt-type QA1 \
    --train data/directional_train.jsonl \
    --val   data/directional_val.jsonl \
    --test  data/directional_test.jsonl \
    --out-dir runs/lora \
    --epochs 2 --lr 1e-4 --lora-r 32

# Qwen2.5-7B with rationale (LLMERE style)
CUDA_VISIBLE_DEVICES=0 python training/lora_finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompt-type QA1 --rationale \
    --train data/directional_train.jsonl \
    --val   data/directional_val.jsonl \
    --test  data/directional_test.jsonl \
    --out-dir runs/llmere \
    --epochs 2 --lr 2e-4 --lora-r 32
```

**Prompt types:**
- `QA1` — "Does Step A happen before Step B?" → YES/NO
- `P`   — "Does Step A happen BEFORE or NOT_BEFORE Step B?" → BEFORE/NOT_BEFORE
- `QA1 --rationale` — generate reasoning trace then predict (LLMERE style)

---

## Internal module structure

The training scripts rely on:
```
training/
├── model_ce2x.py              ← CrossEncoder2x, CrossEncoder2xWithReasoning
src/
└── classifier/
    ├── crossencoder/
    │   └── data.py            ← CEPairDataset, CECollate, CEReasoningCollate, etc.
    └── training/
        └── train.py           ← device_auto(), make_logger(), set_seed()
```

All imports are self-contained within `project_final/` — no external paths needed.
