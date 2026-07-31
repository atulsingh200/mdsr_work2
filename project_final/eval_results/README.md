# eval_results/

Pre-computed evaluation results for all models.

## Structure

```
eval_results/
├── encoder/          CrossEncoder2x pair classification on directional_test.jsonl
├── llm_finetuned/    LLM models on 697-workflow ordering (aep_workflows_eval.json)
└── claude/           Claude Sonnet 4.6 pair classification on directional_test.jsonl
```

---

## encoder/

Pair classification results on `directional_test.jsonl` (4,801 pairs).

| File | Model | Acc | AUC | F1 |
|------|-------|:---:|:---:|:--:|
| `deberta_ce2x_on_final_test.json` | v2_res_baseline (cls only) | 78.3% | 88.1% | 78.7% |
| `deberta_ce2x_reasoning_on_final_test.json` | v2_fixed_α02_ep10 (reasoning) | 86.1% | 91.5% | 86.7% |

---

## llm_finetuned/

Workflow ordering accuracy on `aep_workflows_eval.json` (697 workflows).
Each JSON has `order_acc`, `total_correct`, `total`, `by_n` breakdown.

| File | Model | Total Acc |
|------|-------|:---------:|
| `Qwen2.5-14B-Instruct__QA1___wf_order.json` | Qwen2.5-14B QA1 (FT) | 92.8% |
| `Qwen2.5-7B-Instruct__rationale_lr2e4__wf_order.json` | Qwen2.5-7B rationale @2e-4 (FT) | 91.7% |
| `Qwen2.5-7B-Instruct__rationale_lr1e4__wf_order.json` | Qwen2.5-7B rationale @1e-4 (FT) | 90.5% |
| `Qwen2.5-7B-Instruct__rationale_r32__wf_order.json` | Qwen2.5-7B rationale r32 (FT) | 90.4% |
| `Qwen2.5-14B-Instruct__ICL_k0_QA1__wf_order.json` | Qwen2.5-14B ICL 0-shot | 88.1% |
| `Qwen2.5-7B-Instruct__QA1___wf_order.json` | Qwen2.5-7B QA1 (FT) | 86.9% |
| `Qwen2.5-7B-Instruct__ICL_k0_QA1__wf_order.json` | Qwen2.5-7B ICL 0-shot | 84.2% |
| `Qwen2.5-7B-Instruct__ICL_k2_QA1__wf_order.json` | Qwen2.5-7B ICL 2-shot | 83.8% |
| `Qwen2.5-7B-Instruct__perevent__wf_order.json` | Qwen2.5-7B per-event (FT) | 83.6% |
| `Phi-3.5-mini-instruct__QA1___wf_order.json` | Phi-3.5-mini QA1 (FT) | 83.5% |
| `Qwen2.5-7B-Instruct__P___wf_order.json` | Qwen2.5-7B P prompt (FT) | 60.6% |

---

## claude/

| File | Model | n | Acc | F1 | AUC |
|------|-------|:-:|:---:|:--:|:---:|
| `claude_sonnet46_results.json` | Claude Sonnet 4.6 (2-shot ICL) | 4,801 | 82.9% | 83.3% | N/A |

AUC not computable — Claude outputs hard 0/1 predictions only.

---

## JSON format (llm_finetuned/)

```json
{
  "run_tag": "Qwen2.5-14B-Instruct__QA1___wf_order",
  "base_model": "Qwen/Qwen2.5-14B-Instruct",
  "prompt_type": "QA1",
  "shots": 0,
  "yes_token_id": 14004,
  "no_token_id": 8996,
  "order_acc": 0.928,
  "total_correct": 647,
  "total": 697,
  "by_n": {
    "2": {"correct": 120, "total": 120},
    "3": {"correct": 380, "total": 407},
    "4": {"correct": 147, "total": 170}
  }
}
```
