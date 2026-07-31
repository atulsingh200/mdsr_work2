# data/

All datasets used for training and evaluation.

## Files

| File | Rows | Description |
|------|-----:|-------------|
| `directional_train.jsonl` | 38,731 | Training pairs |
| `directional_val.jsonl` | 3,954 | Validation pairs |
| `directional_test.jsonl` | 4,801 | Test pairs |
| `aep_workflows_eval.json` | 697 | Workflow ordering eval set (2/3/4-step) |

## Pair classification format (`directional_*.jsonl`)

Each line is a JSON object:
```json
{
  "text_1":    "Step A text",
  "text_2":    "Step B text",
  "label":     1,
  "source":    "procedural",
  "step_1":    "t2",
  "step_2":    "t3",
  "reasoning": "Step A must happen before Step B because ... VERDICT: BEFORE"
}
```

- `label=1` → Step A causally precedes Step B (A before B)
- `label=0` → Step A does NOT precede Step B
- `source` ∈ {procedural, aep, ajo}
- `reasoning` field is used by the reasoning decoder during training

## Workflow ordering format (`aep_workflows_eval.json`)

Each entry is a workflow with steps already in the correct order:
```json
{
  "id": 1,
  "title": "Audience to email",
  "source": "Marketing Workflows / Cart abandonment journey",
  "num_steps": 3,
  "t1": "First step text",
  "t2": "Second step text",
  "t3": "Third step text"
}
```

During evaluation, steps are scored pairwise and ranked by net score. The model must recover `t1 → t2 → t3 → ...` order.
