# eval_results/claude/

Claude Sonnet 4.6 pair classification results on `directional_test.jsonl` (4,801 pairs).

## Results

| Model | n | Acc | F1 | AUC |
|-------|:-:|:---:|:--:|:---:|
| Claude Sonnet 4.6 (2-shot ICL) | 4,801 | **82.9%** | **83.3%** | N/A |

**AUC not computable** — Claude outputs hard 0/1 text predictions with no probability scores.

## Prompt

2-shot ICL with `QA1` format ("Does Step A happen before Step B?" → YES/NO):

```
Example 1:
Step A: Open the rule editor and create a new rule for the page load event.
Step B: Add the Data Variable data element to the Data field of the rule action.
Answer: 1

Example 2:
Step A: Click Save to store the configuration settings.
Step B: In the left navigation panel, select Extensions.
Answer: 0

Now answer:
Step A: <text_1>
Step B: <text_2>
Answer:
```

## Re-run

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

python code/eval_claude_sonnet.py
```

**Takes ~60-90 min** at ~4 req/s using Azure Anthropic API.
The script resumes from checkpoint if interrupted — checkpoint saved at:
`/mnt/localssd/causal-embedding-research/final_data/claude_sonnet46_checkpoint.jsonl`
