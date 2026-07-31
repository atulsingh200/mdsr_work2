# eval_results/encoder/

Pair classification results on `directional_test.jsonl` (4,801 pairs).

## Results

| Model | Acc | AUC | F1 |
|-------|:---:|:---:|:--:|
| v2_res_baseline | 78.3% | 88.1% | 78.7% |
| v2_fixed_α02_ep10 (reasoning) | **86.1%** | **91.5%** | **86.7%** |

## Re-run

```bash
cd /mnt/localssd/causal-embedding-research/project_final
source /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/.venv/bin/activate

# Run pair classification eval on both encoder models
python code/eval_encoder_workflows.py
```

Results are saved to this folder automatically.
