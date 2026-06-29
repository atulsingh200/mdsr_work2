# Baseline Sweep

2-untied-encoder `DirectionalClassifier` trained across 7 datasets × 6 models,
evaluated on 2 OOD test sets (Milan multi-step ordering + AJO 3-step ordering).

## Layout

```
baseline_sweep/
├── runs/                       # one subdir per (dataset, model)
│   └── {dataset}/{model}/
│       ├── best.pt             # best-val-acc checkpoint (the trained model)
│       ├── final.pt            # last-epoch checkpoint
│       ├── config.json         # architecture + hyperparams
│       ├── train.log           # training log
│       ├── stdout.log          # full stdout/stderr from the training process
│       ├── test_metrics.json   # in-domain test metrics + per-tier-pair acc
│       ├── test_predictions.jsonl
│       ├── run_summary.json    # compact in-domain metrics (for aggregation)
│       ├── ood_results.json    # Milan + AJO OOD eval results
│       └── ood_stdout.log      # OOD eval process log
├── results/
│   └── final_results.json      # aggregated metrics across all runs
└── logs/
    └── sweep.log               # sweep-level orchestration log
```

## Datasets (7)
`aep_causal_cls34`, `aep_dataset`, `aep_causal_wf_v3`, `ajo_doc_not_tier1`,
`ajo_newstyle`, `new_aep_wf_scrap`, `new_ajo_workflows`

## Models (6, all ≤ 600M)
`deberta-v3-large`, `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `e5-large-v2`,
`bge-base-en-v1.5`, `bge-large-en-v1.5`

## Run

```bash
bash /mnt/localssd/run_baselines.sh
```

Distributes 42 training runs across 4 GPUs (1 job/GPU), then runs OOD eval in
parallel, then aggregates. Re-running skips completed work (idempotent).
```
