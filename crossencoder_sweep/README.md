# Cross-Encoder Baseline Sweep

Standard cross-encoder (`CrossEncoderClassifier`: single transformer, joint
`[CLS] text_1 [SEP] text_2 [SEP]` → CLS → Linear head) trained across 7 datasets
× 6 models, evaluated on 2 OOD test sets (Milan multi-step ordering + AJO 3-step
ordering). Mirror of the bi-encoder sweep in `/mnt/localssd/baseline_sweep/`.

## Layout
```
crossencoder_sweep/
├── runs/{dataset}/{model}/
│   ├── best.pt / final.pt        # trained cross-encoder
│   ├── config.json               # backbone + hyperparams (model_type=cross_encoder)
│   ├── train.log / stdout.log
│   ├── test_metrics.json / test_predictions.jsonl
│   ├── run_summary.json          # compact in-domain metrics (for aggregation)
│   ├── ood_results.json          # Milan + AJO OOD eval
│   └── ood_stdout.log
├── results/final_results.json    # aggregated across all 42 runs
└── logs/sweep.log
```

## Scripts
- Train: `src/classifier/crossencoder/train_ce_baselines.py`
- OOD eval: `src/classifier/crossencoder/eval_ood_ce.py`
- Aggregate: `/mnt/localssd/aggregate_results.py` (shared with the bi-encoder sweep)

## Datasets (7) / Models (6, ≤600M)
Same as the bi-encoder sweep:
`aep_causal_cls34, aep_dataset, aep_causal_wf_v3, ajo_doc_not_tier1, ajo_newstyle,
new_aep_wf_scrap, new_ajo_workflows` ×
`deberta-v3-large, all-mpnet-base-v2, all-MiniLM-L6-v2, e5-large-v2,
bge-base-en-v1.5, bge-large-en-v1.5`.

## Run (launch EXACTLY ONCE — idempotent on re-run)
```bash
nohup bash /mnt/localssd/run_ce_baselines.sh \
    > /mnt/localssd/crossencoder_sweep/logs/nohup.log 2>&1 &
```
Monitor: `tail -f /mnt/localssd/crossencoder_sweep/logs/sweep.log` and `nvidia-smi`.

`ood_results.json` uses the same schema as the bi-encoder sweep, so the two
`final_results.json` files are directly comparable.
