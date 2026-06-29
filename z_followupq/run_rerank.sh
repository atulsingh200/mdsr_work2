#!/bin/bash
set -e

cd /mnt/localssd/z_followupq

python3 rerank_followup.py \
    --input  prod_jan_feb_26-aep-ajo_follow-up-queries.json \
    --model  /mnt/localssd/automation/internship-causal-embedding/runs/classifier/best_mlp_bge_small/best.pt \
    --config /mnt/localssd/automation/internship-causal-embedding/runs/classifier/best_mlp_bge_small/config.json \
    --output reranked_best_mlp_bge_small.json \
    --batch-size 64 \
    --device cuda
