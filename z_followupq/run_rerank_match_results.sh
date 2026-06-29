#!/bin/bash
set -e

cd /mnt/localssd/z_followupq

python3 rerank_match_results.py \
    --input  match_results.jsonl \
    --model  /mnt/localssd/automation/internship-causal-embedding/runs/classifier/models/aep_causal_classification_34_all-mpnet-base-v2/best.pt \
    --config /mnt/localssd/automation/internship-causal-embedding/runs/classifier/models/aep_causal_classification_34_all-mpnet-base-v2/config.json \
    --output reranked_219M_MODEL.jsonl \
    --batch-size 64 \
    --device cuda
