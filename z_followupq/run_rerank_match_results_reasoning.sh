#!/bin/bash
set -e

cd /mnt/localssd/z_followupq

python3 rerank_match_results.py \
    --input  match_results.jsonl \
    --model  /mnt/localssd/automation/internship-causal-embedding/runs/reasoning/reasoning_bge_small/best.pt \
    --config /mnt/localssd/automation/internship-causal-embedding/runs/reasoning/reasoning_bge_small/config.json \
    --output reranked_reasoning.jsonl \
    --arch   reasoning \
    --batch-size 64 \
    --device cuda
