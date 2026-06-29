#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

python3 src/followup_data/builders/build_classification_data_semantic.py \
  --corpus data/comparison_test/aep_docs_collection_v_24-03-2026.json \
  --out-dir data/aep_causal_classification_semantic \
  --unit sentence \
  --min-sentences 2 \
  --max-sentences 5 \
  --tokenizer sentence-transformers/all-mpnet-base-v2 \
  --pair-scope all \
  --single-direction \
  --split-level chunk_tier \
  --max-chunk-pairs-per-tier-pair 1500 \
  --cap-splits train \
  --cap-seed 42 \
  --seed 42 \
  --val-frac 0.10 \
  --test-frac 0.10 \
  --semantic-test \
  --match-counts data/aep_causal_classification/directional_test.jsonl \
  --sim-model sentence-transformers/all-MiniLM-L6-v2
