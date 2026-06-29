#!/usr/bin/env bash
# Pure cross-encoder eval (no CDEv2 / no BiEncoder): AUC + P@1 for CE#1, CE#2,
# and their ensemble, vs 4 random negatives and 4 MiniLM-L6-v2 semantic hard
# negatives on aep_causal. Run from the repo root.
set -euo pipefail

cd /mnt/localssd/internship-causal-embedding-merge-followup

# Pick a free GPU (default 0). Override:  GPU=1 bash my_work/kl/v2/run_eval_ce_ensemble.sh
GPU="${GPU:-0}"

# ---- TEST split (reported numbers) ----
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python my_work/kl/v2/eval_ce_ensemble.py --split test

# ---- VAL split (optional; uncomment to also run) ----
# CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python my_work/kl/v2/eval_ce_ensemble.py --split val
