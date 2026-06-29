#!/usr/bin/env bash
# run_ce2x.sh — train + evaluate CrossEncoder-2x (listwise softmax, uncapped data)
#
# Changes vs prior run:
#   * Loss:        listwise softmax (CE over 1 pos + 4 hardneg)  — was BCE
#   * Data:        finetune_eval/results (workflow=20K, multiwoz/qrecc=15K)  — same as BiEncoder baseline
#   * Early stop:  patience=3, threshold=0.002 on val_P@1  — was patience=2 / threshold=0.001 on AUC
#   * grad_accum:  removed (was halving optimizer steps per epoch)
#   * batch_size:  32 anchors (=160 GPU rows with group=5) — 24-layer model needs ~4x less than BERT-12L
#
# Usage:
#   bash run_ce2x.sh          # runs all 5 datasets sequentially (one GPU each)
#   bash run_ce2x.sh train    # training only
#   bash run_ce2x.sh eval     # evaluation only (requires checkpoints to exist)

set -euo pipefail

PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
CE2X=/mnt/localssd/new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x
DATA=/mnt/localssd/new_base_line/internship-causal-embedding/finetune_eval/results

MODE="${1:-all}"   # all | train | eval

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  echo "========================================================"
  echo " TRAINING  (listwise softmax, patience=3, thresh=0.002)"
  echo "========================================================"

  # batch_size = anchors per step; GPU rows = batch_size * 5 (group)
  # 24-layer BERT with grad needs ~160 rows max at seq_len=256 on 80GB A100
  # → batch_size=32 (32*5=160 rows, seq_len=256)
  # workflow uses seq_len=128 → batch_size=64 (64*5=320 rows, seq_len=128)

  # aep_causal — 5.5K pairs, mined val split available
  CUDA_VISIBLE_DEVICES=0 $PY $CE2X/train_ce2x.py \
      --dataset aep_causal \
      --data-dir $DATA \
      --epochs 15 \
      --batch-size 32 --val-batch-size 128 \
      --max-length 256 --num-workers 2 \
      --n-hard 2 --n-random 2 \
      --early-stop-patience 3 --early-stop-threshold 0.002 &

  # followupqg — 2.8K pairs, 5%-holdout val
  CUDA_VISIBLE_DEVICES=1 $PY $CE2X/train_ce2x.py \
      --dataset followupqg \
      --data-dir $DATA \
      --epochs 15 \
      --batch-size 32 --val-batch-size 128 \
      --max-length 256 --num-workers 2 \
      --n-hard 2 --n-random 2 \
      --early-stop-patience 3 --early-stop-threshold 0.002 &

  # multiwoz_v24 — 15K pairs, 5%-holdout val
  CUDA_VISIBLE_DEVICES=2 $PY $CE2X/train_ce2x.py \
      --dataset multiwoz_v24 \
      --data-dir $DATA \
      --epochs 10 \
      --batch-size 32 --val-batch-size 128 \
      --max-length 256 --num-workers 2 \
      --n-hard 2 --n-random 2 \
      --early-stop-patience 3 --early-stop-threshold 0.002 &

  # qrecc — 15K pairs, 5%-holdout val
  CUDA_VISIBLE_DEVICES=3 $PY $CE2X/train_ce2x.py \
      --dataset qrecc \
      --data-dir $DATA \
      --epochs 10 \
      --batch-size 32 --val-batch-size 128 \
      --max-length 256 --num-workers 2 \
      --n-hard 2 --n-random 2 \
      --early-stop-patience 3 --early-stop-threshold 0.002 &

  wait
  echo "[train] small/medium datasets done"

  # workflow — 20K pairs, seq_len=128 halves memory → can use larger anchor batch
  CUDA_VISIBLE_DEVICES=0 $PY $CE2X/train_ce2x.py \
      --dataset workflow \
      --data-dir $DATA \
      --epochs 5 \
      --batch-size 64 --val-batch-size 256 \
      --max-length 128 --num-workers 2 \
      --early-stop-patience 2 --early-stop-threshold 0.003

  echo "[train] workflow done"
fi

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
if [[ "$MODE" == "all" || "$MODE" == "eval" ]]; then
  echo "========================================================"
  echo " EVALUATION"
  echo "========================================================"

  CUDA_VISIBLE_DEVICES=0 $PY $CE2X/evaluate_ce2x.py \
      --dataset aep_causal \
      --data-dir $DATA \
      --pool-size 1000 --max-queries 2000 \
      --batch-size 64 --max-length 256 &

  CUDA_VISIBLE_DEVICES=1 $PY $CE2X/evaluate_ce2x.py \
      --dataset followupqg \
      --data-dir $DATA \
      --pool-size 1000 --max-queries 2000 \
      --batch-size 64 --max-length 256 &

  CUDA_VISIBLE_DEVICES=2 $PY $CE2X/evaluate_ce2x.py \
      --dataset multiwoz_v24 \
      --data-dir $DATA \
      --pool-size 1000 --max-queries 2000 \
      --batch-size 64 --max-length 256 &

  CUDA_VISIBLE_DEVICES=3 $PY $CE2X/evaluate_ce2x.py \
      --dataset qrecc \
      --data-dir $DATA \
      --pool-size 1000 --max-queries 2000 \
      --batch-size 64 --max-length 256 &

  wait

  CUDA_VISIBLE_DEVICES=0 $PY $CE2X/evaluate_ce2x.py \
      --dataset workflow \
      --data-dir $DATA \
      --pool-size 1000 --max-queries 2000 \
      --batch-size 64 --max-length 128

  echo "========================================================"
  echo " RESULTS SUMMARY"
  echo "========================================================"
  for ds in aep_causal followupqg multiwoz_v24 qrecc workflow; do
    echo "--- $ds ---"
    $PY -c "
import json, pathlib
base = pathlib.Path('$CE2X/results/$ds')
e  = json.loads((base / 'eval.json').read_text())
eh = json.loads((base / 'eval_hardneg.json').read_text())
print(f\"  AUC(rand)={e['auc']:.4f}  P@1(rand)={e['precision_at_1']:.4f}  AUC(HN)={eh['auc']:.4f}  P@1(HN)={eh['precision_at_1']:.4f}\")
if 'mrr' in e:
    rk = e.get('recall_at_k', {})
    print(f\"  MRR={e['mrr']:.4f}  R@1={rk.get('1',0):.3f}  R@3={rk.get('3',0):.3f}  R@5={rk.get('5',0):.3f}  R@10={rk.get('10',0):.3f}  Median={e['median_rank']:.1f}\")
" 2>/dev/null || echo "  (eval.json not found — did training finish?)"
  done
fi

echo "[done]"
