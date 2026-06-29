#!/usr/bin/env bash
# Run the full CDEv2+CE-rerank pipeline for ONE dataset on ONE GPU.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_ds.sh followupqg
set -euo pipefail

DS=$1
GPU=${CUDA_VISIBLE_DEVICES:-0}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KL="$ROOT/my_work/kl"
PY="$ROOT/.venv/bin/python"
BIENC="$ROOT/finetune_eval/results/$DS/checkpoint_best.pt"
LOG="$KL/results/${DS}_pipeline.log"

mkdir -p "$KL/results/$DS"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "  Dataset: $DS   GPU: $GPU   $(date)"
echo "========================================================"

# ── 0. stage data (copy from finetune_eval/results, create val if needed) ──
echo; echo "[step 0] stage data"
cd "$ROOT"

# copy pairs + K=4 hard negatives
for f in train_pairs.jsonl val_pairs.jsonl test_pairs.jsonl \
          hard_negatives.npy val_hard_negatives.npy test_hard_negatives.npy; do
  src="finetune_eval/results/$DS/$f"
  dst="my_work/kl/results/$DS/$f"
  [ -f "$src" ] && [ ! -f "$dst" ] && cp "$src" "$dst" && echo "  copied $f"
done

# create val split (10% of train) if not present
if [ ! -f "my_work/kl/results/$DS/val_pairs.jsonl" ]; then
  echo "  no val split — holding out 10% of train"
  $PY -c "
import random
from pathlib import Path
ds = '$DS'
p = Path('my_work/kl/results/' + ds + '/train_pairs.jsonl')
lines = p.read_text().splitlines()
rng = random.Random(42)
idx = list(range(len(lines))); rng.shuffle(idx)
n_val = max(1, int(len(lines) * 0.10))
val_idx = set(idx[:n_val])
Path('my_work/kl/results/' + ds + '/val_pairs.jsonl').write_text(
    '\n'.join(lines[i] for i in sorted(val_idx)) + '\n')
Path('my_work/kl/results/' + ds + '/train_pairs.jsonl').write_text(
    '\n'.join(lines[i] for i in range(len(lines)) if i not in val_idx) + '\n')
print(f'  val={len(val_idx)}  train={len(lines)-len(val_idx)}')
"
fi

# ── 1. mine K=8 hard negatives (CPU — MiniLM) ──
K8="my_work/kl/results/$DS/hard_negatives_k8.npy"
if [ ! -f "$K8" ]; then
  echo; echo "[step 1] mine K=8 hard negatives (train + test)"
  cd "$ROOT"
  $PY my_work/kl/v2/mine_hard_neg_v2.py --dataset "$DS" --split both --k 8
else
  echo; echo "[step 1] K=8 negs already exist — skipping"
fi

# ── 2. train CDEv2 retriever (run5c config, warm-started) ──
CDE_CKPT="$KL/results/$DS/cde_v2_run5c_best.pt"
if [ ! -f "$CDE_CKPT" ]; then
  echo; echo "[step 2] train CDEv2 retriever"
  cd "$KL"
  $PY v2/train_v2.py --dataset "$DS" \
    --init-from-biencoder "$BIENC" \
    --proj-dim 768 --pooling cls --mu-identity --log-sigma-init 3.0 \
    --kl-scale dim --init-cos-weight 1.0 \
    --backbone-lr 2e-5 --batch-size 64 --max-seq-length 256 \
    --phase-a-steps 3000 --phase-b-steps 0 --total-steps 3000 \
    --lambda-entropy 0 --lambda-antisym 0 --lambda-bpr 0 \
    --hard-neg-file "results/$DS/hard_negatives_k8.npy" \
    --out-suffix _run5c
else
  echo; echo "[step 2] CDEv2 checkpoint exists — skipping"
fi

# ── 3. train cross-encoder reranker ──
CE_CKPT="$KL/cross_encoder/results/${DS}_rerank/checkpoint_best.pt"
if [ ! -f "$CE_CKPT" ]; then
  echo; echo "[step 3] train cross-encoder reranker"
  cd "$ROOT"
  $PY my_work/kl/cross_encoder/train_ce.py --dataset "$DS" \
    --data-dir finetune_eval/results \
    --epochs 4 \
    --out-dir "my_work/kl/cross_encoder/results/${DS}_rerank"
else
  echo; echo "[step 3] CE checkpoint exists — skipping"
fi

# ── 4. rerank eval: tune on val, report on test ──
# workflow has 260k test pairs → use candidate_pool=1000 to avoid N×N OOM
CPOOL_ARG=""
if [ "$DS" = "workflow" ]; then
  CPOOL_ARG="--candidate-pool 1000"
fi

echo; echo "[step 4a] rerank eval on VAL (sweep pool/blend)"
cd "$KL"
$PY v2/rerank_eval.py --dataset "$DS" --split val \
  --cde-suffix _run5c \
  --ce-dir "cross_encoder/results/${DS}_rerank" \
  --pool 20 50 --blend 0.3 0.5 0.7 \
  --batch-size 64 --max-length 256 $CPOOL_ARG

echo; echo "[step 4b] rerank eval on TEST"
# read best val pool/blend from the val rerank_eval.json
BEST_TAG=$($PY -c "
import json
d = json.loads(open('results/$DS/rerank_eval.json').read())
print(d['best']['tag'])
" 2>/dev/null || echo "pool50_blend0.5")
POOL=$(echo $BEST_TAG | sed 's/pool\([0-9]*\)_blend.*/\1/')
BLEND=$(echo $BEST_TAG | sed 's/pool[0-9]*_blend\(.*\)/\1/')
echo "  using val-best: pool=$POOL blend=$BLEND"
$PY v2/rerank_eval.py --dataset "$DS" --split test \
  --cde-suffix _run5c \
  --ce-dir "cross_encoder/results/${DS}_rerank" \
  --pool "$POOL" --blend "$BLEND" \
  --batch-size 64 --max-length 256 $CPOOL_ARG

echo; echo "========================================================"
echo "  DONE $DS   $(date)"
echo "  Results: $KL/results/$DS/rerank_eval.json"
$PY -c "
import json
d = json.loads(open('results/$DS/rerank_eval.json').read())
print(f\"  stage-1 MRR : {d['stage1']['mrr']:.4f}\")
print(f\"  fused   MRR : {d['best']['mrr']:.4f}  ({d['best']['tag']})\")
"
echo "========================================================"
