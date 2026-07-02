#!/usr/bin/env bash
# run_pass2.sh
#
# PASS 2 of the collapse fix: retrain the two heavy ajo_doc_not_tier1 / deberta-v3-large
# runs (bi-encoder + cross-encoder, ~88k train rows, ~9h each) with the VALIDATED recipe
# (effective batch 64 via grad-accum, encoder LR 1e-5, warmup 0.06, deberta fp32/--no-amp;
# CE head LR 1e-5, BI head LR 1e-3). Unlike `rerun_broken.sh pass2` (which runs the two
# full launchers sequentially, ~18h), this runs BOTH in PARALLEL on GPU0+GPU1 (~9h), then
# does OOD eval and re-aggregates both sweeps.
#
# RUN ONLY AFTER pass1 has finished (needs GPU0 + GPU1 free). Usage:
#   nohup bash /mnt/localssd/run_pass2.sh > /mnt/localssd/rerun_pass2.log 2>&1 &
#   tail -f /mnt/localssd/rerun_pass2.log

set -euo pipefail

REPO=/mnt/localssd/automation/internship-causal-embedding
PY=$REPO/.venv/bin/python
DATA=$REPO/data/ajo_doc_dataset_not_tier1
MILAN=/mnt/localssd/test_samples_milan.json
AJO=/mnt/localssd/ajo_orchestrated_workflows_flat.json

BI_TRAIN=$REPO/src/classifier/training/train_baselines.py
BI_EVAL=$REPO/src/classifier/training/eval_ood.py
BI_RD=/mnt/localssd/baseline_sweep/runs/ajo_doc_not_tier1/deberta-v3-large
BI_AGG_ROOT=/mnt/localssd/baseline_sweep/runs
BI_AGG_OUT=/mnt/localssd/baseline_sweep/results/final_results.json

CE_TRAIN=$REPO/src/classifier/crossencoder/train_ce_baselines.py
CE_EVAL=$REPO/src/classifier/crossencoder/eval_ood_ce.py
CE_RD=/mnt/localssd/crossencoder_sweep/runs/ajo_doc_not_tier1/deberta-v3-large
CE_AGG_ROOT=/mnt/localssd/crossencoder_sweep/runs
CE_AGG_OUT=/mnt/localssd/crossencoder_sweep/results/final_results.json

AGG=/mnt/localssd/aggregate_results.py

# ---- back up the collapsed runs and clear them so they retrain cleanly ----
backup() {  # run_dir
  local rd=$1
  [[ -d $rd ]] || { echo "MISSING dir: $rd"; return; }
  echo "backing up collapsed run -> $rd/collapsed_bak/"
  mkdir -p "$rd/collapsed_bak"
  local f
  for f in best.pt final.pt run_summary.json ood_results.json test_metrics.json \
           test_predictions.jsonl config.json train.log stdout.log ood_stdout.log; do
    [[ -e "$rd/$f" ]] && mv -f "$rd/$f" "$rd/collapsed_bak/" || true
  done
}
backup "$BI_RD"
backup "$CE_RD"

# ---- train both in parallel (BI on GPU0, CE on GPU1) ----
echo "== training BI (GPU0) + CE (GPU1) in parallel — ~9h =="
CUDA_VISIBLE_DEVICES=0 "$PY" "$BI_TRAIN" \
    --data-dir "$DATA" --base-model microsoft/deberta-v3-large --out-dir "$BI_RD" \
    --dataset-slug ajo_doc_not_tier1 --model-slug deberta-v3-large --gpu-id 0 \
    --batch-size 8 --grad-accum 8 --eval-batch-size 64 --epochs 4 \
    --head-type mlp --head-hidden-dims 512,128 \
    --no-amp --lr-encoder 1e-5 --lr-head 1e-3 --warmup-frac 0.06 \
    > "$BI_RD/stdout.log" 2>&1 &
BI_PID=$!
echo "  BI pid=$BI_PID -> $BI_RD/train.log"

CUDA_VISIBLE_DEVICES=1 "$PY" "$CE_TRAIN" \
    --data-dir "$DATA" --base-model microsoft/deberta-v3-large --out-dir "$CE_RD" \
    --dataset-slug ajo_doc_not_tier1 --model-slug deberta-v3-large --gpu-id 1 \
    --batch-size 8 --grad-accum 8 --eval-batch-size 64 --epochs 4 \
    --no-amp --lr-encoder 1e-5 --lr-head 1e-5 --warmup-frac 0.06 \
    > "$CE_RD/stdout.log" 2>&1 &
CE_PID=$!
echo "  CE pid=$CE_PID -> $CE_RD/train.log"

# wait for both; report if either fails
FAIL=0
wait "$BI_PID" || { echo "BI training FAILED (see $BI_RD/stdout.log)"; FAIL=1; }
wait "$CE_PID" || { echo "CE training FAILED (see $CE_RD/stdout.log)"; FAIL=1; }
[[ $FAIL -eq 0 ]] || { echo "== a training run failed; NOT running OOD/aggregate =="; exit 1; }
echo "== both trainings done =="

# ---- OOD eval in parallel ----
echo "== OOD eval (BI GPU0 + CE GPU1) =="
CUDA_VISIBLE_DEVICES=0 "$PY" "$BI_EVAL" --run-dir "$BI_RD" --milan-json "$MILAN" --ajo-json "$AJO" \
    > "$BI_RD/ood_stdout.log" 2>&1 &
BI_OOD=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$CE_EVAL" --run-dir "$CE_RD" --milan-json "$MILAN" --ajo-json "$AJO" \
    > "$CE_RD/ood_stdout.log" 2>&1 &
CE_OOD=$!
wait "$BI_OOD" || echo "BI OOD eval failed (see $BI_RD/ood_stdout.log)"
wait "$CE_OOD" || echo "CE OOD eval failed (see $CE_RD/ood_stdout.log)"

# ---- re-aggregate both sweeps ----
echo "== re-aggregating both final_results.json =="
"$PY" "$AGG" --root "$BI_AGG_ROOT" --out "$BI_AGG_OUT"
"$PY" "$AGG" --root "$CE_AGG_ROOT" --out "$CE_AGG_OUT"

# ---- report ----
echo "== PASS 2 RESULTS =="
for rd in "$BI_RD" "$CE_RD"; do
  acc=$("$PY" -c "import json;print(round(json.load(open('$rd/run_summary.json'))['test_acc'],4))" 2>/dev/null || echo "??")
  echo "  $rd  test_acc=$acc  (collapsed was 0.4966)"
done
echo "== DONE =="
