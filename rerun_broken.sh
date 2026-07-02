#!/usr/bin/env bash
# rerun_broken.sh
#
# Retrain the collapsed large-model sweep runs (test_acc ~= 0.50) with the
# corrected recipe. The recipe itself lives in the two (already-edited) launchers
# run_baselines.sh / run_ce_baselines.sh (single 1e-5 LR, 0.06 warmup, effective
# batch 32 via grad-accum for the large models). This script only backs up and
# clears each collapsed run's idempotency markers so the idempotent launchers redo
# EXACTLY those runs, then re-run their OOD eval and re-aggregate both
# final_results.json.
#
# Run AFTER the launcher/train-script patches are in place. Usage:
#   bash        /mnt/localssd/rerun_broken.sh smoke
#   nohup bash  /mnt/localssd/rerun_broken.sh pass1 > /mnt/localssd/rerun_pass1.log 2>&1 &
#   nohup bash  /mnt/localssd/rerun_broken.sh pass2 > /mnt/localssd/rerun_pass2.log 2>&1 &

set -euo pipefail

BI_RUNS=/mnt/localssd/baseline_sweep/runs
CE_RUNS=/mnt/localssd/crossencoder_sweep/runs
BI_LAUNCHER=/mnt/localssd/run_baselines.sh
CE_LAUNCHER=/mnt/localssd/run_ce_baselines.sh

# Collapsed runs, encoded as "sweep|dataset|model"  (sweep = bi | ce).
PASS1=(
  "bi|ajo_newstyle|deberta-v3-large"      "bi|ajo_newstyle|e5-large-v2"
  "bi|new_aep_wf_scrap|deberta-v3-large"  "bi|new_ajo_workflows|deberta-v3-large"
  "bi|new_ajo_workflows|e5-large-v2"
  "ce|aep_causal_cls34|e5-large-v2"       "ce|aep_causal_wf_v3|e5-large-v2"
  "ce|aep_dataset|e5-large-v2"            "ce|ajo_doc_not_tier1|bge-large-en-v1.5"
  "ce|ajo_doc_not_tier1|e5-large-v2"      "ce|ajo_newstyle|bge-large-en-v1.5"
  "ce|ajo_newstyle|deberta-v3-large"      "ce|ajo_newstyle|e5-large-v2"
  "ce|new_aep_wf_scrap|bge-large-en-v1.5" "ce|new_aep_wf_scrap|deberta-v3-large"
  "ce|new_aep_wf_scrap|e5-large-v2"       "ce|new_ajo_workflows|deberta-v3-large"
  "ce|new_ajo_workflows|e5-large-v2"
)
PASS2=( "bi|ajo_doc_not_tier1|deberta-v3-large" "ce|ajo_doc_not_tier1|deberta-v3-large" )
SMOKE="ce|new_aep_wf_scrap|e5-large-v2"

run_dir_of() { [[ $1 == bi ]] && echo "$BI_RUNS/$2/$3" || echo "$CE_RUNS/$2/$3"; }

# Back up + clear a collapsed run's markers, UNLESS it is already fixed (test_acc > 0.55).
clear_run() {
  IFS='|' read -r sweep ds md <<< "$1"
  local rd; rd=$(run_dir_of "$sweep" "$ds" "$md")
  [[ -d $rd ]] || { echo "  MISSING dir, skip: $rd"; return; }
  local acc
  acc=$(python3 -c "import json;print(json.load(open('$rd/run_summary.json')).get('test_acc'))" 2>/dev/null || echo none)
  if [[ $acc != none ]] && awk "BEGIN{exit !($acc>0.55)}" 2>/dev/null; then
    echo "  already fixed (test_acc=$acc), skip: $sweep/$ds/$md"; return
  fi
  echo "  clearing $sweep/$ds/$md (test_acc=$acc) -> collapsed_bak/"
  mkdir -p "$rd/collapsed_bak"
  local f
  for f in best.pt final.pt run_summary.json ood_results.json test_metrics.json \
           test_predictions.jsonl config.json train.log stdout.log ood_stdout.log; do
    [[ -e "$rd/$f" ]] && mv -f "$rd/$f" "$rd/collapsed_bak/" || true
  done
}

# Run the two launchers SEQUENTIALLY (each owns all 4 GPUs — never concurrently).
launch_sequential() {
  echo "== [1/2] bi-encoder sweep (retrains only cleared runs, then OOD + aggregate) =="
  bash "$BI_LAUNCHER"
  echo "== [2/2] cross-encoder sweep =="
  bash "$CE_LAUNCHER"
  echo "== done; both results/final_results.json refreshed =="
}

smoke_verdict() {
  IFS='|' read -r sweep ds md <<< "$SMOKE"
  local rd; rd=$(run_dir_of "$sweep" "$ds" "$md")
  # e5/bge escape collapse slowly, so gate on the FINAL test_acc being clearly off
  # the 0.5000 collapse pin rather than on epoch-1 loss.
  local acc; acc=$(python3 -c "import json;print(json.load(open('$rd/run_summary.json')).get('test_acc'))" 2>/dev/null || echo none)
  echo "== smoke: final test_acc=${acc:-none}  (want >0.53; collapse pins at 0.5000) =="
  if [[ $acc != none ]] && awk "BEGIN{exit !($acc>0.53)}" 2>/dev/null; then
    echo "SMOKE PASS  -> safe to run: nohup bash $0 pass1 > /mnt/localssd/rerun_pass1.log 2>&1 &"
  else
    echo "SMOKE FAIL  -> do NOT run pass1; revisit recipe."
  fi
}

case "${1:-}" in
  smoke) echo "### SMOKE: $SMOKE ###"; clear_run "$SMOKE"; bash "$CE_LAUNCHER"; smoke_verdict ;;
  pass1) echo "### PASS 1 (18 runs) ###"; for t in "${PASS1[@]}"; do clear_run "$t"; done; launch_sequential ;;
  pass2) echo "### PASS 2 (2 runs) ###"; for t in "${PASS2[@]}"; do clear_run "$t"; done; launch_sequential ;;
  *) echo "usage: $0 {smoke|pass1|pass2}"; exit 1 ;;
esac
