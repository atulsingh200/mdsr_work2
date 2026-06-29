#!/usr/bin/env bash
# Build the workflow-ordered causal dataset (~300k pairs, text-level split, 50/50 labels).
# Run from anywhere; paths are absolute.
set -euo pipefail

PY=python3
SCRIPT=/mnt/localssd/automation/internship-causal-embedding/src/followup_data/builders/build_classification_data_workflow.py
CORPUS=/mnt/localssd/automation/internship-causal-embedding/data/comparison_test/aep_docs_collection_v_24-03-2026.json
OUTDIR=/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_workflow_v1

"$PY" "$SCRIPT" \
  --corpus  "$CORPUS" \
  --out-dir "$OUTDIR" \
  --unit sentence --min-sentences 1 --max-sentences 3 \
  --pair-scope all \
  --base-cap 3000 --gap-decay 0.055 --far-gap-floor 250 \
  --intra-doc-max-gap 5 --intra-doc-cap 60 \
  --intra-tier-base-cap 900 \
  --split-level text \
  --seed 42 --val-frac 0.15 --test-frac 0.15

echo ""
echo "=== quick checks ==="
echo -n "recommendation-more-help (want 0): "; grep -ch "recommendation-more-help" "$OUTDIR"/*.jsonl | paste -sd+ | bc
echo -n "raw UUIDs (want 0):                "; grep -chE "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" "$OUTDIR"/*.jsonl | paste -sd+ | bc
echo -n "Documentation breadcrumb (want 0): "; grep -ch "Documentation Journey Optimizer\|Documentation Tutorials" "$OUTDIR"/*.jsonl | paste -sd+ | bc
echo ""
"$PY" - "$OUTDIR" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1] + "/directional_manifest.json"))
print("total samples     :", m["total_samples"])
print("per split         :", m["sample_counts"])
print("per source        :", m["source_counts"])
print("label balance     :", m["label_balance"], "(must be equal)")
print("leak check        :", {k: m["leak_check"][k] for k in
      ("train_inter_val","train_inter_test","val_inter_test")}, "(all must be 0)")
PYEOF
