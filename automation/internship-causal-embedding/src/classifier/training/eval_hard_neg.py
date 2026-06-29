#!/usr/bin/env python3
"""
Evaluate a checkpoint on the hard-negative test set and produce a detailed
JSON report with micro/macro accuracy and adjacent vs. non-adjacent breakdown.

Invoke (from repo root):
  python3 -m src.classifier.training.eval_hard_neg \
      --ckpt runs/classifier/mlp_bge_small_new/best.pt \
      --data-dir data/aep_causal_classification_hard_neg_test

Output:
  runs/classifier/mlp_bge_small_new/hard_neg_test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def acc_stats(rows: list[dict]) -> dict:
    """Return micro accuracy, macro accuracy (mean per-tier-pair), n, and
    per-tier-pair breakdown for a list of prediction rows."""
    if not rows:
        return {"n": 0, "micro_acc": float("nan"), "macro_acc": float("nan"), "per_tier_pair": {}}

    correct = [int((r["prob"] > 0.5) == (r["label"] == 1)) for r in rows]
    micro = float(np.mean(correct))

    # Per-tier-pair
    pair_stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r, c in zip(rows, correct):
        key = f"T{min(r['tier_1'], r['tier_2'])}-T{max(r['tier_1'], r['tier_2'])}"
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += c

    per_pair = {
        k: {"n": v["n"], "acc": v["correct"] / v["n"]}
        for k, v in sorted(pair_stats.items(), key=lambda x: [int(t[1:]) for t in x[0].split("-")])
    }
    macro = float(np.mean([v["acc"] for v in per_pair.values()]))

    return {
        "n": len(rows),
        "micro_acc": round(micro, 6),
        "macro_acc": round(macro, 6),
        "per_tier_pair": per_pair,
    }


def split_adjacent(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into adjacent (|tier_1 - tier_2| == 1) and non-adjacent."""
    adj, non_adj = [], []
    for r in rows:
        if abs(r["tier_1"] - r["tier_2"]) == 1:
            adj.append(r)
        else:
            non_adj.append(r)
    return adj, non_adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for eval_only.py subprocess")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    data_dir  = Path(args.data_dir)
    out_dir   = ckpt_path.parent

    # ------------------------------------------------------------------
    # Step 1: run eval_only.py --write-out to get per-row predictions
    # ------------------------------------------------------------------
    pred_file = out_dir / f"{args.split}_predictions_eval.jsonl"

    cmd = [
        args.python, "-m", "src.classifier.training.eval_only",
        "--ckpt",           str(ckpt_path),
        "--data-dir",       str(data_dir),
        "--split",          args.split,
        "--eval-batch-size", str(args.eval_batch_size),
        "--write-out",
    ]
    print(f"[eval_hard_neg] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # Step 2: load predictions + original rows (for hard_neg flag)
    # ------------------------------------------------------------------
    preds = []
    with open(pred_file) as f:
        for ln in f:
            preds.append(json.loads(ln))

    orig_rows = []
    with open(data_dir / f"directional_{args.split}.jsonl") as f:
        for ln in f:
            orig_rows.append(json.loads(ln))

    assert len(preds) == len(orig_rows), (
        f"Prediction count {len(preds)} != original row count {len(orig_rows)}"
    )

    # Attach hard_neg flag (and sim_y_ystar if present) to each pred row
    for pred, orig in zip(preds, orig_rows):
        pred["hard_neg"]    = orig.get("hard_neg", False)
        pred["sim_y_ystar"] = orig.get("sim_y_ystar", None)

    # ------------------------------------------------------------------
    # Step 3: compute breakdowns
    # ------------------------------------------------------------------
    all_rows   = preds
    pos_rows   = [r for r in preds if r["label"] == 1]               # original positives
    hn_rows    = [r for r in preds if r.get("hard_neg")]              # hard negatives

    adj_all,     non_adj_all     = split_adjacent(all_rows)
    adj_pos,     non_adj_pos     = split_adjacent(pos_rows)
    adj_hn,      non_adj_hn      = split_adjacent(hn_rows)

    report = {
        "ckpt":     str(ckpt_path),
        "data_dir": str(data_dir),
        "split":    args.split,

        # Overall (all rows: positives + hard negatives)
        "overall": acc_stats(all_rows),

        # Positives only  (x->y, label=1)
        "positives_only": {
            "all":          acc_stats(pos_rows),
            "adjacent":     acc_stats(adj_pos),
            "non_adjacent": acc_stats(non_adj_pos),
        },

        # Hard negatives only  (x->y*, label=0)
        "hard_negatives_only": {
            "all":          acc_stats(hn_rows),
            "adjacent":     acc_stats(adj_hn),
            "non_adjacent": acc_stats(non_adj_hn),
        },

        # Combined but split by adjacency
        "by_adjacency": {
            "adjacent":     acc_stats(adj_all),
            "non_adjacent": acc_stats(non_adj_all),
        },
    }

    # ------------------------------------------------------------------
    # Step 4: print summary
    # ------------------------------------------------------------------
    def fmt(stats: dict) -> str:
        return (f"n={stats['n']:5d}  micro={stats['micro_acc']:.4f}  "
                f"macro={stats['macro_acc']:.4f}")

    print("\n==================== HARD NEG TEST RESULTS ====================")
    print(f"  OVERALL               {fmt(report['overall'])}")
    print()
    print(f"  Positives  (all)      {fmt(report['positives_only']['all'])}")
    print(f"  Positives  (adj)      {fmt(report['positives_only']['adjacent'])}")
    print(f"  Positives  (non-adj)  {fmt(report['positives_only']['non_adjacent'])}")
    print()
    print(f"  Hard-neg   (all)      {fmt(report['hard_negatives_only']['all'])}")
    print(f"  Hard-neg   (adj)      {fmt(report['hard_negatives_only']['adjacent'])}")
    print(f"  Hard-neg   (non-adj)  {fmt(report['hard_negatives_only']['non_adjacent'])}")
    print()
    print(f"  Adjacency  (adj)      {fmt(report['by_adjacency']['adjacent'])}")
    print(f"  Adjacency  (non-adj)  {fmt(report['by_adjacency']['non_adjacent'])}")
    print("================================================================\n")

    # ------------------------------------------------------------------
    # Step 5: write JSON report
    # ------------------------------------------------------------------
    out_json = out_dir / "hard_neg_test_metrics.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[eval_hard_neg] wrote {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
