#!/usr/bin/env python3
"""Aggregate baseline sweep results into a single final_results.json.

Walks {root}/{dataset_slug}/{model_slug}/ and reads:
  - run_summary.json   (in-domain train/val/test metrics)
  - ood_results.json   (Milan + AJO OOD eval)

Merges into a structured JSON keyed by dataset then model.
Missing runs produce null entries so partial completions are handled.

Usage:
  python aggregate_results.py \
      --root /mnt/localssd/baselines_output \
      --out  /mnt/localssd/baselines_output/final_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = [
    "aep_causal_cls34",
    "aep_dataset",
    "aep_causal_wf_v3",
    "ajo_doc_not_tier1",
    "ajo_newstyle",
    "new_aep_wf_scrap",
    "new_ajo_workflows",
]

MODELS = [
    "deberta-v3-large",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "e5-large-v2",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def aggregate(root: Path) -> dict:
    results: dict = {}

    for ds in DATASETS:
        results[ds] = {}
        for model in MODELS:
            run_dir = root / ds / model
            summary = _load_json(run_dir / "run_summary.json")
            ood = _load_json(run_dir / "ood_results.json")

            if summary is None and ood is None:
                results[ds][model] = None
                continue

            entry: dict = {}

            # In-domain metrics
            if summary:
                entry["base_model"] = summary.get("base_model", "")
                entry["schema"] = summary.get("schema", "")
                entry["n_train"] = summary.get("n_train")
                entry["n_test"] = summary.get("n_test")
                entry["epochs"] = summary.get("epochs")
                entry["best_val_acc"] = summary.get("best_val_acc")
                entry["best_val_auc"] = summary.get("best_val_auc")
                entry["test_acc"] = summary.get("test_acc")
                entry["test_auc"] = summary.get("test_auc")
                entry["test_f1"] = summary.get("test_f1")

            # OOD metrics
            if ood:
                milan = ood.get("milan", {})
                ajo = ood.get("ajo_ordering", {})

                entry["milan"] = {
                    "n_events": milan.get("n_events"),
                    "ground_truth_order": milan.get("ground_truth_order"),
                    "predicted_order": milan.get("predicted_order"),
                    "order_match": milan.get("order_match"),
                    "pairwise_acc": milan.get("pairwise_acc"),
                    "n_pairs_correct": milan.get("n_pairs_correct"),
                    "n_pairs_total": milan.get("n_pairs_total"),
                    "per_event": milan.get("per_event", {}),
                }

                entry["ajo_ordering"] = {
                    "n_samples": ajo.get("n_samples"),
                    "n_correct": ajo.get("n_correct"),
                    "perfect_order_rate": ajo.get("perfect_order_rate"),
                    "per_sample": ajo.get("per_sample", []),
                }
            else:
                entry["milan"] = None
                entry["ajo_ordering"] = None

            results[ds][model] = entry

    return results


def print_summary(results: dict) -> None:
    """Print a compact leaderboard to stdout."""
    header = f"{'Dataset':<22} {'Model':<22} {'test_acc':>8} {'test_auc':>8} {'test_f1':>7} {'milan_pairw':>11} {'ajo_perfect':>11}"
    print(header)
    print("-" * len(header))
    for ds in DATASETS:
        for model in MODELS:
            entry = results.get(ds, {}).get(model)
            if entry is None:
                print(f"{ds:<22} {model:<22}  {'(missing)':>8}")
                continue
            ta = entry.get("test_acc")
            au = entry.get("test_auc")
            f1 = entry.get("test_f1")
            milan_acc = (entry.get("milan") or {}).get("pairwise_acc")
            ajo_rate = (entry.get("ajo_ordering") or {}).get("perfect_order_rate")
            fmt = lambda v: f"{v:.4f}" if v is not None else "  n/a "
            print(
                f"{ds:<22} {model:<22} {fmt(ta):>8} {fmt(au):>8} {fmt(f1):>7} "
                f"{fmt(milan_acc):>11} {fmt(ajo_rate):>11}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate baseline sweep results.")
    ap.add_argument("--root", required=True, help="Output root directory of the sweep.")
    ap.add_argument("--out", required=True, help="Path for final_results.json.")
    args = ap.parse_args()

    root = Path(args.root)
    results = aggregate(root)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")

    print()
    print_summary(results)

    # Quick stats
    total = sum(
        1 for ds in DATASETS for m in MODELS if results.get(ds, {}).get(m) is not None
    )
    print(f"\n{total}/{len(DATASETS) * len(MODELS)} runs completed.")


if __name__ == "__main__":
    main()
