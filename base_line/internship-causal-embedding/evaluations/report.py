"""Result formatting: JSON, CSV, and console table output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_json(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {path}")


def save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    k_values = sorted(results[0].get("recall_at_k", {}).keys())
    fieldnames = [
        "model", "dataset", "split", "negative_strategy",
        "n_pairs", "pool_size", "auc", "precision_at_1",
        "mrr", "mean_rank", "median_rank",
    ] + [f"recall_at_{k}" for k in k_values] + [f"hit_at_{k}" for k in k_values]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in k_values:
                row[f"recall_at_{k}"] = r.get("recall_at_k", {}).get(k, "")
                row[f"hit_at_{k}"] = r.get("hit_at_k", {}).get(k, "")
            writer.writerow(row)
    print(f"[saved] {path}")


def print_table(results: list[dict]) -> None:
    if not results:
        return
    k_values = sorted(results[0].get("recall_at_k", {}).keys())
    k_show = [k for k in k_values if k in (1, 5, 10)]

    # Header
    header = (
        f"{'Model':<35} {'Dataset':<14} {'Neg':<8} "
        f"{'AUC':>7} {'P@1':>7} {'MRR':>7} "
    )
    for k in k_show:
        header += f"{'R@'+str(k):>7} "
    header += f"{'MnRank':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"  ERROR: {r}")
            continue
        row = (
            f"{r['model']:<35} {r['dataset']:<14} {r['negative_strategy']:<8} "
            f"{r['auc']:>7.4f} {r['precision_at_1']:>7.4f} {r['mrr']:>7.4f} "
        )
        for k in k_show:
            row += f"{r['recall_at_k'].get(k, 0):>7.4f} "
        row += f"{r['mean_rank']:>8.2f}"
        print(row)
    print("=" * len(header))
