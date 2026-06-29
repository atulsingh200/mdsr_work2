#!/usr/bin/env python3
"""
Load predictions.jsonl (from run_eval.py) and produce:
  - analysis.json   : failed examples + per-bucket stats
  - plots/sim_distribution.png
  - plots/sim_vs_accuracy.png
  - plots/tier_pair_heatmap.png
  - plots/prob_vs_sim_scatter.png

Usage (from repo root):
  python analysis/run_analysis.py \
      --predictions analysis/predictions.jsonl \
      --out-dir analysis/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_predictions(path: Path) -> pd.DataFrame:
    rows = [json.loads(ln) for ln in open(path)]
    return pd.DataFrame(rows)


def plot_sim_distribution(df: pd.DataFrame, out: Path) -> None:
    correct = df[df["correct"]]["sem_sim"]
    wrong = df[~df["correct"]]["sem_sim"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, alpha=0.6, color="steelblue", label=f"Correct (n={len(correct)})", density=True)
    ax.hist(wrong, bins=bins, alpha=0.6, color="tomato", label=f"Wrong (n={len(wrong)})", density=True)
    ax.axvline(correct.mean(), color="steelblue", linestyle="--", linewidth=1.5, label=f"Correct mean={correct.mean():.3f}")
    ax.axvline(wrong.mean(), color="tomato", linestyle="--", linewidth=1.5, label=f"Wrong mean={wrong.mean():.3f}")
    ax.set_xlabel("Semantic Similarity (all-MiniLM-L6-v2)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Semantic Similarity: Correct vs. Wrong Predictions", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analysis] saved {out}")


def plot_sim_vs_accuracy(df: pd.DataFrame, out: Path, n_bins: int = 10) -> None:
    bins = np.linspace(df["sem_sim"].min(), df["sem_sim"].max(), n_bins + 1)
    df = df.copy()
    df["sim_bin"] = pd.cut(df["sem_sim"], bins=bins, include_lowest=True)
    grouped = df.groupby("sim_bin", observed=True)["correct"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["sim_bin", "accuracy", "n"]
    labels = [str(b) for b in grouped["sim_bin"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(grouped)), grouped["accuracy"], color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Semantic Similarity Bucket", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy vs. Semantic Similarity Bucket", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    # Annotate with sample count
    for i, (bar, row) in enumerate(zip(bars, grouped.itertuples())):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={row.n}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analysis] saved {out}")


def plot_tier_pair_heatmap(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    # Normalise so tier_a <= tier_b for consistent grouping
    df["ta"] = df[["tier_1", "tier_2"]].min(axis=1)
    df["tb"] = df[["tier_1", "tier_2"]].max(axis=1)
    pivot = df.groupby(["ta", "tb"])["correct"].mean().reset_index()
    pivot.columns = ["tier_a", "tier_b", "accuracy"]
    all_tiers = sorted(set(pivot["tier_a"]) | set(pivot["tier_b"]))
    matrix = pd.DataFrame(index=all_tiers, columns=all_tiers, dtype=float)
    for _, row in pivot.iterrows():
        matrix.loc[row["tier_a"], row["tier_b"]] = row["accuracy"]
        matrix.loc[row["tier_b"], row["tier_a"]] = row["accuracy"]

    fig, ax = plt.subplots(figsize=(max(8, len(all_tiers)), max(6, len(all_tiers) - 1)))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"label": "Accuracy"},
        annot_kws={"size": 8},
    )
    ax.set_title("Per Tier-Pair Accuracy (symmetric)", fontsize=13)
    ax.set_xlabel("Tier", fontsize=11)
    ax.set_ylabel("Tier", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analysis] saved {out}")


def plot_prob_vs_sim_scatter(df: pd.DataFrame, out: Path) -> None:
    correct = df[df["correct"]]
    wrong = df[~df["correct"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(correct["sem_sim"], correct["prob"], alpha=0.3, s=6,
               color="steelblue", label=f"Correct (n={len(correct)})", rasterized=True)
    ax.scatter(wrong["sem_sim"], wrong["prob"], alpha=0.5, s=8,
               color="tomato", label=f"Wrong (n={len(wrong)})", rasterized=True)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision boundary (0.5)")
    ax.set_xlabel("Semantic Similarity (all-MiniLM-L6-v2)", fontsize=12)
    ax.set_ylabel("Predicted Probability (P[label=1])", fontsize=12)
    ax.set_title("Predicted Probability vs. Semantic Similarity", fontsize=13)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] saved {out}")


def build_analysis_json(df: pd.DataFrame, n_bins: int = 10) -> dict:
    n_total = len(df)
    n_correct = int(df["correct"].sum())
    n_failed = n_total - n_correct

    correct_sims = df[df["correct"]]["sem_sim"]
    failed_sims = df[~df["correct"]]["sem_sim"]

    # Bin-level accuracy stats
    bins = np.linspace(df["sem_sim"].min(), df["sem_sim"].max(), n_bins + 1)
    df = df.copy()
    df["sim_bin"] = pd.cut(df["sem_sim"], bins=bins, include_lowest=True)
    sim_bins = []
    for b, grp in df.groupby("sim_bin", observed=True):
        sim_bins.append({
            "range": str(b),
            "n": int(len(grp)),
            "n_correct": int(grp["correct"].sum()),
            "accuracy": round(float(grp["correct"].mean()), 4),
        })

    failed_df = df[~df["correct"]].copy()
    # Sort by sem_sim descending (hardest / most confusing first)
    failed_df = failed_df.sort_values("sem_sim", ascending=False)

    failed_examples = []
    for _, row in failed_df.iterrows():
        failed_examples.append({
            "idx": int(row["idx"]),
            "text_1": row["text_1"],
            "text_2": row["text_2"],
            "label": int(row["label"]),
            "pred": int(row["pred"]),
            "prob": round(float(row["prob"]), 6),
            "sem_sim": round(float(row["sem_sim"]), 6),
            "tier_1": int(row["tier_1"]),
            "tier_2": int(row["tier_2"]),
            "sub_1": str(row.get("sub_1", "")),
            "sub_2": str(row.get("sub_2", "")),
            "url_1": str(row.get("url_1", "")),
            "url_2": str(row.get("url_2", "")),
        })

    return {
        "summary": {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_failed": n_failed,
            "accuracy": round(n_correct / n_total, 4),
            "avg_sem_sim_correct": round(float(correct_sims.mean()), 4),
            "avg_sem_sim_failed": round(float(failed_sims.mean()), 4),
            "std_sem_sim_correct": round(float(correct_sims.std()), 4),
            "std_sem_sim_failed": round(float(failed_sims.std()), 4),
            "sim_bins": sim_bins,
        },
        "failed_examples": failed_examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="analysis/predictions.jsonl",
                    help="Output of run_eval.py")
    ap.add_argument("--out-dir", default="analysis/", help="Directory for outputs")
    ap.add_argument("--sim-bins", type=int, default=10, help="Number of similarity buckets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] loading {args.predictions}", flush=True)
    df = load_predictions(Path(args.predictions))
    print(f"[analysis] {len(df)} examples  "
          f"correct={df['correct'].sum()}  wrong={(~df['correct']).sum()}", flush=True)

    # ---- Plots ---- #
    plot_sim_distribution(df, plots_dir / "sim_distribution.png")
    plot_sim_vs_accuracy(df, plots_dir / "sim_vs_accuracy.png", n_bins=args.sim_bins)
    plot_tier_pair_heatmap(df, plots_dir / "tier_pair_heatmap.png")
    plot_prob_vs_sim_scatter(df, plots_dir / "prob_vs_sim_scatter.png")

    # ---- analysis.json ---- #
    result = build_analysis_json(df, n_bins=args.sim_bins)
    out_json = out_dir / "analysis.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[analysis] wrote {out_json}  "
          f"({result['summary']['n_failed']} failed examples)", flush=True)

    # ---- Print quick summary ---- #
    s = result["summary"]
    print("\n========== ANALYSIS SUMMARY ==========")
    print(f"  total       : {s['n_total']}")
    print(f"  correct     : {s['n_correct']}")
    print(f"  failed      : {s['n_failed']}")
    print(f"  accuracy    : {s['accuracy']:.4f}")
    print(f"  avg sim (correct) : {s['avg_sem_sim_correct']:.4f} ± {s['std_sem_sim_correct']:.4f}")
    print(f"  avg sim (failed)  : {s['avg_sem_sim_failed']:.4f} ± {s['std_sem_sim_failed']:.4f}")
    print("=======================================\n")
    print("Accuracy per similarity bucket:")
    for b in s["sim_bins"]:
        print(f"  {b['range']:20s}  n={b['n']:5d}  acc={b['accuracy']:.4f}")


if __name__ == "__main__":
    main()
