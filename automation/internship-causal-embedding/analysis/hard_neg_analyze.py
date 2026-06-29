"""Hard-negative dataset analysis for best_mlp_bge_small.

Joins directional_test.jsonl (with sim_y_ystar) to the model's predictions by ROW INDEX
(the two files are 1:1 row-aligned, verified), and produces:
  1. analysis/hard_neg_analysis.md     -- write-up with tables + hypothesis verdict
  2. analysis/hard_neg_top_similarity.csv -- all hard negs sorted by sim_y_ystar
  3. analysis/plots/hardneg_error_vs_sim.png
     analysis/plots/hardneg_error_vs_gap.png
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/aep_causal_classification_hard_neg/directional_test.jsonl"
PREDS = ROOT / "runs/classifier/best_mlp_bge_small/test_predictions_eval.jsonl"
OUT = ROOT / "analysis"
PLOTS = OUT / "plots"
THR = 0.5


def load():
    recs = [json.loads(l) for l in open(DATA)]
    preds = [json.loads(l) for l in open(PREDS)]
    assert len(recs) == len(preds), (len(recs), len(preds))
    # verify 1:1 row alignment on full identity
    for r, p in zip(recs, preds):
        key_r = (r["url_1"], r["url_2"], r["sub_1"], r["sub_2"], r["label"], r["tier_1"], r["tier_2"])
        key_p = (p["url_1"], p["url_2"], p["sub_1"], p["sub_2"], p["label"], p["tier_1"], p["tier_2"])
        assert key_r == key_p, "row misalignment -- abort"
    rows = []
    for r, p in zip(recs, preds):
        d = dict(r)
        d["prob"] = p["prob"]
        d["pred"] = 1 if p["prob"] >= THR else 0
        d["correct"] = d["pred"] == d["label"]
        d["gap"] = d["tier_1"] - d["tier_2"]
        rows.append(d)
    return rows


def err(group):
    return sum(not d["correct"] for d in group) / len(group) if group else float("nan")


def acc(group):
    return sum(d["correct"] for d in group) / len(group) if group else float("nan")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = load()
    pos = [d for d in rows if d["label"] == 1]
    hn = [d for d in rows if d["hard_neg"]]

    sims = np.array([d["sim_y_ystar"] for d in hn])
    errs = np.array([0.0 if d["correct"] else 1.0 for d in hn])
    probs = np.array([d["prob"] for d in hn])
    gaps = np.array([d["gap"] for d in hn], dtype=float)

    c_sim_err = float(np.corrcoef(sims, errs)[0, 1])
    c_sim_prob = float(np.corrcoef(sims, probs)[0, 1])
    c_gap_err = float(np.corrcoef(gaps, errs)[0, 1])
    c_gap_sim = float(np.corrcoef(gaps, sims)[0, 1])

    sim_fooled = float(np.mean([d["sim_y_ystar"] for d in hn if d["pred"] == 1]))
    sim_reject = float(np.mean([d["sim_y_ystar"] for d in hn if d["pred"] == 0]))

    # quintiles by sim
    edges = np.quantile(sims, [0, .2, .4, .6, .8, 1.0])
    quint = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        b = [d for d in hn if d["sim_y_ystar"] >= lo and (d["sim_y_ystar"] < hi or i == 4)]
        quint.append((lo, hi, len(b), err(b), float(np.mean([d["prob"] for d in b]))))

    # by tier gap
    gap_rows = []
    for g in sorted({d["gap"] for d in hn}):
        b = [d for d in hn if d["gap"] == g]
        gap_rows.append((g, len(b), err(b), float(np.mean([d["sim_y_ystar"] for d in b]))))

    # ---- CSV: all hard negs sorted by sim_y_ystar desc ----
    csv_path = OUT / "hard_neg_top_similarity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_y_ystar", "prob", "pred", "fooled", "tier_1", "tier_2", "gap",
                    "sub_1", "sub_2", "url_1", "url_2", "text_1", "text_2"])
        for d in sorted(hn, key=lambda x: -x["sim_y_ystar"]):
            w.writerow([f"{d['sim_y_ystar']:.6f}", f"{d['prob']:.6f}", d["pred"],
                        int(d["pred"] == 1), d["tier_1"], d["tier_2"], d["gap"],
                        d["sub_1"], d["sub_2"], d["url_1"], d["url_2"],
                        d["text_1"], d["text_2"]])

    # ---- plot 1: error vs sim quintile ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [f"[{lo:.2f},{hi:.2f}]" for lo, hi, *_ in quint]
    ax.bar(labels, [q[3] for q in quint], color="#c44e52")
    ax.set_xlabel("sim_y_ystar quintile (similarity of y to y*)")
    ax.set_ylabel("hard-neg error rate")
    ax.set_title(f"Hard-neg error vs semantic similarity  (corr={c_sim_err:+.2f})")
    for i, q in enumerate(quint):
        ax.text(i, q[3] + 0.01, f"{q[3]:.2f}\nn={q[2]}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "hardneg_error_vs_sim.png", dpi=130)
    plt.close(fig)

    # ---- plot 2: error vs tier gap ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    gs = [str(g) for g, *_ in gap_rows]
    ax.bar(gs, [r[2] for r in gap_rows], color="#4c72b0")
    ax.set_xlabel("tier gap (tier_1 - tier_2, reversed direction distance)")
    ax.set_ylabel("hard-neg error rate")
    ax.set_title(f"Hard-neg error vs tier gap  (corr={c_gap_err:+.2f})")
    for i, r in enumerate(gap_rows):
        ax.text(i, r[2] + 0.01, f"n={r[1]}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "hardneg_error_vs_gap.png", dpi=130)
    plt.close(fig)

    # ---- markdown report ----
    top12 = []
    seen = set()
    for d in sorted(hn, key=lambda x: -x["sim_y_ystar"]):
        k = (d["url_1"], d["url_2"], d["sub_1"], d["sub_2"])
        if k in seen:
            continue
        seen.add(k)
        top12.append(d)
        if len(top12) == 12:
            break

    md = []
    md.append("# Hard-Negative Dataset Analysis\n")
    md.append(f"Model: `best_mlp_bge_small` (BGE-small + MLP). Threshold {THR}. "
              f"n={len(rows)} (verified 1:1 row-aligned with predictions).\n")
    md.append("## Headline\n")
    md.append(f"- Overall acc **{acc(rows):.4f}** (vs 0.857 on the original eval set).")
    md.append(f"- Positives acc **{acc(pos):.4f}**; hard negatives acc **{acc(hn):.4f}** "
              f"(error **{err(hn):.4f}**) — the entire drop is false positives on hard negs.\n")
    md.append("## Hypothesis: high sim(y,y*) fools the model → **REJECTED (opposite)**\n")
    md.append(f"- corr(sim_y_ystar, error) = **{c_sim_err:+.2f}**, corr(sim_y_ystar, prob) = {c_sim_prob:+.2f} "
              f"(higher similarity → fewer errors).")
    md.append(f"- avg sim_y_ystar: fooled(pred=1) = **{sim_fooled:.3f}** vs rejected(pred=0) = **{sim_reject:.3f}**.\n")
    md.append("Error by sim_y_ystar quintile:\n")
    md.append("| sim quintile | n | error | avg prob |")
    md.append("|---|---|---|---|")
    for lo, hi, n, e, p in quint:
        md.append(f"| [{lo:.3f},{hi:.3f}] | {n} | {e:.3f} | {p:.3f} |")
    md.append("")
    md.append("## Real driver: tier gap (reversed-direction distance)\n")
    md.append(f"corr(tier_gap, error) = {c_gap_err:+.2f}; corr(tier_gap, sim) = {c_gap_sim:+.2f} "
              "(high sim co-occurs with large gaps → generic hub y*, easy to reject).\n")
    md.append("| tier gap | n | error | mean sim |")
    md.append("|---|---|---|---|")
    for g, n, e, s in gap_rows:
        md.append(f"| {g} | {n} | {e:.3f} | {s:.3f} |")
    md.append("")
    md.append("## Top 12 highest-sim_y_ystar hard negatives\n")
    md.append("| sim | prob | result | dir | sub | x (truncated) | y* (truncated) |")
    md.append("|---|---|---|---|---|---|---|")
    for d in top12:
        res = "FOOLED" if d["pred"] == 1 else "rejected"
        x = d["text_1"][:60].replace("|", "/").replace("\n", " ")
        y = d["text_2"][:60].replace("|", "/").replace("\n", " ")
        md.append(f"| {d['sim_y_ystar']:.3f} | {d['prob']:.3f} | {res} | "
                  f"t{d['tier_1']}→t{d['tier_2']} | {d['sub_1']}→{d['sub_2']} | {x} | {y} |")
    md.append("")
    md.append("Full sorted list: `analysis/hard_neg_top_similarity.csv`. "
              "Plots: `analysis/plots/hardneg_error_vs_sim.png`, `analysis/plots/hardneg_error_vs_gap.png`.\n")
    md.append("**Conclusion:** the model is not fooled by content similarity of y/y*; "
              "it fails on small reversed tier gaps (weak directionality) and a few topic pairs "
              "(gap=4: 10f/10c/10d→2a). It leans toward predicting causal=1.\n")
    (OUT / "hard_neg_analysis.md").write_text("\n".join(md))

    # console summary
    print(f"overall acc={acc(rows):.4f}  pos acc={acc(pos):.4f}  hardneg acc={acc(hn):.4f}")
    print(f"corr(sim,error)={c_sim_err:+.3f}  corr(gap,error)={c_gap_err:+.3f}  corr(gap,sim)={c_gap_sim:+.3f}")
    print(f"sim fooled={sim_fooled:.3f}  rejected={sim_reject:.3f}")
    print("wrote:")
    print(" ", OUT / "hard_neg_analysis.md")
    print(" ", csv_path)
    print(" ", PLOTS / "hardneg_error_vs_sim.png")
    print(" ", PLOTS / "hardneg_error_vs_gap.png")


if __name__ == "__main__":
    main()
