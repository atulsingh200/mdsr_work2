"""Build the headline consistency table + accuracy-vs-cycle-rate scatter (paper 2 Fig 2 style)."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "outputs/consistency"
LABELS = {
    "baseline_plain": "Baseline CE (plain)",
    "baseline_confidence": "Baseline CE + confidence-break",
    "baseline_llm": "Baseline CE + LLM-assisted break",
    "reasoning_plain": "OURS: Reasoning-CE (plain)",
    "reasoning_confidence": "OURS: Reasoning-CE + confidence-break",
}
ORDER = ["baseline_plain", "baseline_confidence", "baseline_llm",
         "reasoning_plain", "reasoning_confidence"]


def table(res, sl):
    hdr = ["system", "LLM@inf", "cycle_rate", "mean_cyc", "ordering_acc", "edges_removed", "llm_calls"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for k in ORDER:
        if k not in res:
            continue
        m = res[k]["metrics"][sl]
        lines.append("| " + " | ".join([
            LABELS[k], "yes" if res[k]["llm_at_inference"] else "no",
            f"{m['cycle_rate']:.3f}", f"{m['mean_cycles_per_wf']:.3f}",
            f"{m['ordering_accuracy']:.4f}", str(m["edges_removed"]), str(m["llm_calls"]),
        ]) + " |")
    return "\n".join(lines)


def scatter(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip figure: {e})"); return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"baseline_plain": "#e6550d", "baseline_confidence": "#fd8d3c",
              "baseline_llm": "#d62728", "reasoning_plain": "#2b8cbe",
              "reasoning_confidence": "#3690c0"}
    for k in ORDER:
        if k not in res:
            continue
        m = res[k]["metrics"]["procedural"]
        marker = "*" if k.startswith("reasoning") else "o"
        ax.scatter(m["cycle_rate"], m["ordering_accuracy"], s=220 if marker == "*" else 120,
                   c=colors[k], marker=marker, edgecolors="black", linewidths=0.6, label=LABELS[k])
    ax.set_xlabel("cycle rate  (lower = more consistent)")
    ax.set_ylabel("ordering accuracy (micro-F1)")
    ax.set_title("Accuracy vs. consistency — procedural workflows (5–8 steps)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    res = json.loads((OUT / "consistency_results.json").read_text())
    md = []
    for sl in ["procedural", "proc_n>=6", "all", "curated"]:
        md.append(f"### slice: {sl}\n")
        md.append(table(res, sl))
        md.append("")
    (OUT / "consistency_table.md").write_text("\n".join(md))
    scatter(res, OUT / "consistency_scatter.png")
    print("\n".join(md))

    # claim check on procedural slice
    bp = res["baseline_plain"]["metrics"]["procedural"]
    rp = res["reasoning_plain"]["metrics"]["procedural"]
    print(f"\nprocedural: baseline cycle_rate={bp['cycle_rate']:.3f} acc={bp['ordering_accuracy']:.4f}"
          f"  |  OURS(reasoning) cycle_rate={rp['cycle_rate']:.3f} acc={rp['ordering_accuracy']:.4f}")
    verdict = ("HOLDS: reasoning-CE is natively more consistent (fewer cycles) with no inference LLM."
               if rp["cycle_rate"] < bp["cycle_rate"] else
               "reasoning-CE cycle_rate not lower than baseline on this slice.")
    print(verdict)
    if "baseline_llm" in res:
        bl = res["baseline_llm"]["metrics"]["procedural"]
        bc = res["baseline_confidence"]["metrics"]["procedural"]
        print(f"paper-parity: confidence-break acc={bc['ordering_accuracy']:.4f} vs "
              f"LLM-assisted acc={bl['ordering_accuracy']:.4f} "
              f"({'confidence >= LLM (matches paper)' if bc['ordering_accuracy'] >= bl['ordering_accuracy'] else 'LLM > confidence'})")


if __name__ == "__main__":
    main()
