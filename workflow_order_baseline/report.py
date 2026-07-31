"""Aggregate per-(model, shot) outputs into a Table-1-style report (CSV + markdown)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / "outputs"

# paper Table 1 numbers (food-recipe domain, 6-8 steps) for reference / trend comparison
# Note: paper used Qwen3-8B; we use Qwen2.5-7B-Instruct (vllm 0.6.6 does not support Qwen3)
PAPER = {
    ("Mistral-7B-Instruct-v0.2", 0): (0.29, 0.61, 0.73, 0.55),
    ("Mistral-7B-Instruct-v0.2", 3): (0.32, 0.66, 0.79, 0.51),
    ("Mistral-7B-Instruct-v0.2", 5): (0.31, 0.66, 0.79, 0.51),
    ("Qwen2.5-7B-Instruct", 0): (0.71, 0.88, 0.92, 0.22),   # paper's Qwen3-8B numbers for trend ref
    ("Qwen2.5-7B-Instruct", 3): (0.63, 0.82, 0.88, 0.30),
    ("Qwen2.5-7B-Instruct", 5): (0.62, 0.81, 0.87, 0.30),
}


def main() -> None:
    rows = []
    for f in sorted(OUT_DIR.glob("*__k*.json")):
        d = json.loads(f.read_text())
        m = d["metrics"]
        model = d["model"].split("/")[-1]
        rows.append({"model": model, "shots": d["shots"], **m})

    rows.sort(key=lambda r: (r["model"], r["shots"]))

    csv_path = OUT_DIR / "results_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "shots", "acc", "nlcs", "ktau", "ned",
                    "unparsed_rate", "n_parsed", "n"])
        for r in rows:
            w.writerow([r["model"], r["shots"], f"{r['acc']:.4f}", f"{r['nlcs']:.4f}",
                        f"{r['ktau']:.4f}", f"{r['ned']:.4f}", f"{r['unparsed_rate']:.4f}",
                        r["n_parsed"], r["n"]])

    lines = [
        "# Workflow Step-Ordering Baseline — Results",
        "",
        "Task and metrics follow arXiv:2511.04688v2 (inference-only; the paper did NOT fine-tune).",
        "Eval set: 697 Adobe workflow items (2-4 steps). Note step counts are much shorter than",
        "the paper's recipe domain (6-8 steps), so absolute numbers are not directly comparable —",
        "the paper column is shown only for trend/ranking context.",
        "",
        "Acc / NLCS / KTau higher = better; NED lower = better.",
        "",
        "| Model | Shots | Acc | NLCS | KTau | NED | Unparsed | (paper: Acc/NLCS/KTau/NED) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = PAPER.get((r["model"], r["shots"]))
        pstr = f"{p[0]}/{p[1]}/{p[2]}/{p[3]}" if p else "—"
        lines.append(f"| {r['model']} | {r['shots']} | {r['acc']:.3f} | {r['nlcs']:.3f} | "
                     f"{r['ktau']:.3f} | {r['ned']:.3f} | {r['unparsed_rate']:.3f} | {pstr} |")
    md = "\n".join(lines) + "\n"
    (OUT_DIR / "results_table.md").write_text(md)
    print(md)
    print(f"wrote {csv_path} and results_table.md")


if __name__ == "__main__":
    main()
