"""Aggregate the LLMERE baseline into two tables + a figure.

Table A (accuracy + inference cost) — pairwise causal classification on final_data test:
  encoder DeBERTa-CE / reasoning-CE (papers 1-2) · label-only LoRA · rationale LoRA (LLMERE) ·
  ICL-CoT (non-FT). micro-F1 (=acc), per-source, and inference nature.
Table B (efficiency) — per-event O(n) extraction vs pairwise O(n^2).

Reads outputs/llmere/*.json + outputs/encoder + outputs/lora (paper-1) + outputs/consistency.
Writes outputs/llmere/{table_accuracy.md, table_efficiency.md, llmere_bar.png}.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "outputs"
LL = OUT / "llmere"
SRC = ["procedural", "aep", "ajo"]


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def acc_row(name, params, rationale, cost, d, key_overall="overall", by="by_source"):
    if d is None:
        return None
    o = d[key_overall] if key_overall in d else d
    return {"system": name, "params": params, "rationale": rationale, "cost": cost,
            "micro_f1": o["acc"], "macro_f1": round((o["f1_before"] + o["f1_not_before"]) / 2, 4)
            if "f1_before" in o else None,
            **{f"src_{s}": (d.get(by, {}).get(s, {}) or {}).get("acc") for s in SRC}}


def main():
    rows = []
    # encoder + reasoning-CE (from paper-2 consistency eval on final_data test is not stored; use
    # paper-1 encoder json + reasoning-CE test_metrics)
    enc = _read(OUT / "encoder/deberta_ce2x_on_final_test.json")
    if enc:
        rows.append(acc_row("DeBERTa-CE (encoder)", "435M", "no", "1 fwd (encoder)", enc))
    # reasoning-CE: its own test_metrics.json (final_data test) — pull acc/f1
    rc = _read(Path("/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/"
                    "runs/crossencoder2x_deberta_reasoning/v2_fixed_alpha02_ep10/test_metrics.json"))
    if rc:
        rows.append({"system": "Reasoning-CE (OURS)", "params": "435M", "rationale": "train-time",
                     "cost": "1 fwd (encoder)", "micro_f1": rc.get("acc"),
                     "macro_f1": None, **{f"src_{s}": None for s in SRC}})
    # label-only LoRA (matched LLMERE hp) + paper-1 label-only
    for tag, name in [("Qwen2.5-7B-Instruct__QA1__lora", "LoRA label-only (Qwen-7B, matched hp)")]:
        d = _read(LL / f"{tag}.json") or _read(OUT / f"lora/{tag}.json")
        if d:
            rows.append(acc_row(name, "7B", "no", "autoregressive (short)", d.get("test", d)))
    # rationale LoRA (LLMERE) — 3 configs; report each present
    for sub, cfg in [("llmere", "2e-4/r64"), ("llmere_lr1e4", "1e-4/r64"), ("llmere_r32", "r32/1e-4")]:
        d = _read(OUT / sub / "Qwen2.5-7B-Instruct__rationale__lora.json")
        if d:
            rows.append(acc_row(f"LoRA rationale (LLMERE, {cfg})", "7B", "train+infer",
                                "autoregressive (long)", d.get("test", d)))
    # ICL-CoT reference
    for k in (0, 2):
        d = _read(LL / f"icl_Qwen2.5-7B-Instruct__cot__k{k}.json")
        if d:
            rows.append(acc_row(f"ICL-CoT (Qwen-7B, k={k}, no FT)", "7B", "infer",
                                "autoregressive (long)", d))
    rows = [r for r in rows if r]

    hdr = ["system", "params", "rationale", "inference", "micro_f1", "macro_f1"] + [f"src_{s}" for s in SRC]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in sorted(rows, key=lambda x: -(x["micro_f1"] or 0)):
        fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join([
            r["system"], r["params"], r["rationale"], r["cost"],
            fmt(r["micro_f1"]), fmt(r["macro_f1"])] + [fmt(r.get(f"src_{s}")) for s in SRC]) + " |")
    tableA = "\n".join(lines)
    (LL / "table_accuracy.md").write_text("## Table A — accuracy + inference cost\n\n" + tableA + "\n")

    # Table B — efficiency
    pe = _read(LL / "Qwen2.5-7B-Instruct__perevent__lora.json")
    if pe:
        tB = ("## Table B — per-event O(n) vs pairwise O(n^2)\n\n"
              "| method | relation-F1 | precision | recall | forwards/doc | total forwards | speedup |\n"
              "|---|---|---|---|---|---|---|\n"
              f"| LLMERE per-event O(n) (Qwen-7B) | {pe['f1']:.4f} | {pe['precision']:.4f} | "
              f"{pe['recall']:.4f} | O(n) | {pe['queries_On']} | — |\n"
              f"| pairwise O(n^2) (same docs) | — | — | — | O(n^2) | {pe['pairs_On2']} | "
              f"{pe['speedup_x']}x more forwards |\n")
        (LL / "table_efficiency.md").write_text(tB)
        print(tB)

    print(tableA)

    # bar figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r["system"].replace(" (", "\n(") for r in rows]
        vals = [r["micro_f1"] for r in rows]
        colors = ["#2b8cbe" if "CE" in r["system"] else "#e6550d" for r in rows]
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(rows)), 4.8))
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel("micro-F1 (final_data test)"); ax.set_ylim(0, 1)
        ax.set_title("LLMERE-style rationale LLM vs encoders (causal directional TRC)")
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        fig.tight_layout(); fig.savefig(LL / "llmere_bar.png", dpi=150)
        print(f"wrote {LL/'llmere_bar.png'}")
    except Exception as e:
        print("(figure skipped)", e)


if __name__ == "__main__":
    main()
