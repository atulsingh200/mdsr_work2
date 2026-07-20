"""Aggregate encoder + LLM (ICL & LoRA) results into the headline comparison table + figure.

Reads:
  outputs/encoder/*.json   (DeBERTa CrossEncoder2x on final_data test)
  outputs/icl/*.json        (per-setting; *__agg.json for few-shot mean/std)
  outputs/lora/*.json

Metric convention (matches the paper): for a single-label task micro-F1 == accuracy, so the
headline column `micro_f1` is accuracy. We also report `macro_f1` = mean(F1_BEFORE, F1_NOT_BEFORE),
and per-source micro-F1 (= per-source accuracy). NOTE: the raw per-run JSON also stores a positive-
class F1 under `overall.f1`; we deliberately do NOT use it as the headline because an LLM that
over-predicts BEFORE can inflate it while accuracy stays near chance.

Produces:
  outputs/results_table.md   - markdown table (micro/macro + per-source micro-F1)
  outputs/results.csv        - flat CSV
  outputs/results_bar.png    - bar chart, encoder vs best-LLM-per-method
Asserts the encoder's micro-F1 exceeds every LLM setting's micro-F1 (the paper's claim).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "outputs"
SOURCES = ["procedural", "aep", "ajo"]


def _macro(m: dict):
    b, nb = m.get("f1_before"), m.get("f1_not_before")
    return None if b is None or nb is None else (b + nb) / 2.0


def _src_micro(by_source: dict, s: str):
    return by_source.get(s, {}).get("acc")   # per-source micro-F1 == per-source accuracy


def collect():
    rows = []

    # ── encoder ──────────────────────────────────────────────────────────────
    for f in sorted((OUT / "encoder").glob("*.json")):
        d = json.loads(f.read_text())
        o = d["overall"]
        rows.append({"model": d.get("model", f.stem), "method": "encoder", "setting": "-",
                     "micro_f1": o["acc"], "micro_std": 0.0, "macro_f1": _macro(o),
                     **{f"src_{s}": _src_micro(d.get("by_source", {}), s) for s in SOURCES}})

    # ── icl: prefer *__agg.json (few-shot mean/std); include zero-shot singletons ─
    icl_dir = OUT / "icl"
    agg_bases = set()
    for f in sorted(icl_dir.glob("*__agg.json")):
        d = json.loads(f.read_text())
        agg_bases.add(f.stem.replace("__agg", ""))
        rows.append({"model": d["model"], "method": "ICL",
                     "setting": f"{d['prompt_type']} k{d['shots']}",
                     "micro_f1": d.get("acc_mean"), "micro_std": d.get("acc_std", 0.0),
                     "macro_f1": None,
                     **{f"src_{s}": None for s in SOURCES}})
    for f in sorted(icl_dir.glob("*.json")):
        if f.stem.endswith("__agg") or f.stem in agg_bases or "__s" in f.stem:
            continue
        d = json.loads(f.read_text())
        if "overall" not in d:
            continue
        o = d["overall"]
        rows.append({"model": d["model"], "method": "ICL",
                     "setting": f"{d['prompt_type']} k{d['shots']}",
                     "micro_f1": o["acc"], "micro_std": 0.0, "macro_f1": _macro(o),
                     **{f"src_{s}": _src_micro(d.get("by_source", {}), s) for s in SOURCES}})

    # ── lora ─────────────────────────────────────────────────────────────────
    for f in sorted((OUT / "lora").glob("*.json")):
        d = json.loads(f.read_text())
        if "test" not in d:
            continue
        o = d["test"]["overall"]
        rows.append({"model": d["model"], "method": "LoRA",
                     "setting": f"{d['prompt_type']} ft",
                     "micro_f1": o["acc"], "micro_std": 0.0, "macro_f1": _macro(o),
                     **{f"src_{s}": _src_micro(d["test"].get("by_source", {}), s) for s in SOURCES}})
    return rows


def _fmt(v):
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def to_markdown(rows):
    hdr = ["model", "method", "setting", "micro_f1", "macro_f1"] + [f"src_{s}" for s in SOURCES]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in sorted(rows, key=lambda x: (-(x["micro_f1"] or 0),)):
        micro = _fmt(r["micro_f1"]) + (f" ±{r['micro_std']:.3f}" if r.get("micro_std") else "")
        cells = [r["model"], r["method"], r["setting"], micro, _fmt(r.get("macro_f1"))] + \
                [_fmt(r.get(f"src_{s}")) for s in SOURCES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bar_figure(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping figure: {e})")
        return
    enc = [r for r in rows if r["method"] == "encoder"]
    llm = [r for r in rows if r["method"] != "encoder" and r["micro_f1"] is not None]
    best = {}
    for r in llm:
        key = f"{r['model'].split('/')[-1]}\n{r['method']}"
        if key not in best or r["micro_f1"] > best[key]:
            best[key] = r["micro_f1"]
    labels = (["DeBERTa CE\n(encoder)"] if enc else []) + list(best.keys())
    vals = ([enc[0]["micro_f1"]] if enc else []) + list(best.values())
    colors = (["#2b8cbe"] if enc else []) + ["#e6550d"] * len(best)
    fig, ax = plt.subplots(figsize=(max(6, 1.15 * len(labels)), 4.5))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("micro-F1 (= accuracy), final_data test")
    ax.set_title("Encoder vs decoder-only LLMs — directional TRC (best setting per model)")
    if enc:
        ax.axhline(enc[0]["micro_f1"], ls="--", lw=1, color="#2b8cbe", alpha=0.7)
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    rows = collect()
    if not rows:
        print("no result files found under outputs/")
        return
    (OUT / "results_table.md").write_text(to_markdown(rows))
    with open(OUT / "results.csv", "w", newline="") as f:
        cols = ["model", "method", "setting", "micro_f1", "micro_std", "macro_f1"] + \
               [f"src_{s}" for s in SOURCES]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    bar_figure(rows, OUT / "results_bar.png")
    print(to_markdown(rows))

    enc = [r for r in rows if r["method"] == "encoder"]
    llm = [r for r in rows if r["method"] != "encoder" and r["micro_f1"] is not None]
    if enc and llm:
        enc_f1 = enc[0]["micro_f1"]
        best_llm = max(llm, key=lambda x: x["micro_f1"])
        print(f"\nencoder micro-F1 = {enc_f1:.4f}  |  best LLM ({best_llm['model']} "
              f"{best_llm['method']} {best_llm['setting']}) micro-F1 = {best_llm['micro_f1']:.4f}")
        print("CLAIM HOLDS: encoder beats every LLM setting."
              if enc_f1 > best_llm["micro_f1"] else
              "CLAIM DOES NOT HOLD (an LLM matched/beat the encoder).")


if __name__ == "__main__":
    main()
