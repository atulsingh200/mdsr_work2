#!/usr/bin/env python3
"""Build deberta-v3-large focused CSV + README.

Extracts all results for the deberta-v3-large model across both the
bi-encoder (baseline_sweep) and cross-encoder (crossencoder_sweep) sweeps.

Outputs (in /mnt/localssd/sweep_analysis/):
  deberta_results.csv        — in-domain + OOD summary, one row per (encoder, dataset)
  deberta_milan_per_event.csv — Milan per-event detail for both encoders
  deberta_ajo_per_sample.csv  — AJO 30-sample per-sample detail for both encoders
  deberta_README.md           — human-readable summary of all the above

Run:
  python3 /mnt/localssd/sweep_analysis/build_deberta_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/localssd")
from aggregate_results import aggregate, DATASETS  # noqa: E402

BASE   = Path("/mnt/localssd")
OUT    = BASE / "sweep_analysis"
MODEL  = "deberta-v3-large"
SWEEPS = {
    "biencoder":    BASE / "baseline_sweep"     / "runs",
    "crossencoder": BASE / "crossencoder_sweep" / "runs",
}
ENCODERS = ["biencoder", "crossencoder"]


# ── helpers ────────────────────────────────────────────────────────────────── #
def fmt_order(order) -> str:
    return " > ".join(order) if order else ""

def fnum(v, nd: int = 6) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ""

def md(v, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

def pct(v) -> str:
    """Format as percentage string, e.g. 0.9667 -> '96.67%'."""
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"


# ── load aggregated data ────────────────────────────────────────────────────── #
print("Aggregating sweeps ...")
agg = {enc: aggregate(root) for enc, root in SWEEPS.items()}

def get(enc: str, ds: str):
    return (agg[enc][ds] or {}).get(MODEL)

OUT.mkdir(parents=True, exist_ok=True)


# ── CSV 1: summary (in-domain + OOD high-level) ────────────────────────────── #
with (OUT / "deberta_results.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "encoder", "dataset",
        "n_train", "n_test", "epochs",
        "best_val_acc", "best_val_auc",
        "test_acc", "test_auc", "test_f1",
        "milan_pairwise_acc", "milan_order_match", "milan_predicted_order",
        "ajo_n_correct", "ajo_n_samples", "ajo_perfect_order_rate",
    ])
    for enc in ENCODERS:
        for ds in DATASETS:
            e = get(enc, ds)
            if e is None:
                w.writerow([enc, ds] + [""] * 14 + ["missing"])
                continue
            milan = e.get("milan") or {}
            ajo   = e.get("ajo_ordering") or {}
            w.writerow([
                enc, ds,
                e.get("n_train", ""), e.get("n_test", ""), e.get("epochs", ""),
                fnum(e.get("best_val_acc")), fnum(e.get("best_val_auc")),
                fnum(e.get("test_acc")), fnum(e.get("test_auc")), fnum(e.get("test_f1")),
                fnum(milan.get("pairwise_acc")), milan.get("order_match", ""),
                fmt_order(milan.get("predicted_order")),
                ajo.get("n_correct", ""), ajo.get("n_samples", ""),
                fnum(ajo.get("perfect_order_rate")),
            ])

print(f"  Wrote deberta_results.csv")


# ── CSV 2: Milan per-event detail ──────────────────────────────────────────── #
with (OUT / "deberta_milan_per_event.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "encoder", "dataset",
        "event", "tournament_wins", "net_precedence",
        "predicted_rank", "true_rank",
    ])
    for enc in ENCODERS:
        for ds in DATASETS:
            e = get(enc, ds)
            milan = (e or {}).get("milan") or {}
            per_event = milan.get("per_event") or {}
            for ev in ["t1", "t2", "t3", "t4", "t5", "t6"]:
                d = per_event.get(ev, {})
                w.writerow([
                    enc, ds, ev,
                    d.get("tournament_wins", ""), d.get("net_precedence", ""),
                    d.get("predicted_rank", ""), d.get("true_rank", ""),
                ])

print(f"  Wrote deberta_milan_per_event.csv")


# ── CSV 3: AJO per-sample detail ───────────────────────────────────────────── #
with (OUT / "deberta_ajo_per_sample.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "encoder", "dataset",
        "sample_id", "ground_truth_order", "predicted_order", "correct",
    ])
    for enc in ENCODERS:
        for ds in DATASETS:
            e = get(enc, ds)
            ajo = (e or {}).get("ajo_ordering") or {}
            for s in ajo.get("per_sample") or []:
                w.writerow([
                    enc, ds,
                    s.get("id", ""),
                    fmt_order(s.get("ground_truth_order")),
                    fmt_order(s.get("predicted_order")),
                    s.get("correct", ""),
                ])

print(f"  Wrote deberta_ajo_per_sample.csv")


# ── README ─────────────────────────────────────────────────────────────────── #

# pre-compute per-encoder per-dataset dicts for clean template use
results = {enc: {ds: get(enc, ds) for ds in DATASETS} for enc in ENCODERS}

def milan_row(enc, ds):
    e = results[enc][ds]
    milan = (e or {}).get("milan") or {}
    pred  = fmt_order(milan.get("predicted_order"))
    gt    = fmt_order(milan.get("ground_truth_order"))
    acc   = milan.get("pairwise_acc")
    match = milan.get("order_match")
    nc    = milan.get("n_pairs_correct", "?")
    nt    = milan.get("n_pairs_total", "?")
    mark  = "✓" if match else "✗"
    return f"| `{ds}` | {pred} | {md(acc)} | {nc}/{nt} | {mark} |"

def ajo_row(enc, ds):
    e  = results[enc][ds]
    ajo = (e or {}).get("ajo_ordering") or {}
    rate = ajo.get("perfect_order_rate")
    nc   = ajo.get("n_correct", "?")
    ns   = ajo.get("n_samples", "?")
    return f"| `{ds}` | {nc}/{ns} | {pct(rate)} |"

def indomain_row(enc, ds):
    e = results[enc][ds]
    if not e:
        return f"| `{ds}` | — | — | — | — | — | — |"
    return (
        f"| `{ds}` | {e.get('n_train','?')} | {e.get('n_test','?')} "
        f"| {md(e.get('test_acc'))} | {md(e.get('test_auc'))} | {md(e.get('test_f1'))} "
        f"| {md(e.get('best_val_acc'))} |"
    )

def milan_event_table(enc, ds):
    e = results[enc][ds]
    milan = (e or {}).get("milan") or {}
    per_event = milan.get("per_event") or {}
    lines = [
        "| Event | Tournament wins | Net precedence | Predicted rank | True rank |",
        "|---|---|---|---|---|",
    ]
    for ev in ["t1", "t2", "t3", "t4", "t5", "t6"]:
        d = per_event.get(ev, {})
        lines.append(
            f"| {ev} | {d.get('tournament_wins','—')} | {d.get('net_precedence','—')} "
            f"| {d.get('predicted_rank','—')} | {d.get('true_rank','—')} |"
        )
    return "\n".join(lines)

def ajo_sample_table(enc, ds):
    e = results[enc][ds]
    ajo = (e or {}).get("ajo_ordering") or {}
    samples = ajo.get("per_sample") or []
    lines = [
        "| Sample | Ground truth | Predicted | Correct |",
        "|---|---|---|---|",
    ]
    for s in samples:
        tick = "✓" if s.get("correct") else "✗"
        lines.append(
            f"| {s.get('id','?')} | {fmt_order(s.get('ground_truth_order'))} "
            f"| {fmt_order(s.get('predicted_order'))} | {tick} |"
        )
    return "\n".join(lines)

GT_ORDER = "t1 > t2 > t3 > t4 > t5 > t6"

# Build long sections for each encoder
def encoder_section(enc):
    label  = "Bi-encoder (`DirectionalClassifier`)" if enc == "biencoder" else "Cross-encoder (`CrossEncoderClassifier`)"
    arch   = ("Two untied encoders; features = [A ; B ; A−B ; A⊙B] → MLP head."
              if enc == "biencoder" else
              "Single joint transformer with `[CLS] text_1 [SEP] text_2 [SEP]` → Linear head.")

    # in-domain table
    indomain_rows = "\n".join(indomain_row(enc, ds) for ds in DATASETS)

    # milan summary table
    milan_rows = "\n".join(milan_row(enc, ds) for ds in DATASETS)

    # ajo summary table
    ajo_rows = "\n".join(ajo_row(enc, ds) for ds in DATASETS)

    # per-dataset deep dives
    deep = []
    for ds in DATASETS:
        e = results[enc][ds]
        if not e:
            deep.append(f"#### `{ds}` — no data\n")
            continue
        milan = e.get("milan") or {}
        ajo   = e.get("ajo_ordering") or {}
        pred_order = fmt_order(milan.get("predicted_order"))
        match_str  = "exact match ✓" if milan.get("order_match") else "no exact match ✗"
        deep.append(f"""#### `{ds}`

**Milan** — predicted order: `{pred_order}` ({match_str}, pairwise acc {md(milan.get('pairwise_acc'))})

{milan_event_table(enc, ds)}

**AJO** — {ajo.get('n_correct','?')}/{ajo.get('n_samples','?')} correct ({pct(ajo.get('perfect_order_rate'))})

{ajo_sample_table(enc, ds)}
""")

    return f"""## {label}

{arch}

### In-domain test metrics

| Dataset | n_train | n_test | test_acc | test_auc | test_f1 | best_val_acc |
|---|---|---|---|---|---|---|
{indomain_rows}

### OOD — Milan 6-step workflow

Ground truth: **{GT_ORDER}**

| Dataset | Predicted order | Pairwise acc | Pairs correct | Exact match |
|---|---|---|---|---|
{milan_rows}

### OOD — AJO 30-sample ordering

| Dataset | Correct / 30 | Perfect-order rate |
|---|---|---|
{ajo_rows}

### Per-dataset detail

{"".join(deep)}"""

bi_section = encoder_section("biencoder")
ce_section = encoder_section("crossencoder")

# delta table
delta_lines = [
    "| Dataset | bi test_acc | ce test_acc | Δ acc | bi Milan acc | ce Milan acc | Δ Milan | bi AJO rate | ce AJO rate | Δ AJO |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for ds in DATASETS:
    bi = results["biencoder"][ds] or {}
    ce = results["crossencoder"][ds] or {}
    bi_acc  = bi.get("test_acc")
    ce_acc  = ce.get("test_acc")
    bi_mp   = (bi.get("milan") or {}).get("pairwise_acc")
    ce_mp   = (ce.get("milan") or {}).get("pairwise_acc")
    bi_ajo  = (bi.get("ajo_ordering") or {}).get("perfect_order_rate")
    ce_ajo  = (ce.get("ajo_ordering") or {}).get("perfect_order_rate")
    da      = (ce_acc - bi_acc)  if (bi_acc  is not None and ce_acc  is not None) else None
    dm      = (ce_mp  - bi_mp)   if (bi_mp   is not None and ce_mp   is not None) else None
    daj     = (ce_ajo - bi_ajo)  if (bi_ajo  is not None and ce_ajo  is not None) else None
    delta_lines.append(
        f"| `{ds}` | {md(bi_acc)} | {md(ce_acc)} | {md(da)} "
        f"| {md(bi_mp)} | {md(ce_mp)} | {md(dm)} "
        f"| {pct(bi_ajo)} | {pct(ce_ajo)} | {md(daj, 3)} |"
    )
delta_table = "\n".join(delta_lines)

readme = f"""# DeBERTa-v3-large Results — Bi-encoder vs Cross-encoder

Full breakdown of `deberta-v3-large` across all 7 datasets and both architectures.

- **7 datasets**, **2 architectures** (bi-encoder, cross-encoder)
- **In-domain** metrics: test accuracy, AUC, F1
- **OOD — Milan 6-step**: model ranks 6 events by tournament scoring; evaluated on
  pairwise accuracy (30 ordered pairs) and exact order match.
  Ground truth: **{GT_ORDER}**
- **OOD — AJO 30-sample**: 30 independent 3-step ordering instances;
  metric = perfect-order rate (fraction exactly correct).

## Head-to-head summary (deberta-v3-large)

{delta_table}

---

{bi_section}

---

{ce_section}

---

## CSV files

| File | Contents |
|---|---|
| [deberta_results.csv](deberta_results.csv) | In-domain + OOD summary, one row per (encoder, dataset) |
| [deberta_milan_per_event.csv](deberta_milan_per_event.csv) | Milan per-event tournament wins, net precedence, predicted rank |
| [deberta_ajo_per_sample.csv](deberta_ajo_per_sample.csv) | AJO 30 samples — ground truth vs predicted order, correct flag |

*Generated by [build_deberta_analysis.py](build_deberta_analysis.py).*
"""

(OUT / "deberta_README.md").write_text(readme)
print(f"  Wrote deberta_README.md")
print(f"Done — outputs in {OUT}")
