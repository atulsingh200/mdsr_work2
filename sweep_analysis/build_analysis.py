#!/usr/bin/env python3
"""Build comparison CSVs + README from the bi-encoder (baseline_sweep) and
cross-encoder (crossencoder_sweep) result trees.

Reuses aggregate_results.aggregate() to parse run_summary.json + ood_results.json
for each sweep root, and reads test_metrics.json directly for full-precision
per-tier-pair breakdowns.

Cross-encoder OOD data (Milan + AJO) is now fully included — all 42 runs were
evaluated on both OOD benchmarks.

All outputs land in /mnt/localssd/sweep_analysis/.

Run:
  python3 /mnt/localssd/sweep_analysis/build_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/localssd")
from aggregate_results import aggregate, DATASETS, MODELS  # noqa: E402

BASE = Path("/mnt/localssd")
SWEEPS = {
    "biencoder": BASE / "baseline_sweep" / "runs",
    "crossencoder": BASE / "crossencoder_sweep" / "runs",
}
ENCODERS = ["biencoder", "crossencoder"]
OUT = BASE / "sweep_analysis"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def fmt_order(order) -> str:
    """['t1','t2','t5'] -> 't1 > t2 > t5'."""
    return " > ".join(order) if order else ""


def fnum(v, nd: int = 6) -> str:
    """Format a float to nd places; '' for None/non-numeric."""
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ""


def md(v, nd: int = 4) -> str:
    """Float for a markdown cell; em-dash for None."""
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def load_test_metrics(root: Path, ds: str, model: str):
    p = root / ds / model / "test_metrics.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# load both sweeps
# --------------------------------------------------------------------------- #
print("Aggregating sweeps ...")
agg = {name: aggregate(root) for name, root in SWEEPS.items()}
tm = {
    name: {
        ds: {m: load_test_metrics(root, ds, m) for m in MODELS} for ds in DATASETS
    }
    for name, root in SWEEPS.items()
}

OUT.mkdir(parents=True, exist_ok=True)


def entry(enc: str, ds: str, m: str):
    return agg[enc][ds][m]


def iter_entries(enc: str):
    for ds in DATASETS:
        for m in MODELS:
            e = entry(enc, ds, m)
            if e is not None:
                yield ds, m, e


# --------------------------------------------------------------------------- #
# CSV 1 — test accuracy (both sweeps)
# --------------------------------------------------------------------------- #
with (OUT / "test_accuracy_all.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        ["encoder", "dataset", "model", "base_model", "n_train", "n_test", "epochs",
         "best_val_acc", "best_val_auc", "test_acc", "test_auc", "test_f1", "status"]
    )
    for enc in ENCODERS:
        for ds in DATASETS:
            for m in MODELS:
                e = entry(enc, ds, m)
                if e is None:
                    w.writerow([enc, ds, m, "", "", "", "", "", "", "", "", "", "missing"])
                    continue
                w.writerow([
                    enc, ds, m, e.get("base_model", ""),
                    e.get("n_train", ""), e.get("n_test", ""), e.get("epochs", ""),
                    fnum(e.get("best_val_acc")), fnum(e.get("best_val_auc")),
                    fnum(e.get("test_acc")), fnum(e.get("test_auc")), fnum(e.get("test_f1")),
                    "ok",
                ])

# --------------------------------------------------------------------------- #
# CSV 2 — OOD 30-sample summary (both sweeps, both encoders now evaluated)
# --------------------------------------------------------------------------- #
with (OUT / "ood_30sample_all.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        ["encoder", "dataset", "model",
         "milan_n_pairs_total", "milan_n_pairs_correct", "milan_pairwise_acc", "milan_order_match",
         "ajo_n_samples", "ajo_n_correct", "ajo_perfect_order_rate", "status"]
    )
    for enc in ENCODERS:
        for ds in DATASETS:
            for m in MODELS:
                e = entry(enc, ds, m)
                if e is None:
                    w.writerow([enc, ds, m, "", "", "", "", "", "", "", "missing"])
                    continue
                milan = e.get("milan")
                ajo = e.get("ajo_ordering")
                if milan is None and ajo is None:
                    w.writerow([enc, ds, m, "", "", "", "", "", "", "", "not_evaluated"])
                    continue
                milan = milan or {}
                ajo = ajo or {}
                w.writerow([
                    enc, ds, m,
                    milan.get("n_pairs_total", ""), milan.get("n_pairs_correct", ""),
                    fnum(milan.get("pairwise_acc")), milan.get("order_match", ""),
                    ajo.get("n_samples", ""), ajo.get("n_correct", ""),
                    fnum(ajo.get("perfect_order_rate")), "ok",
                ])

# --------------------------------------------------------------------------- #
# CSV 3 — Milan 6-step workflow predicted order (both encoders)
# --------------------------------------------------------------------------- #
with (OUT / "ood_6step_workflow_milan.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        ["encoder", "dataset", "model", "n_events", "ground_truth_order", "predicted_order",
         "order_match", "n_pairs_correct", "n_pairs_total", "pairwise_acc"]
    )
    for enc in ENCODERS:
        for ds, m, e in iter_entries(enc):
            milan = e.get("milan")
            if not milan:
                continue
            w.writerow([
                enc, ds, m, milan.get("n_events", ""),
                fmt_order(milan.get("ground_truth_order")), fmt_order(milan.get("predicted_order")),
                milan.get("order_match", ""), milan.get("n_pairs_correct", ""),
                milan.get("n_pairs_total", ""), fnum(milan.get("pairwise_acc")),
            ])

# --------------------------------------------------------------------------- #
# CSV 4 — bi-encoder vs cross-encoder test-accuracy comparison
# --------------------------------------------------------------------------- #
with (OUT / "comparison_biencoder_vs_crossencoder.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        ["dataset", "model", "bi_test_acc", "ce_test_acc", "delta_acc_ce_minus_bi",
         "bi_test_auc", "ce_test_auc", "bi_test_f1", "ce_test_f1"]
    )
    for ds in DATASETS:
        for m in MODELS:
            bi = entry("biencoder", ds, m) or {}
            ce = entry("crossencoder", ds, m) or {}
            bi_acc, ce_acc = bi.get("test_acc"), ce.get("test_acc")
            delta = ce_acc - bi_acc if (bi_acc is not None and ce_acc is not None) else None
            w.writerow([
                ds, m, fnum(bi_acc), fnum(ce_acc), fnum(delta),
                fnum(bi.get("test_auc")), fnum(ce.get("test_auc")),
                fnum(bi.get("test_f1")), fnum(ce.get("test_f1")),
            ])

# --------------------------------------------------------------------------- #
# CSV 5 — Milan per-event detail (both encoders)
# --------------------------------------------------------------------------- #
with (OUT / "ood_milan_per_event.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["encoder", "dataset", "model", "event", "tournament_wins", "net_precedence",
                "predicted_rank", "true_rank"])
    for enc in ENCODERS:
        for ds, m, e in iter_entries(enc):
            milan = e.get("milan")
            if not milan:
                continue
            for ev, d in (milan.get("per_event") or {}).items():
                w.writerow([enc, ds, m, ev, d.get("tournament_wins", ""), d.get("net_precedence", ""),
                            d.get("predicted_rank", ""), d.get("true_rank", "")])

# --------------------------------------------------------------------------- #
# CSV 6 — AJO per-sample detail (both encoders)
# --------------------------------------------------------------------------- #
with (OUT / "ood_ajo_per_sample.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["encoder", "dataset", "model", "sample_id", "ground_truth_order",
                "predicted_order", "correct"])
    for enc in ENCODERS:
        for ds, m, e in iter_entries(enc):
            ajo = e.get("ajo_ordering")
            if not ajo:
                continue
            for s in (ajo.get("per_sample") or []):
                w.writerow([enc, ds, m, s.get("id", ""), fmt_order(s.get("ground_truth_order")),
                            fmt_order(s.get("predicted_order")), s.get("correct", "")])

# --------------------------------------------------------------------------- #
# CSV 7 — per-tier-pair in-domain breakdown (both sweeps)
# --------------------------------------------------------------------------- #
with (OUT / "per_tier_pair_all.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["encoder", "dataset", "model", "tier_pair", "n", "acc"])
    for enc in ENCODERS:
        for ds in DATASETS:
            for m in MODELS:
                t = tm[enc][ds][m]
                if not t:
                    continue
                for pair, d in (t.get("per_tier_pair") or {}).items():
                    w.writerow([enc, ds, m, pair, d.get("n", ""), fnum(d.get("acc"))])

# --------------------------------------------------------------------------- #
# CSV 8 — OOD comparison: bi-encoder vs cross-encoder (Milan + AJO)
# --------------------------------------------------------------------------- #
with (OUT / "comparison_ood_biencoder_vs_crossencoder.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "dataset", "model",
        "bi_milan_pairwise_acc", "ce_milan_pairwise_acc", "delta_milan_ce_minus_bi",
        "bi_milan_order_match", "ce_milan_order_match",
        "bi_ajo_perfect_order_rate", "ce_ajo_perfect_order_rate", "delta_ajo_ce_minus_bi",
    ])
    for ds in DATASETS:
        for m in MODELS:
            bi = entry("biencoder", ds, m) or {}
            ce = entry("crossencoder", ds, m) or {}
            bi_milan = bi.get("milan") or {}
            ce_milan = ce.get("milan") or {}
            bi_ajo = bi.get("ajo_ordering") or {}
            ce_ajo = ce.get("ajo_ordering") or {}
            bi_mp = bi_milan.get("pairwise_acc")
            ce_mp = ce_milan.get("pairwise_acc")
            bi_ap = bi_ajo.get("perfect_order_rate")
            ce_ap = ce_ajo.get("perfect_order_rate")
            delta_milan = (ce_mp - bi_mp) if (bi_mp is not None and ce_mp is not None) else None
            delta_ajo = (ce_ap - bi_ap) if (bi_ap is not None and ce_ap is not None) else None
            w.writerow([
                ds, m,
                fnum(bi_mp), fnum(ce_mp), fnum(delta_milan),
                bi_milan.get("order_match", ""), ce_milan.get("order_match", ""),
                fnum(bi_ap), fnum(ce_ap), fnum(delta_ajo),
            ])

print(f"Wrote 8 CSVs to {OUT}")


# --------------------------------------------------------------------------- #
# README.md
# --------------------------------------------------------------------------- #
def matrix_table(value_fn) -> str:
    """Markdown matrix: rows=datasets, cols=models, cell=value_fn(ds, m)."""
    hdr = "| Dataset | " + " | ".join(MODELS) + " |"
    sep = "|" + "|".join(["---"] * (len(MODELS) + 1)) + "|"
    lines = [hdr, sep]
    for ds in DATASETS:
        cells = [value_fn(ds, m) for m in MODELS]
        lines.append(f"| `{ds}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cell_acc(enc):
    def fn(ds, m):
        e = entry(enc, ds, m)
        return md(e.get("test_acc")) if e else "—"
    return fn


def cell_auc(enc):
    def fn(ds, m):
        e = entry(enc, ds, m)
        return md(e.get("test_auc")) if e else "—"
    return fn


def cell_f1(enc):
    def fn(ds, m):
        e = entry(enc, ds, m)
        return md(e.get("test_f1")) if e else "—"
    return fn


def cell_milan(enc):
    def fn(ds, m):
        e = entry(enc, ds, m)
        milan = (e or {}).get("milan")
        if not milan:
            return "—"
        mark = " ✓" if milan.get("order_match") else ""
        return f"{md(milan.get('pairwise_acc'))}{mark}"
    return fn


def cell_ajo(enc):
    def fn(ds, m):
        e = entry(enc, ds, m)
        ajo = (e or {}).get("ajo_ordering")
        if not ajo:
            return "—"
        rate = ajo.get("perfect_order_rate")
        nc, ns = ajo.get("n_correct"), ajo.get("n_samples")
        return f"{md(rate, 3)} ({nc}/{ns})"
    return fn


# headline numbers (computed, not hardcoded)
def best_by(enc, key_fn):
    cands = [(ds, m, key_fn(e)) for ds, m, e in iter_entries(enc) if key_fn(e) is not None]
    return max(cands, key=lambda x: x[2]) if cands else None


bi_best = best_by("biencoder", lambda e: e.get("test_acc"))
ce_best = best_by("crossencoder", lambda e: e.get("test_acc"))
bi_ajo_best = best_by("biencoder", lambda e: (e.get("ajo_ordering") or {}).get("perfect_order_rate"))
ce_ajo_best = best_by("crossencoder", lambda e: (e.get("ajo_ordering") or {}).get("perfect_order_rate"))
bi_milan_best = best_by("biencoder", lambda e: (e.get("milan") or {}).get("pairwise_acc"))
ce_milan_best = best_by("crossencoder", lambda e: (e.get("milan") or {}).get("pairwise_acc"))

n_bi = sum(1 for _ in iter_entries("biencoder"))
n_ce = sum(1 for _ in iter_entries("crossencoder"))

# Milan ground truth (constant across runs); grab from first available
milan_gt = ""
for _ds, _m, _e in iter_entries("biencoder"):
    if _e.get("milan"):
        milan_gt = fmt_order(_e["milan"].get("ground_truth_order"))
        break

# Milan predicted-order long listing (both encoders)
def milan_rows_for(enc):
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            e = entry(enc, ds, m)
            milan = (e or {}).get("milan")
            if not milan:
                continue
            mark = "✓" if milan.get("order_match") else "✗"
            rows.append(
                f"| `{ds}` | `{m}` | {fmt_order(milan.get('predicted_order'))} | "
                f"{md(milan.get('pairwise_acc'))} | {mark} |"
            )
    return rows

bi_milan_rows = milan_rows_for("biencoder")
ce_milan_rows = milan_rows_for("crossencoder")

readme = f"""# Sweep Analysis — Bi-Encoder vs Cross-Encoder

Comparison of two training sweeps over **7 datasets × 6 models** for the
document-pair ordering task, plus out-of-distribution (OOD) ordering evaluation
on both architectures.

- **Bi-encoder** (`/mnt/localssd/baseline_sweep/`) — `DirectionalClassifier`
  (2 untied encoders). **{n_bi}/42 runs complete**, each with in-domain test
  metrics and OOD eval (Milan 6-step ordering + AJO 30-sample ordering).
- **Cross-encoder** (`/mnt/localssd/crossencoder_sweep/`) — `CrossEncoderClassifier`
  (single joint transformer, `[CLS] text_1 [SEP] text_2 [SEP]`). **{n_ce}/42 runs
  complete**. OOD eval was run for all 42 cross-encoder runs.

## Headline Numbers

### In-domain test accuracy
- Best **bi-encoder**: `{bi_best[0]}` / `{bi_best[1]}` = **{bi_best[2]:.4f}**
- Best **cross-encoder**: `{ce_best[0]}` / `{ce_best[1]}` = **{ce_best[2]:.4f}**

### OOD — AJO 30-sample perfect-order rate
- Best **bi-encoder**: `{bi_ajo_best[0]}` / `{bi_ajo_best[1]}` = **{bi_ajo_best[2]:.3f}**
- Best **cross-encoder**: `{ce_ajo_best[0]}` / `{ce_ajo_best[1]}` = **{ce_ajo_best[2]:.3f}**

### OOD — Milan 6-step pairwise accuracy
- Best **bi-encoder**: `{bi_milan_best[0]}` / `{bi_milan_best[1]}` = **{bi_milan_best[2]:.4f}**
- Best **cross-encoder**: `{ce_milan_best[0]}` / `{ce_milan_best[1]}` = **{ce_milan_best[2]:.4f}**

## Metrics Glossary

- **test_acc / test_auc / test_f1** — in-domain test-set pairwise-ordering metrics.
- **Milan 6-step workflow** — a single fixed 6-event chain; the model scores all
  ordered pairs, events are ranked by tournament wins, and the result is scored by
  `pairwise_acc` (fraction of the 30 ordered pairs correct) and `order_match`
  (whether the full predicted order equals ground truth). Ground truth: **{milan_gt}**.
- **AJO 30-sample ordering** — 30 small ordering instances; `perfect_order_rate`
  is the fraction whose full predicted order is exactly correct.
- **Bi-encoder** — `DirectionalClassifier`: two separate encoders produce
  embeddings `A`, `B`; the classifier uses `[A; B; A−B; A⊙B]` as input features.
- **Cross-encoder** — `CrossEncoderClassifier`: a single transformer sees
  `[CLS] text_1 [SEP] text_2 [SEP]` jointly; the `[CLS]` token feeds a linear head.

---

## Table A — In-domain test accuracy: Bi-encoder

{matrix_table(cell_acc("biencoder"))}

## Table B — In-domain test accuracy: Cross-encoder

{matrix_table(cell_acc("crossencoder"))}

## Table C — In-domain test AUC: Bi-encoder

{matrix_table(cell_auc("biencoder"))}

## Table D — In-domain test AUC: Cross-encoder

{matrix_table(cell_auc("crossencoder"))}

## Table E — In-domain test F1: Bi-encoder

{matrix_table(cell_f1("biencoder"))}

## Table F — In-domain test F1: Cross-encoder

{matrix_table(cell_f1("crossencoder"))}

## Table G — Bi vs Cross (best model per dataset, by test_acc)

| Dataset | Best bi-encoder | bi acc | Best cross-encoder | ce acc | Δ (ce − bi) |
|---|---|---|---|---|---|
"""

for ds in DATASETS:
    bi_c = [(m, entry("biencoder", ds, m)["test_acc"]) for m in MODELS
            if entry("biencoder", ds, m) and entry("biencoder", ds, m).get("test_acc") is not None]
    ce_c = [(m, entry("crossencoder", ds, m)["test_acc"]) for m in MODELS
            if entry("crossencoder", ds, m) and entry("crossencoder", ds, m).get("test_acc") is not None]
    bi_top = max(bi_c, key=lambda x: x[1]) if bi_c else (None, None)
    ce_top = max(ce_c, key=lambda x: x[1]) if ce_c else (None, None)
    delta = (ce_top[1] - bi_top[1]) if (bi_top[1] is not None and ce_top[1] is not None) else None
    readme += (
        f"| `{ds}` | {('`'+bi_top[0]+'`') if bi_top[0] else '—'} | {md(bi_top[1])} | "
        f"{('`'+ce_top[0]+'`') if ce_top[0] else '—'} | {md(ce_top[1])} | {md(delta)} |\n"
    )

readme += f"""
*Full per-cell comparison in [comparison_biencoder_vs_crossencoder.csv](comparison_biencoder_vs_crossencoder.csv).*

---

## Table H — OOD Milan 6-step workflow: Bi-encoder

Pairwise accuracy over 30 ordered pairs; **✓** = full order exactly matched.

{matrix_table(cell_milan("biencoder"))}

### Milan predicted order per model — Bi-encoder

Ground-truth order: **{milan_gt}**

| Dataset | Model | Predicted order | Pairwise acc | Exact match |
|---|---|---|---|---|
""" + "\n".join(bi_milan_rows) + f"""

---

## Table I — OOD Milan 6-step workflow: Cross-encoder

Pairwise accuracy over 30 ordered pairs; **✓** = full order exactly matched.

{matrix_table(cell_milan("crossencoder"))}

### Milan predicted order per model — Cross-encoder

Ground-truth order: **{milan_gt}**

| Dataset | Model | Predicted order | Pairwise acc | Exact match |
|---|---|---|---|---|
""" + "\n".join(ce_milan_rows) + f"""

---

## Table J — OOD AJO 30-sample ordering: Bi-encoder

`perfect_order_rate` (correct / 30).

{matrix_table(cell_ajo("biencoder"))}

---

## Table K — OOD AJO 30-sample ordering: Cross-encoder

`perfect_order_rate` (correct / 30).

{matrix_table(cell_ajo("crossencoder"))}

---

## Table L — OOD comparison: Bi vs Cross (best per dataset)

| Dataset | bi Milan acc | ce Milan acc | Δ Milan | bi AJO rate | ce AJO rate | Δ AJO |
|---|---|---|---|---|---|---|
"""

for ds in DATASETS:
    def _best_milan(enc):
        cands = [(m, (entry(enc, ds, m) or {}).get("milan", {}).get("pairwise_acc"))
                 for m in MODELS]
        cands = [(m, v) for m, v in cands if v is not None]
        return max(cands, key=lambda x: x[1]) if cands else (None, None)

    def _best_ajo(enc):
        cands = [(m, ((entry(enc, ds, m) or {}).get("ajo_ordering") or {}).get("perfect_order_rate"))
                 for m in MODELS]
        cands = [(m, v) for m, v in cands if v is not None]
        return max(cands, key=lambda x: x[1]) if cands else (None, None)

    bi_m = _best_milan("biencoder")[1]
    ce_m = _best_milan("crossencoder")[1]
    bi_a = _best_ajo("biencoder")[1]
    ce_a = _best_ajo("crossencoder")[1]
    dm = (ce_m - bi_m) if (bi_m is not None and ce_m is not None) else None
    da = (ce_a - bi_a) if (bi_a is not None and ce_a is not None) else None
    readme += f"| `{ds}` | {md(bi_m)} | {md(ce_m)} | {md(dm)} | {md(bi_a, 3)} | {md(ce_a, 3)} | {md(da, 3)} |\n"

readme += f"""
*Full per-cell OOD comparison in [comparison_ood_biencoder_vs_crossencoder.csv](comparison_ood_biencoder_vs_crossencoder.csv).*

---

## CSV Files

| File | Contents |
|---|---|
| [test_accuracy_all.csv](test_accuracy_all.csv) | In-domain test metrics (acc, auc, f1), every run, both sweeps |
| [ood_30sample_all.csv](ood_30sample_all.csv) | OOD Milan + AJO 30-sample summary, both encoders |
| [ood_6step_workflow_milan.csv](ood_6step_workflow_milan.csv) | Milan 6-step predicted order per model, both encoders |
| [comparison_biencoder_vs_crossencoder.csv](comparison_biencoder_vs_crossencoder.csv) | Bi vs cross in-domain test acc/auc/f1 per (dataset, model) |
| [comparison_ood_biencoder_vs_crossencoder.csv](comparison_ood_biencoder_vs_crossencoder.csv) | Bi vs cross OOD Milan + AJO per (dataset, model) |
| [ood_milan_per_event.csv](ood_milan_per_event.csv) | Milan per-event ranks + tournament wins, both encoders |
| [ood_ajo_per_sample.csv](ood_ajo_per_sample.csv) | AJO per-sample predicted vs ground-truth order, both encoders |
| [per_tier_pair_all.csv](per_tier_pair_all.csv) | In-domain per-tier-pair accuracy, both sweeps |

*Generated by [build_analysis.py](build_analysis.py).*
"""

(OUT / "README.md").write_text(readme)
print(f"Wrote README.md to {OUT}")
print(f"Runs found: bi-encoder {n_bi}/42, cross-encoder {n_ce}/42")
