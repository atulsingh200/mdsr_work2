"""Blend two trained cross-encoders and evaluate on the (untouched) test set.

Reads the per-row predictions written by train_ce.py for each run:
    <run>/val_predictions.jsonl
    <run>/test_predictions.jsonl

Both runs were trained on the SAME (identical, untouched) val/test splits in the
same row order, so predictions align by index. Blend:

    p_blend = alpha * p_semantic + (1 - alpha) * p_indomain

Reports:
  - each CE alone (acc / auc / f1) on test
  - blend @ alpha=0.5 (simple mean)
  - blend @ best-val-alpha (alpha swept on val, applied to test)
  - per-tier-pair accuracy for the best blend
  - dual-encoder baseline (best_mlp_bge_small) for context

Invoke (from repo root):
  python3 -m src.classifier.crossencoder.ensemble_eval \
      --semantic-run runs/crossencoder/ce_semantic \
      --indomain-run runs/crossencoder/ce_indomain \
      --out runs/crossencoder/ensemble_metrics.json

# Hard-negative test set (run inference on a custom JSONL instead of the
# pre-computed test_predictions.jsonl; val predictions still come from the
# run dirs for the alpha sweep):
#
#   python3 -m src.classifier.crossencoder.ensemble_eval \
#       --semantic-run runs/crossencoder/ce_semantic \
#       --indomain-run runs/crossencoder/ce_indomain \
#       --hard-neg-test data/aep_causal_classification_hard_neg_test/directional_test.jsonl \
#       --out runs/crossencoder/ensemble_hard_neg_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_preds(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        auc = float(roc_auc_score(labels, probs))
        f1 = float(f1_score(labels, preds))
    except Exception:
        auc = f1 = float("nan")
    return {"acc": round(acc, 6), "auc": round(auc, 6),
            "f1": round(f1, 6), "n": int(len(labels))}


def per_tier_pair(rows: list[dict], probs: np.ndarray) -> dict:
    pair = defaultdict(lambda: {"n": 0, "correct": 0})
    for r, p in zip(rows, probs):
        key = f"T{min(r['tier_1'], r['tier_2'])}-T{max(r['tier_1'], r['tier_2'])}"
        pair[key]["n"] += 1
        pair[key]["correct"] += int((p > 0.5) == (r["label"] == 1))
    return {k: {"n": v["n"], "acc": round(v["correct"] / v["n"], 6)}
            for k, v in sorted(pair.items(),
                               key=lambda x: [int(t[1:]) for t in x[0].split("-")])}


def infer_probs(ckpt_path: Path, jsonl_path: Path, batch_size: int = 64) -> tuple[list[dict], np.ndarray]:
    """Load a CE checkpoint and run inference on an arbitrary JSONL file.

    Returns (rows, probs) in file order. Used when --hard-neg-test is given
    so we score a custom dataset instead of the pre-computed test_predictions.jsonl.
    """
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from src.classifier.crossencoder.data import CECollate, CEPairDataset
        from src.classifier.crossencoder.model import CrossEncoderClassifier
    else:
        from .data import CECollate, CEPairDataset
        from .model import CrossEncoderClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = CrossEncoderClassifier(cfg["base_model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = CEPairDataset(jsonl_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=CECollate(tokenizer, cfg["max_seq_len"]),
                    pin_memory=device.type == "cuda")

    use_bf16 = device.type == "cuda"
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    all_probs = []
    with torch.no_grad():
        for batch in dl:
            enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            with ctx():
                logits = model(enc)
            all_probs.append(torch.sigmoid(logits).float().cpu().numpy())

    probs = np.concatenate(all_probs)
    # Attach prob back to rows so align_positional key-check works the same way
    rows = []
    for i, r in enumerate(ds.rows):
        rows.append({**r, "prob": float(probs[i])})
    return rows, probs


def _row_key(r: dict) -> tuple:
    """Identity of a pair (handles duplicate pairs only via position, not here)."""
    return (r["url_1"], r["url_2"], r["tier_1"], r["tier_2"], r["label"])


def align_positional(rows_a: list[dict], rows_b: list[dict], what: str):
    """Align two prediction lists POSITIONALLY over their common prefix.

    Both runs write predictions in dataset order. The original rows (test, and
    the first part of val) sit in the same positions in both runs, so position i
    in run A and run B refer to the same pair — this correctly keeps duplicate
    pairs (unlike a key-join, which would collapse them).

    The val sets then diverge (each run appended DIFFERENT mined hard negs); we
    stop at the first position whose keys disagree. For test (identical files)
    every position matches, so the full set is used.
    """
    n = min(len(rows_a), len(rows_b))
    rows, pa, pb, ys = [], [], [], []
    for i in range(n):
        a, b = rows_a[i], rows_b[i]
        if _row_key(a) != _row_key(b):
            break  # reached the run-specific mined rows; stop at the shared prefix
        rows.append(a)
        pa.append(a["prob"])
        pb.append(b["prob"])
        ys.append(a["label"])
    if not rows:
        raise AssertionError(f"[{what}] no aligned rows between the two runs")
    print(f"[{what}] aligned {len(rows)} shared rows "
          f"(sem={len(rows_a)}, ind={len(rows_b)})")
    return rows, np.array(pa), np.array(pb), np.array(ys).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-run", required=True)
    ap.add_argument("--indomain-run", required=True)
    ap.add_argument("--out", default="runs/crossencoder/ensemble_metrics.json")
    ap.add_argument("--baseline-metrics",
                    default="runs/classifier/best_mlp_bge_small/test_metrics.json")
    ap.add_argument(
        "--hard-neg-test",
        default=None,
        metavar="JSONL",
        help="Path to a custom test JSONL (e.g. the hard-negative test set). "
             "When set, runs inference via each run's best.pt on this file "
             "instead of loading the pre-computed test_predictions.jsonl. "
             "Val predictions still come from the run dirs for the alpha sweep.",
    )
    ap.add_argument("--eval-batch-size", type=int, default=64)
    args = ap.parse_args()

    sem, ind = Path(args.semantic_run), Path(args.indomain_run)

    if args.hard_neg_test:
        hn_path = Path(args.hard_neg_test)
        print(f"[ensemble] --hard-neg-test mode: running inference on {hn_path}")
        print(f"[ensemble]   loading ce_semantic  ({sem / 'best.pt'})")
        sem_test_rows, p_sem_test_raw = infer_probs(sem / "best.pt", hn_path, args.eval_batch_size)
        print(f"[ensemble]   loading ce_indomain  ({ind / 'best.pt'})")
        ind_test_rows, p_ind_test_raw = infer_probs(ind / "best.pt", hn_path, args.eval_batch_size)
        # Both ran on the identical file in the same order — no alignment needed
        assert len(sem_test_rows) == len(ind_test_rows)
        test_rows = sem_test_rows
        p_sem_test = p_sem_test_raw
        p_ind_test = p_ind_test_raw
        y_test = np.array([r["label"] for r in test_rows]).astype(int)
        print(f"[ensemble]   hard-neg test rows: {len(test_rows)}")
    else:
        sem_test = load_preds(sem / "test_predictions.jsonl")
        ind_test = load_preds(ind / "test_predictions.jsonl")

    sem_val = load_preds(sem / "val_predictions.jsonl")
    ind_val = load_preds(ind / "val_predictions.jsonl")

    # Align positionally over the shared prefix. Test sets are identical -> all
    # 6004 rows. Val sets share the original val rows up front, then diverge into
    # run-specific mined hard negs -> we keep the shared original-val rows, which
    # is exactly the right (test-distribution) set for the alpha sweep.
    if not args.hard_neg_test:
        test_rows, p_sem_test, p_ind_test, y_test = align_positional(sem_test, ind_test, "test")
    _val_rows, p_sem_val, p_ind_val, y_val = align_positional(sem_val, ind_val, "val")

    # ---- individual CEs ----
    m_sem = metrics(p_sem_test, y_test)
    m_ind = metrics(p_ind_test, y_test)

    # ---- mean blend (alpha=0.5) ----
    p_mean = 0.5 * p_sem_test + 0.5 * p_ind_test
    m_mean = metrics(p_mean, y_test)

    # ---- sweep alpha on VAL, apply best to TEST ----
    alphas = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    val_accs = []
    for a in alphas:
        pv = a * p_sem_val + (1 - a) * p_ind_val
        val_accs.append(((pv > 0.5).astype(int) == y_val).mean())
    best_i = int(np.argmax(val_accs))
    best_alpha = float(alphas[best_i])
    best_val_acc = float(val_accs[best_i])
    p_best = best_alpha * p_sem_test + (1 - best_alpha) * p_ind_test
    m_best = metrics(p_best, y_test)

    # ---- baseline (dual-encoder) for context ----
    baseline = None
    bpath = Path(args.baseline_metrics)
    if bpath.exists():
        b = json.loads(bpath.read_text())
        baseline = {"acc": b.get("acc"), "auc": b.get("auc"), "f1": b.get("f1")}

    report = {
        "semantic_run": str(sem), "indomain_run": str(ind),
        "n_test": int(len(y_test)), "n_val": int(len(y_val)),
        "ce_semantic_alone": m_sem,
        "ce_indomain_alone": m_ind,
        "blend_mean_0.5": m_mean,
        "blend_best_alpha": {
            "alpha_on_semantic": best_alpha,
            "val_acc_at_alpha": round(best_val_acc, 6),
            **m_best,
        },
        "alpha_sweep_val": {str(a): round(float(v), 6)
                            for a, v in zip(alphas, val_accs)},
        "dual_encoder_baseline": baseline,
        "best_blend_per_tier_pair": per_tier_pair(test_rows, p_best),
    }

    print("\n==================== ENSEMBLE TEST RESULTS ====================")
    def line(name, m):
        print(f"  {name:24s} acc={m['acc']:.4f}  auc={m['auc']:.4f}  f1={m['f1']:.4f}")
    line("CE-semantic alone", m_sem)
    line("CE-indomain alone", m_ind)
    line("blend mean (0.5)", m_mean)
    line(f"blend best a={best_alpha:.2f}", m_best)
    if baseline:
        print(f"  {'dual-encoder baseline':24s} acc={baseline['acc']:.4f}  "
              f"auc={baseline['auc']:.4f}  f1={baseline['f1']:.4f}")
    print(f"  (best alpha {best_alpha:.2f} chosen on val, val_acc={best_val_acc:.4f})")
    print("===============================================================\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[ensemble] wrote {args.out}")


if __name__ == "__main__":
    main()
