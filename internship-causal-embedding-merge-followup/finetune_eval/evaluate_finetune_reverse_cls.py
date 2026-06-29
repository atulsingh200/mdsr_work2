"""Threshold-based classification eval for a reverse-direction bi-encoder.

Unlike evaluate_finetune_reverse.py (which compares forward vs reverse
direction as a 2-candidate ranking), this script treats the task as binary
classification on the *raw directional* files:

  each test row has  text_1, text_2, label ∈ {0, 1}
  the model produces a single directed score
      s = enc_anchor(text_1) · enc_positive(text_2)
  we choose a threshold t (tuned on the val split to maximize accuracy)
  prediction = 1 if s >= t else 0

Reports accuracy / precision / recall / F1 / AUC on the test split, plus the
chosen threshold. The test file is never modified.

Usage:
  uv run python finetune_eval/evaluate_finetune_reverse_cls.py --dataset aep_causal_cls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts         # noqa: E402
from finetune_eval.datasets import (                                         # noqa: E402
    list_datasets, raw_labeled_val_path, split_path,
)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_labeled_pairs(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read directional rows (text_1, text_2, label). Falls back to
    anchor/positive field names if text_1/text_2 are absent."""
    t1: list[str] = []
    t2: list[str] = []
    labels: list[int] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            a = (d.get("text_1") or d.get("anchor") or "").strip()
            b = (d.get("text_2") or d.get("positive") or "").strip()
            lab = d.get("label")
            if not a or not b or lab is None:
                continue
            t1.append(a)
            t2.append(b)
            labels.append(int(lab))
    return t1, t2, np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def encode_texts(
    model: BiEncoder, tokenizer, texts: list[str], which: str,
    max_length: int, batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        ids, mask = tokenize_texts(tokenizer, chunk, max_length)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        emb = model.encode_anchor(ids, mask) if which == "anchor" else model.encode_positive(ids, mask)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def directed_scores(
    model, tokenizer, t1: list[str], t2: list[str],
    max_length: int, batch_size: int, device: torch.device,
) -> np.ndarray:
    """s = enc_anchor(text_1) · enc_positive(text_2)  for each row."""
    cause  = encode_texts(model, tokenizer, t1, "anchor",   max_length, batch_size, device)
    effect = encode_texts(model, tokenizer, t2, "positive", max_length, batch_size, device)
    return (cause * effect).sum(axis=1)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Pick the threshold maximizing accuracy. Candidate thresholds are the
    midpoints between sorted unique scores (plus the extremes)."""
    order = np.argsort(scores)
    s_sorted = scores[order]
    uniq = np.unique(s_sorted)
    cands = [uniq[0] - 1e-6]
    cands += list((uniq[:-1] + uniq[1:]) / 2.0)
    cands.append(uniq[-1] + 1e-6)
    best_t, best_acc = cands[0], -1.0
    for t in cands:
        acc = float(((scores >= t).astype(int) == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def classification_metrics(scores: np.ndarray, labels: np.ndarray, t: float) -> dict:
    pred = (scores >= t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = len(labels)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": float(t),
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank (Mann–Whitney U) formula, ties = 0.5."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    n_pos = len(pos)
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * len(neg)))


def evaluate_one(dataset: str, results_dir: Path, batch_size: int) -> dict:
    ds_dir = results_dir / dataset
    ckpt_path = ds_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

    print(f"\n[{dataset}] loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    device = auto_device()
    tokenizer = get_tokenizer(cfg["backbone"])
    model = BiEncoder(
        model_name=cfg["backbone"],
        pooling_strategy=cfg["pooling"],
        anchor_prefix="",
        positive_prefix="",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  backbone: {cfg['backbone']}  pooling: {cfg['pooling']}  device: {device}")
    max_len = cfg["max_seq_length"]

    # ---- tune threshold on val ------------------------------------------
    # Prefer the raw labeled val (text_1/text_2 + 0/1 label); fall back to the
    # regular val split (load_labeled_pairs also reads anchor/positive).
    val_path = raw_labeled_val_path(dataset) or split_path(dataset, "val")
    tuned_t = None
    val_tuned_acc = None
    if val_path is not None and val_path.exists():
        v1, v2, vy = load_labeled_pairs(val_path)
        print(f"  val rows: {len(vy)}  (pos={int((vy == 1).sum())}, neg={int((vy == 0).sum())})")
        t0 = time.time()
        v_scores = directed_scores(model, tokenizer, v1, v2, max_len, batch_size, device)
        tuned_t, val_tuned_acc = best_threshold(v_scores, vy)
        print(f"  tuned threshold={tuned_t:.4f}  (val acc={val_tuned_acc:.4f}, {time.time() - t0:.1f}s)")
    else:
        print("  no val file — will fall back to threshold tuned on test (optimistic)")

    # ---- evaluate on test -----------------------------------------------
    test_path = split_path(dataset, "test")
    if test_path is None or not test_path.exists():
        raise FileNotFoundError(f"no test file for {dataset}: {test_path}")
    e1, e2, ey = load_labeled_pairs(test_path)
    print(f"  test rows: {len(ey)}  (pos={int((ey == 1).sum())}, neg={int((ey == 0).sum())})")

    t0 = time.time()
    e_scores = directed_scores(model, tokenizer, e1, e2, max_len, batch_size, device)
    enc_time = time.time() - t0
    print(f"  encoded test in {enc_time:.1f}s")

    # threshold from val (preferred) or self-tuned on test (oracle, reported separately)
    oracle_t, oracle_acc = best_threshold(e_scores, ey)
    if tuned_t is None:
        tuned_t = oracle_t

    test_metrics = classification_metrics(e_scores, ey, tuned_t)
    test_metrics_oracle = classification_metrics(e_scores, ey, oracle_t)
    auc = auc_score(e_scores, ey)

    print(f"  AUC: {auc:.4f}")
    print(f"  [val-tuned t={tuned_t:.4f}]   acc={test_metrics['accuracy']:.4f}  "
          f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}  F1={test_metrics['f1']:.4f}")
    print(f"  [oracle  t={oracle_t:.4f}]   acc={test_metrics_oracle['accuracy']:.4f}  "
          f"(upper bound, threshold fit on test)")

    row = {
        "dataset":            dataset,
        "checkpoint":         str(ckpt_path),
        "backbone":           cfg["backbone"],
        "pooling":            cfg["pooling"],
        "n_test":             len(ey),
        "n_test_pos":         int((ey == 1).sum()),
        "n_test_neg":         int((ey == 0).sum()),
        "encoding_seconds":   enc_time,
        "auc":                float(auc),
        "val_threshold":      float(tuned_t),
        "val_tuned_accuracy": (None if val_tuned_acc is None else float(val_tuned_acc)),
        "test":               test_metrics,
        "test_oracle":        test_metrics_oracle,
        "eval_type":          "directed_score_threshold_classification",
    }
    (ds_dir / "eval_reverse_cls.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal_cls", help="Dataset name or 'all'.")
    ap.add_argument("--results-dir", default=str(ROOT / "finetune_eval/results_reverse"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    targets = list_datasets() if args.dataset == "all" else [args.dataset]

    rows: list[dict] = []
    for ds in targets:
        try:
            rows.append(evaluate_one(ds, results_dir, args.batch_size))
        except FileNotFoundError as e:
            print(f"\n[{ds}] SKIP: {e}")
            rows.append({"dataset": ds, "error": str(e)})

    out_path = Path(args.out) if args.out else results_dir / "eval_reverse_cls_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": {"eval_type": "directed_score_threshold_classification"}, "results": rows},
        indent=2,
    ))
    print(f"\n[done] {out_path}")

    print("\n" + "=" * 72)
    print(f"{'dataset':<16} {'N':>7} {'AUC':>7} {'acc':>7} {'F1':>7} {'thr':>8}")
    print("-" * 72)
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<16}  ERROR: {r['error']}")
            continue
        print(f"{r['dataset']:<16} {r['n_test']:>7d} {r['auc']:>7.4f} "
              f"{r['test']['accuracy']:>7.4f} {r['test']['f1']:>7.4f} {r['val_threshold']:>8.4f}")


if __name__ == "__main__":
    main()
