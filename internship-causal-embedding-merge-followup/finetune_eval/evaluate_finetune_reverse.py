"""Evaluate reverse-direction fine-tuned bi-encoder checkpoints.

For every test pair (A, B) we compute exactly 2 scores, mirroring the
training loss s(A,B) - s(B,A):

    forward : s(A,B) = enc_anchor(A) · enc_positive(B)  →  label 1
    reverse : s(B,A) = enc_anchor(B) · enc_positive(A)  →  label 0

Metrics:
  AUC     — area under ROC (forward score vs reverse score, binary)
  P@1     — fraction of pairs where forward > reverse  (ties count as 0.5)

Usage:
  uv run python finetune_eval/evaluate_finetune_reverse.py --dataset all
  uv run python finetune_eval/evaluate_finetune_reverse.py --dataset aep_causal
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
from finetune_eval.data import load_pairs                                    # noqa: E402
from finetune_eval.datasets import list_datasets, split_path                 # noqa: E402


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _auc_binary(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUC for N independent binary decisions (1 positive, 1 negative each)."""
    n = len(pos_scores)
    # fraction of pairs where pos > neg, + 0.5 * ties
    wins = float((pos_scores > neg_scores).sum())
    ties = float((pos_scores == neg_scores).sum())
    return (wins + 0.5 * ties) / n


def _precision_at_1(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """P@1 = fraction where forward > reverse (ties count as 0.5)."""
    wins = float((pos_scores > neg_scores).sum())
    ties = float((pos_scores == neg_scores).sum())
    return (wins + 0.5 * ties) / len(pos_scores)


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

    test_path = split_path(dataset, "test")
    if test_path is None or not test_path.exists():
        raise FileNotFoundError(f"no test file for {dataset}: {test_path}")
    pairs = load_pairs(test_path, cap=None)
    anchors  = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    print(f"  test pairs: {len(pairs)}")

    t0 = time.time()
    # Four embeddings mirroring the training loss exactly:
    #   s(A,B) = enc_anchor(A) · enc_positive(B)   forward  (label 1)
    #   s(B,A) = enc_anchor(B) · enc_positive(A)   reverse  (label 0)
    a_cause  = encode_texts(model, tokenizer, anchors,   "anchor",   cfg["max_seq_length"], batch_size, device)
    b_effect = encode_texts(model, tokenizer, positives, "positive", cfg["max_seq_length"], batch_size, device)
    b_cause  = encode_texts(model, tokenizer, positives, "anchor",   cfg["max_seq_length"], batch_size, device)
    a_effect = encode_texts(model, tokenizer, anchors,   "positive", cfg["max_seq_length"], batch_size, device)
    enc_time = time.time() - t0
    print(f"  encoded in {enc_time:.1f}s")

    # Per-pair dot products matching training objective
    fwd_scores = (a_cause * b_effect).sum(axis=1)   # s(A,B)  label=1
    rev_scores = (b_cause * a_effect).sum(axis=1)   # s(B,A)  label=0

    auc = _auc_binary(fwd_scores, rev_scores)
    p1  = _precision_at_1(fwd_scores, rev_scores)

    print(f"  AUC  s(A→B) > s(B→A): {auc:.4f}")
    print(f"  P@1  s(A→B) > s(B→A): {p1:.4f}")
    print(f"  mean s(A→B)={fwd_scores.mean():.4f}   mean s(B→A)={rev_scores.mean():.4f}")

    row = {
        "dataset":          dataset,
        "checkpoint":       str(ckpt_path),
        "backbone":         cfg["backbone"],
        "pooling":          cfg["pooling"],
        "n_test_pairs":     len(pairs),
        "encoding_seconds": enc_time,
        "auc":              float(auc),
        "precision_at_1":   float(p1),
        "mean_fwd_score":   float(fwd_scores.mean()),
        "mean_rev_score":   float(rev_scores.mean()),
        "eval_type":        "forward_vs_reverse_2candidates",
    }
    (ds_dir / "eval_reverse.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
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

    out_path = Path(args.out) if args.out else results_dir / "eval_reverse_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"eval_type": "forward_vs_reverse_2candidates"},
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] {out_path}")

    print("\n" + "=" * 60)
    print(f"{'dataset':<14} {'N':>7} {'AUC':>8} {'P@1':>8}")
    print("-" * 60)
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<14}  ERROR: {r['error']}")
            continue
        print(f"{r['dataset']:<14} {r['n_test_pairs']:>7d} {r['auc']:>8.4f} {r['precision_at_1']:>8.4f}")


if __name__ == "__main__":
    main()
