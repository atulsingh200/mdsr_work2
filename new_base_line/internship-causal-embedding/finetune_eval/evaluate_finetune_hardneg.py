"""Evaluate fine-tuned bi-encoder checkpoints against SEMANTIC hard negatives.

Companion to evaluate_finetune.py. The difference:

  evaluate_finetune.py            → AUC / P@1 use 4 RANDOM negatives per anchor.
  evaluate_finetune_hardneg.py    → AUC / P@1 use 4 SEMANTIC hard negatives per
                                    anchor — the same kind of negative the model
                                    was trained against, pre-mined offline on
                                    the test split via frozen MiniLM kNN.

For each anchor i, the candidate set per pair-metric calculation is exactly:
   { positive_i (gold) } ∪ { the 4 nearest other positives in MiniLM space }
NO random negatives, NO retrieval pool. P@1 must beat all 4 hard negs.

Requires that mine_hard_negatives.py was run with --split test first so each
results/<ds>/ contains test_pairs.jsonl + test_hard_negatives.npy.

Usage:
  .venv/bin/python finetune_eval/evaluate_finetune_hardneg.py --dataset all
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

from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts            # noqa: E402

from evaluation_6.metrics_extra import (                                         # noqa: E402
    auc_precision_at_1_with_semantic_hard_negatives,
)

from finetune_eval.data import load_pairs                                        # noqa: E402
from finetune_eval.datasets import list_datasets                                 # noqa: E402


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_with_biencoder(
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
        emb = (model.encode_anchor(ids, mask) if which == "anchor"
               else model.encode_positive(ids, mask))
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.hidden_size), dtype=np.float32)


def evaluate_one(dataset: str, results_dir: Path, batch_size: int, results_suffix: str = "") -> dict:
    ds_dir = results_dir / (dataset + results_suffix)
    # Mining artifacts (test_pairs, hard_negatives) always live in the base dataset dir.
    mining_dir = results_dir / dataset
    ckpt_path = ds_dir / "checkpoint_best.pt"
    pairs_path = mining_dir / "test_pairs.jsonl"
    hn_path = mining_dir / "test_hard_negatives.npy"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    if not pairs_path.exists() or not hn_path.exists():
        raise FileNotFoundError(
            f"missing mined test artifacts. Run:\n"
            f"  .venv/bin/python finetune_eval/mine_hard_negatives.py "
            f"--dataset {dataset} --split test --k 4"
        )

    print(f"\n[{dataset}] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = auto_device()

    tokenizer = get_tokenizer(cfg["backbone"])
    model = BiEncoder(
        model_name=cfg["backbone"], pooling_strategy=cfg["pooling"],
        anchor_prefix="", positive_prefix="",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  backbone: {cfg['backbone']}  pooling: {cfg['pooling']}  device: {device}")

    pairs = load_pairs(pairs_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    hard_neg_idx = np.load(hn_path)
    print(f"  test pairs: {len(pairs)}   hard-neg idx shape: {hard_neg_idx.shape}")
    if hard_neg_idx.shape[0] != len(pairs):
        raise ValueError(
            f"hard_neg_idx rows {hard_neg_idx.shape[0]} != n_pairs {len(pairs)} "
            f"(re-mine with --split test for this dataset)"
        )

    t0 = time.time()
    a_emb = encode_with_biencoder(model, tokenizer, anchors,   "anchor",   cfg["max_seq_length"], batch_size, device)
    p_emb = encode_with_biencoder(model, tokenizer, positives, "positive", cfg["max_seq_length"], batch_size, device)
    enc_time = time.time() - t0
    print(f"  encoded {len(pairs) * 2} texts in {enc_time:.1f}s")

    metrics = auc_precision_at_1_with_semantic_hard_negatives(a_emb, p_emb, hard_neg_idx)
    print(f"  AUC: {metrics['auc']:.4f}    P@1 (vs {metrics['n_negatives_per_anchor']} semantic hard negs): "
          f"{metrics['precision_at_1']:.4f}")

    row = {
        "dataset":               dataset,
        "checkpoint":            str(ckpt_path),
        "backbone":              cfg["backbone"],
        "pooling":               cfg["pooling"],
        "n_test_pairs":          len(pairs),
        "n_negatives":           int(metrics["n_negatives_per_anchor"]),
        "encoding_seconds":      enc_time,
        "auc":                   metrics["auc"],
        "precision_at_1":        metrics["precision_at_1"],
        "eval_type":             "semantic_hard_negatives",
        "miner":                 "sentence-transformers/all-MiniLM-L6-v2",
    }
    (ds_dir / "eval_hardneg.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    ap.add_argument("--results-dir", default=str(ROOT / "finetune_eval/results"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--results-suffix", default="",
                    help="Suffix appended to each dataset subdir name when loading checkpoints, "
                         "e.g. '_no_inbatch'. Mining artifacts are still read from the base dir.")
    ap.add_argument("--out", default=None,
                    help="Aggregated results JSON. Default: <results-dir>/eval_hardneg_summary.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    # aep_followup has no train split → no checkpoint
    targets = [d for d in list_datasets() if d != "aep_followup"] if args.dataset == "all" else [args.dataset]

    rows: list[dict] = []
    for ds in targets:
        try:
            rows.append(evaluate_one(ds, results_dir, args.batch_size, results_suffix=args.results_suffix))
        except FileNotFoundError as e:
            print(f"\n[{ds}] SKIP: {e}")
            rows.append({"dataset": ds, "error": str(e)})

    out_path = Path(args.out) if args.out else results_dir / "eval_hardneg_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "eval_type": "semantic_hard_negatives_only",
            "n_negatives_per_anchor": 4,
            "miner": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] {out_path}")

    print("\n" + "=" * 70)
    print(f"{'dataset':<14} {'N':>7} {'AUC':>8} {'P@1':>8}")
    print("-" * 70)
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<14}  ERROR: {r['error']}")
            continue
        print(f"{r['dataset']:<14} {r['n_test_pairs']:>7d} {r['auc']:>8.4f} {r['precision_at_1']:>8.4f}")


if __name__ == "__main__":
    main()
