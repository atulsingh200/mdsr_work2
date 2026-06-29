"""Evaluate a trained checkpoint on its dataset's test split.

Loads a `BiEncoder` checkpoint produced by train_finetune.py, encodes the
test split's anchors and positives, and computes the same pair-level +
random-pool retrieval metrics as `evaluation_6/`:

  * AUC and Precision@1 (vs 4 random negatives)
  * MRR, Recall@K, Hit@K, mean / median rank (per-anchor pool of 1000)

Usage:
  .venv/bin/python finetune_eval/evaluate_finetune.py --dataset aep_causal
  .venv/bin/python finetune_eval/evaluate_finetune.py --dataset all
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

from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts        # noqa: E402

from evaluation_6.metrics_extra import (                                     # noqa: E402
    auc_precision_at_1_with_random_negatives,
    random_pool_retrieval_metrics,
)

from finetune_eval.data import load_pairs                                    # noqa: E402
from finetune_eval.datasets import list_datasets, split_path                 # noqa: E402


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_with_biencoder(
    model: BiEncoder,
    tokenizer,
    texts: list[str],
    which: str,                # "anchor" or "positive"
    max_length: int,
    batch_size: int,
    device: torch.device,
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
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.hidden_size), dtype=np.float32)


def evaluate_one(
    dataset: str,
    results_dir: Path,
    pool_size: int,
    n_negatives: int,
    k_values: list[int],
    batch_size: int,
    seed: int,
    results_suffix: str = "",
) -> dict:
    ds_dir = results_dir / (dataset + results_suffix)
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
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    print(f"  test pairs: {len(pairs)}  (file: {test_path.name})")

    t0 = time.time()
    a_emb = encode_with_biencoder(model, tokenizer, anchors,   "anchor",   cfg["max_seq_length"], batch_size, device)
    p_emb = encode_with_biencoder(model, tokenizer, positives, "positive", cfg["max_seq_length"], batch_size, device)
    enc_time = time.time() - t0
    print(f"  encoded {len(pairs) * 2} texts in {enc_time:.1f}s")

    pair_metrics = auc_precision_at_1_with_random_negatives(
        a_emb, p_emb, n_negatives=n_negatives, seed=seed,
    )
    retrieval_metrics = random_pool_retrieval_metrics(
        a_emb, p_emb,
        pool_size=pool_size, k_values=k_values, seed=seed,
    )

    print(f"  AUC: {pair_metrics['auc']:.4f}    P@1 (vs {n_negatives}): {pair_metrics['precision_at_1']:.4f}")
    print(f"  MRR: {retrieval_metrics['mrr']:.4f}    "
          f"Mean rank: {retrieval_metrics['mean_rank']:.2f}    "
          f"Median rank: {retrieval_metrics['median_rank']:.1f}")
    for k in k_values:
        print(f"  Recall@{k:<3}{retrieval_metrics['recall_at_k'][k]:.4f}"
              f"    Hit@{k:<3}{retrieval_metrics['hit_at_k'][k]:.4f}")

    row = {
        "dataset":               dataset,
        "checkpoint":            str(ckpt_path),
        "backbone":              cfg["backbone"],
        "pooling":               cfg["pooling"],
        "n_test_pairs":          len(pairs),
        "n_candidates_per_query": retrieval_metrics["n_candidates_per_query"],
        "encoding_seconds":      enc_time,
        "auc":                   pair_metrics["auc"],
        "precision_at_1":        pair_metrics["precision_at_1"],
        "n_negatives":           n_negatives,
        "mrr":                   retrieval_metrics["mrr"],
        "mean_rank":             retrieval_metrics["mean_rank"],
        "median_rank":           retrieval_metrics["median_rank"],
        "recall_at_k":           retrieval_metrics["recall_at_k"],
        "hit_at_k":              retrieval_metrics["hit_at_k"],
    }
    (ds_dir / "eval.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    ap.add_argument("--results-dir", default=str(ROOT / "finetune_eval/results"))
    ap.add_argument("--pool-size", type=int, default=1000,
                    help="Candidates per anchor for retrieval metrics.")
    ap.add_argument("--n-negatives", type=int, default=4, help="Random negs for AUC / P@1.")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-suffix", default="",
                    help="Suffix appended to each dataset subdir name when loading checkpoints, "
                         "e.g. '_no_inbatch'. Matches the suffix used in train_finetune.py.")
    ap.add_argument("--out", default=None,
                    help="Aggregated results JSON. Default: <results-dir>/eval_summary.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    targets = list_datasets() if args.dataset == "all" else [args.dataset]
    k_values = sorted(set(args.k))

    rows: list[dict] = []
    for ds in targets:
        try:
            rows.append(evaluate_one(
                dataset=ds,
                results_dir=results_dir,
                pool_size=args.pool_size,
                n_negatives=args.n_negatives,
                k_values=k_values,
                batch_size=args.batch_size,
                seed=args.seed,
                results_suffix=args.results_suffix,
            ))
        except FileNotFoundError as e:
            print(f"\n[{ds}] SKIP: {e}")
            rows.append({"dataset": ds, "error": str(e)})

    out_path = Path(args.out) if args.out else results_dir / "eval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "pool_size":   args.pool_size,
            "n_negatives": args.n_negatives,
            "k_values":    k_values,
            "seed":        args.seed,
        },
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] {out_path}")

    # Tabular summary
    print("\n" + "=" * 110)
    print(f"{'dataset':<14} {'N':>7} {'AUC':>7} {'P@1':>7} {'MRR':>7} "
          f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'mean':>7} {'median':>7}")
    print("-" * 110)
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<14}  ERROR: {r['error']}")
            continue
        print(f"{r['dataset']:<14} {r['n_test_pairs']:>7d} "
              f"{r['auc']:>7.4f} {r['precision_at_1']:>7.4f} {r['mrr']:>7.4f} "
              f"{r['recall_at_k'][1]:>7.4f} {r['recall_at_k'][5]:>7.4f} "
              f"{r['recall_at_k'][10]:>7.4f} {r['mean_rank']:>7.2f} {r['median_rank']:>7.1f}")


if __name__ == "__main__":
    main()
