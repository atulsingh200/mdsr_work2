"""Full-dataset evaluation with precomputed, cached embeddings.

Unlike evaluate_inference.py (which caps candidate pools at 1000), this script:

  1. Encodes anchors and positives once per (model, dataset) and saves the
     resulting numpy arrays to  evaluation_6/emb_cache/<model>/<dataset>/.
     On subsequent runs the cached files are loaded instantly — no re-encoding.

  2. Ranks each anchor against ALL N positives in the test split
     (true same-split retrieval, exact O(N²) via batched GPU matmul).

  3. Reports exact metrics — no approximations:
       MRR, Recall@K, Hit@K, mean/median rank  — from the full ranking
       AUC  = mean((N - rank_i) / (N - 1))     — exact, no sampling
       P@1  = Recall@1                          — fraction ranked strictly 1st

Usage:
  .venv/bin/python evaluation_6/evaluate_full.py
  .venv/bin/python evaluation_6/evaluate_full.py --models all-MiniLM-L6-v2
  .venv/bin/python evaluation_6/evaluate_full.py --datasets aep_causal qrecc
  .venv/bin/python evaluation_6/evaluate_full.py --force-recompute   # re-encode
  .venv/bin/python evaluation_6/evaluate_full.py --rank-batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.retrievers import PretrainedRetriever            # noqa: E402
from evaluation_6.metrics_extra import full_retrieval_metrics    # noqa: E402


# ---------------------------------------------------------------------------
# Configuration  (mirrors evaluate_inference.py)
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "all-MiniLM-L6-v2": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "pooling": "mean",
        "max_seq_length": 256,
    },
    "bert-base-uncased": {
        "hf_id": "google-bert/bert-base-uncased",
        "pooling": "cls",
        "max_seq_length": 512,
    },
}

DATASETS: dict[str, Path] = {
    "aep_causal":   ROOT / "data_6/aep_causal/test.jsonl",
    "aep_followup": ROOT / "data_6/aep_followup/followup_pairs.jsonl",
    "followupqg":   ROOT / "data_6/followupqg/test.jsonl",
    "multiwoz_v24": ROOT / "data_6/multiwoz_v24/test.jsonl",
    "qrecc":        ROOT / "data_6/qrecc/test.jsonl",
    "workflow":     ROOT / "data_6/workflow/test_pairs.jsonl",
}

EMB_CACHE = ROOT / "evaluation_6/emb_cache"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = (row.get("anchor") or "").strip()
            p = (row.get("positive") or "").strip()
            if a and p:
                pairs.append((a, p))
    return pairs


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
def get_or_encode(
    model_key: str,
    dataset_name: str,
    anchors: list[str],
    positives: list[str],
    retriever: PretrainedRetriever,
    encode_batch_size: int,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (anchor_emb, positive_emb), loading from cache when available."""
    cache_dir = EMB_CACHE / model_key / dataset_name
    a_path = cache_dir / "anchors.npy"
    p_path = cache_dir / "positives.npy"

    if not force_recompute and a_path.exists() and p_path.exists():
        a_emb = np.load(a_path)
        p_emb = np.load(p_path)
        print(f"  [cache] loaded {a_emb.shape} from {cache_dir}")
        return a_emb, p_emb

    cache_dir.mkdir(parents=True, exist_ok=True)
    retriever.batch_size = encode_batch_size

    t0 = time.time()
    a_emb = retriever.encode_anchors(anchors)
    p_emb = retriever.encode_candidates(positives)
    enc_time = time.time() - t0

    np.save(a_path, a_emb)
    np.save(p_path, p_emb)
    print(f"  encoded {len(anchors) * 2} texts in {enc_time:.1f}s  "
          f"(dim={a_emb.shape[1]}) → saved to {cache_dir}")
    return a_emb, p_emb


# ---------------------------------------------------------------------------
# Per-(model, dataset) evaluation
# ---------------------------------------------------------------------------
def evaluate_model_on_dataset(
    model_key: str,
    model_cfg: dict,
    dataset_name: str,
    dataset_path: Path,
    k_values: list[int],
    retriever: PretrainedRetriever,
    encode_batch_size: int,
    rank_batch_size: int,
    device: str,
    force_recompute: bool,
) -> dict:
    print(f"\n[{model_key} × {dataset_name}]")

    pairs = load_pairs(dataset_path)
    if len(pairs) < 2:
        print(f"  SKIP: only {len(pairs)} valid pairs")
        return {"model": model_key, "dataset": dataset_name, "error": "not enough pairs"}

    anchors   = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    print(f"  {len(pairs):,} pairs")

    a_emb, p_emb = get_or_encode(
        model_key, dataset_name, anchors, positives,
        retriever, encode_batch_size, force_recompute,
    )

    t0 = time.time()
    metrics = full_retrieval_metrics(
        a_emb, p_emb,
        k_values=k_values,
        device=device,
        batch_size=rank_batch_size,
    )
    rank_time = time.time() - t0
    N = metrics["n_queries"]

    print(f"  ranked {N:,}×{N:,} in {rank_time:.1f}s  [{device}]")
    print(f"  AUC: {metrics['auc']:.4f}   P@1: {metrics['precision_at_1']:.4f}   "
          f"MRR: {metrics['mrr']:.4f}   Mean rank: {metrics['mean_rank']:.2f}   "
          f"Median rank: {metrics['median_rank']:.1f}")
    for k in k_values:
        print(f"  Recall@{k:<3}{metrics['recall_at_k'][k]:.4f}")

    return {
        "model":           model_key,
        "model_hf_id":     model_cfg["hf_id"],
        "dataset":         dataset_name,
        "n_queries":       N,
        "n_candidates":    N,
        "ranking_seconds": rank_time,
        "auc":             metrics["auc"],
        "precision_at_1":  metrics["precision_at_1"],
        "mrr":             metrics["mrr"],
        "mean_rank":       metrics["mean_rank"],
        "median_rank":     metrics["median_rank"],
        "recall_at_k":     metrics["recall_at_k"],
        "hit_at_k":        metrics["hit_at_k"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], k_values: list[int]) -> None:
    k_show = [k for k in [1, 5, 10] if k in k_values]
    col_w = 120 + len(k_show) * 8
    print("\n" + "=" * col_w)
    print("SUMMARY  (full N×N retrieval — no pool sampling, exact metrics)")
    print("=" * col_w)
    r_headers = "".join(f"{'R@'+str(k):>8}" for k in k_show)
    header = (
        f"{'model':<22} {'dataset':<14} {'N':>8} "
        f"{'AUC':>7} {'P@1':>7} {'MRR':>7}"
        f"{r_headers} {'mean_r':>8} {'med_r':>7}"
    )
    print(header)
    print("-" * col_w)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<22} {r['dataset']:<14}  ERROR: {r['error']}")
            continue
        r_vals = "".join(f"{r['recall_at_k'].get(k, float('nan')):>8.4f}" for k in k_show)
        print(
            f"{r['model']:<22} {r['dataset']:<14} {r['n_queries']:>8,d} "
            f"{r['auc']:>7.4f} {r['precision_at_1']:>7.4f} {r['mrr']:>7.4f}"
            f"{r_vals} {r['mean_rank']:>8.2f} {r['median_rank']:>7.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                    help="K values for Recall@K / Hit@K.")
    ap.add_argument("--encode-batch-size", type=int, default=128,
                    help="Encoder batch size (text → embeddings).")
    ap.add_argument("--rank-batch-size", type=int, default=512,
                    help="Query rows per GPU matmul batch during ranking.")
    ap.add_argument("--device", default=None, help="cuda / cpu / None=auto.")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Re-encode and overwrite cached embeddings.")
    ap.add_argument("--out", default=str(ROOT / "evaluation_6/results_full.json"))
    args = ap.parse_args()

    import torch
    k_values = sorted(set(args.k))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Models:            {args.models}")
    print(f"Datasets:          {args.datasets}")
    print(f"K values:          {k_values}")
    print(f"Device:            {device}")
    print(f"Encode batch size: {args.encode_batch_size}")
    print(f"Rank batch size:   {args.rank_batch_size}")
    print(f"Emb cache:         {EMB_CACHE}")
    print(f"Output:            {args.out}")

    results: list[dict] = []
    for model_key in args.models:
        model_cfg = MODELS[model_key]
        print(f"\n[load] {model_key} ({model_cfg['hf_id']}, {model_cfg['pooling']} pool)")
        retriever = PretrainedRetriever(
            model_name=model_cfg["hf_id"],
            pooling=model_cfg["pooling"],
            max_seq_length=model_cfg.get("max_seq_length", 512),
            batch_size=args.encode_batch_size,
        )
        for ds_name in args.datasets:
            try:
                row = evaluate_model_on_dataset(
                    model_key=model_key,
                    model_cfg=model_cfg,
                    dataset_name=ds_name,
                    dataset_path=DATASETS[ds_name],
                    k_values=k_values,
                    retriever=retriever,
                    encode_batch_size=args.encode_batch_size,
                    rank_batch_size=args.rank_batch_size,
                    device=device,
                    force_recompute=args.force_recompute,
                )
                results.append(row)
            except Exception as e:
                print(f"  ERROR ({type(e).__name__}): {e}")
                import traceback; traceback.print_exc()
                results.append({
                    "model":   model_key,
                    "dataset": ds_name,
                    "error":   f"{type(e).__name__}: {e}",
                })

    payload = {
        "config": {
            "models":            {k: MODELS[k] for k in args.models},
            "datasets":          {k: str(DATASETS[k]) for k in args.datasets},
            "k_values":          k_values,
            "encode_batch_size": args.encode_batch_size,
            "rank_batch_size":   args.rank_batch_size,
            "device":            device,
            "full_retrieval":    True,
        },
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] saved to {out_path}")

    print_summary(results, k_values)


if __name__ == "__main__":
    main()
