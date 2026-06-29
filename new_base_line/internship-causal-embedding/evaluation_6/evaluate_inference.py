"""Inference-only evaluation of off-the-shelf encoders on data_6 test splits.

Evaluates *every* anchor in the test split, but bounds the candidate pool
to a fixed `--candidate-pool-size` (default 1000) of random distractors per
anchor (gold + N-1 random other positives). This gives O(N · pool_size) work
rather than O(N²), so even workflow's 260k-row test split is tractable.

Per (model, dataset):
  * Pair-level: AUC, Precision@1 (vs N random negatives, default 4).
  * Retrieval:  MRR, Recall@K, Hit@K, mean / median rank
                (random candidate pool, pool_size per anchor).

Models (defaults — no fine-tuning, encoder used as-is):
  * all-MiniLM-L6-v2  — sentence-transformers/all-MiniLM-L6-v2  (mean pool)
  * bert-base-uncased — google-bert/bert-base-uncased           (cls pool)

Datasets (test split each, from data_6/):
  aep_causal, aep_followup, followupqg, multiwoz_v24, qrecc, workflow

Usage:
  .venv/bin/python evaluation_6/evaluate_inference.py
  .venv/bin/python evaluation_6/evaluate_inference.py --candidate-pool-size 500
  .venv/bin/python evaluation_6/evaluate_inference.py \\
        --models all-MiniLM-L6-v2 --datasets aep_causal qrecc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make src/ and the project root importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.retrievers import PretrainedRetriever                    # noqa: E402

from evaluation_6.metrics_extra import (                                  # noqa: E402
    auc_precision_at_1_with_random_negatives,
    random_pool_retrieval_metrics,
)


# ---------------------------------------------------------------------------
# Configuration
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
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "pooling": "cls",
        "max_seq_length": 512,
    },
    "e5-base-v2": {
        "hf_id": "intfloat/e5-base-v2",
        "pooling": "mean",
        "max_seq_length": 512,
        # E5 was trained with role prefixes; quality drops markedly without them.
        "anchor_prefix":   "query: ",
        "positive_prefix": "passage: ",
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Read (anchor, positive) pairs from a JSONL file (uses the full file)."""
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
# Per-(model, dataset) eval
# ---------------------------------------------------------------------------
def evaluate_model_on_dataset(
    model_key: str,
    model_cfg: dict,
    dataset_name: str,
    dataset_path: Path,
    candidate_pool_size: int,
    n_negatives: int,
    k_values: list[int],
    seed: int,
    retriever: PretrainedRetriever,
) -> dict:
    print(f"\n[{model_key} × {dataset_name}]")

    pairs = load_pairs(dataset_path)
    if len(pairs) < 2:
        print(f"  SKIP: only {len(pairs)} valid pairs in {dataset_path}")
        return {"model": model_key, "dataset": dataset_name, "error": "not enough pairs"}

    anchor_prefix   = model_cfg.get("anchor_prefix",   "")
    positive_prefix = model_cfg.get("positive_prefix", "")
    anchors = [anchor_prefix + a for a, _ in pairs]
    positives = [positive_prefix + p for _, p in pairs]
    if anchor_prefix or positive_prefix:
        print(f"  prefixes: anchor='{anchor_prefix}'  positive='{positive_prefix}'")
    print(f"  loaded {len(pairs)} pairs (file: {dataset_path.name})")

    t0 = time.time()
    a_emb = retriever.encode_anchors(anchors)
    p_emb = retriever.encode_candidates(positives)
    enc_time = time.time() - t0
    print(f"  encoded {len(pairs) * 2} texts in {enc_time:.1f}s (dim={a_emb.shape[1]})")

    pair_metrics = auc_precision_at_1_with_random_negatives(
        a_emb, p_emb, n_negatives=n_negatives, seed=seed,
    )

    t1 = time.time()
    retrieval_metrics = random_pool_retrieval_metrics(
        a_emb, p_emb,
        pool_size=candidate_pool_size,
        k_values=k_values,
        seed=seed,
    )
    ret_time = time.time() - t1
    print(f"  ranked vs pool of {retrieval_metrics['n_candidates_per_query']} in {ret_time:.1f}s")

    print(f"  AUC: {pair_metrics['auc']:.4f}    "
          f"P@1 (vs {n_negatives} negs): {pair_metrics['precision_at_1']:.4f}")
    print(f"  MRR: {retrieval_metrics['mrr']:.4f}    "
          f"Mean rank: {retrieval_metrics['mean_rank']:.2f}    "
          f"Median rank: {retrieval_metrics['median_rank']:.1f}")
    for k in k_values:
        print(f"  Recall@{k:<3}{retrieval_metrics['recall_at_k'][k]:.4f}"
              f"    Hit@{k:<3}{retrieval_metrics['hit_at_k'][k]:.4f}")

    return {
        "model":              model_key,
        "model_hf_id":        model_cfg["hf_id"],
        "dataset":            dataset_name,
        "n_queries":          retrieval_metrics["n_queries"],
        "n_candidates_per_query": retrieval_metrics["n_candidates_per_query"],
        "encoding_seconds":   enc_time,
        "ranking_seconds":    ret_time,
        "auc":                pair_metrics["auc"],
        "precision_at_1":     pair_metrics["precision_at_1"],
        "n_negatives":        n_negatives,
        "mrr":                retrieval_metrics["mrr"],
        "mean_rank":          retrieval_metrics["mean_rank"],
        "median_rank":        retrieval_metrics["median_rank"],
        "recall_at_k":        retrieval_metrics["recall_at_k"],
        "hit_at_k":           retrieval_metrics["hit_at_k"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], k_values: list[int]) -> None:
    print("\n" + "=" * 120)
    print("SUMMARY  (full test set per dataset, random candidate pool per anchor)")
    print("=" * 120)
    header = (
        f"{'model':<22} {'dataset':<14} {'N':>7} {'pool':>5} "
        f"{'AUC':>7} {'P@1':>7} {'MRR':>7} "
        f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'mean':>7} {'median':>7}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<22} {r['dataset']:<14}  ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:<22} {r['dataset']:<14} {r['n_queries']:>7d} "
            f"{r['n_candidates_per_query']:>5d} "
            f"{r['auc']:>7.4f} {r['precision_at_1']:>7.4f} {r['mrr']:>7.4f} "
            f"{r['recall_at_k'][1]:>7.4f} {r['recall_at_k'][5]:>7.4f} "
            f"{r['recall_at_k'][10]:>7.4f} {r['mean_rank']:>7.2f} {r['median_rank']:>7.1f}"
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
                    choices=list(MODELS.keys()),
                    help="Which models to evaluate.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()),
                    help="Which datasets to evaluate on.")
    ap.add_argument("--candidate-pool-size", type=int, default=1000,
                    help="Random candidates per anchor (gold + pool_size-1 distractors). "
                         "Capped at the dataset size.")
    ap.add_argument("--n-negatives", type=int, default=4,
                    help="Random negatives per anchor for AUC / P@1.")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                    help="K values for Recall@K / Hit@K.")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Encoder batch size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "evaluation_6/results.json"),
                    help="Output JSON path.")
    args = ap.parse_args()

    k_values = sorted(set(args.k))

    print(f"Models:              {args.models}")
    print(f"Datasets:            {args.datasets}")
    print(f"Candidate pool size: {args.candidate_pool_size}")
    print(f"AUC/P@1 negatives:   {args.n_negatives}")
    print(f"K values:            {k_values}")
    print(f"Output:              {args.out}")

    results: list[dict] = []
    for model_key in args.models:
        model_cfg = MODELS[model_key]
        print(f"\n[load] {model_key} ({model_cfg['hf_id']}, {model_cfg['pooling']} pool)")
        retriever = PretrainedRetriever(
            model_name=model_cfg["hf_id"],
            pooling=model_cfg["pooling"],
            max_seq_length=model_cfg.get("max_seq_length", 512),
            batch_size=args.batch_size,
        )
        for ds_name in args.datasets:
            try:
                row = evaluate_model_on_dataset(
                    model_key=model_key,
                    model_cfg=model_cfg,
                    dataset_name=ds_name,
                    dataset_path=DATASETS[ds_name],
                    candidate_pool_size=args.candidate_pool_size,
                    n_negatives=args.n_negatives,
                    k_values=k_values,
                    seed=args.seed,
                    retriever=retriever,
                )
                results.append(row)
            except Exception as e:  # keep going even if one combo fails
                print(f"  ERROR ({type(e).__name__}): {e}")
                results.append({
                    "model": model_key,
                    "dataset": ds_name,
                    "error": f"{type(e).__name__}: {e}",
                })

    payload = {
        "config": {
            "models":              {k: MODELS[k] for k in args.models},
            "datasets":            {k: str(DATASETS[k]) for k in args.datasets},
            "candidate_pool_size": args.candidate_pool_size,
            "n_negatives":         args.n_negatives,
            "k_values":            k_values,
            "batch_size":          args.batch_size,
            "seed":                args.seed,
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
