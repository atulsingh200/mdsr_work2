"""BM25 hard-negative evaluation of off-the-shelf encoders on data_6 test splits.

Same setup as evaluation_6/evaluate_inference.py but uses BM25-mined hard
negatives (instead of random negatives) for the AUC / Precision@1 metrics.
Retrieval metrics (MRR, Recall@K) use the same random candidate pool as before.

For each anchor the top-k BM25-scoring positives (excluding the anchor's own
gold) serve as negatives — these are lexically similar to the anchor, making
the discrimination task strictly harder than random negatives.

For large datasets (> --bm25-corpus-size) the BM25 index is built from a
random sample of the corpus to keep evaluation tractable (default 10 000).
For datasets larger than --max-bm25-queries (default 10 000) only that many
anchors are used for the BM25 AUC/P@1 step; retrieval metrics still run over
the full split.

Models:
  * all-MiniLM-L6-v2  — sentence-transformers/all-MiniLM-L6-v2 (mean pool)
  * bert-base-uncased — google-bert/bert-base-uncased            (cls pool)

Datasets (test split, from data_6/):
  aep_causal, aep_followup, followupqg, multiwoz_v24, qrecc, workflow

Usage:
  .venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py
  .venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py --bm25-corpus-size 5000
  .venv/bin/python evaluation_6/bm25negative/evaluate_bm25.py \\
        --models all-MiniLM-L6-v2 --datasets aep_causal qrecc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation.retrievers import PretrainedRetriever           # noqa: E402
from evaluation_6.metrics_extra import random_pool_retrieval_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration  (mirrors evaluation_6/evaluate_inference.py)
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
# BM25 hard-negative mining
# ---------------------------------------------------------------------------
def mine_bm25_negatives(
    anchor_texts: list[str],
    positive_texts: list[str],
    n_negatives: int = 4,
    max_corpus_size: int | None = 10_000,
    max_queries: int | None = 10_000,
    oversample: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine BM25 hard negatives.

    Returns:
        neg_idx   : (Q, n_negatives) indices into positive_texts / positive_emb
        query_idx : (Q,) which anchors were used (all if Q == N)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 required. Install with:  uv pip install '.[hard-negatives]'"
        ) from exc

    n = len(anchor_texts)
    rng = np.random.default_rng(seed)
    tokenize = lambda s: s.lower().split()

    # Which anchors to query (subsample for very large datasets)
    if max_queries and n > max_queries:
        query_idx = np.sort(rng.choice(n, size=max_queries, replace=False))
    else:
        query_idx = np.arange(n)

    # Which positives form the BM25 corpus
    if max_corpus_size and n > max_corpus_size:
        corpus_idx = np.sort(rng.choice(n, size=max_corpus_size, replace=False))
    else:
        corpus_idx = np.arange(n)

    corpus_texts = [positive_texts[i] for i in corpus_idx]
    print(
        f"  building BM25 index: corpus={len(corpus_texts)}, queries={len(query_idx)}",
        flush=True,
    )
    t0 = time.time()
    bm25 = BM25Okapi([tokenize(t) for t in corpus_texts])
    print(f"  BM25 index built in {time.time()-t0:.1f}s", flush=True)

    fetch_n = max(n_negatives * oversample, n_negatives + 1)
    Q = len(query_idx)
    neg_idx = np.empty((Q, n_negatives), dtype=np.int64)

    t1 = time.time()
    for qi, i in enumerate(query_idx):
        scores = bm25.get_scores(tokenize(anchor_texts[i]))
        top_local = np.argsort(scores)[-fetch_n:][::-1]  # indices into corpus

        negs: list[int] = []
        for j in top_local:
            global_idx = int(corpus_idx[j])
            if positive_texts[global_idx] == positive_texts[i]:
                continue
            negs.append(global_idx)
            if len(negs) >= n_negatives:
                break

        # Fallback to random if BM25 doesn't produce enough distinct negatives
        attempts = 0
        while len(negs) < n_negatives and attempts < n_negatives * 20:
            r = int(rng.integers(0, n))
            if positive_texts[r] != positive_texts[i] and r not in negs:
                negs.append(r)
            attempts += 1

        neg_idx[qi] = negs[:n_negatives]

        if (qi + 1) % 2000 == 0:
            elapsed = time.time() - t1
            rate = (qi + 1) / elapsed
            remaining = (Q - qi - 1) / rate
            print(
                f"  mined {qi+1}/{Q} anchors  ({rate:.0f}/s, ~{remaining:.0f}s left)",
                flush=True,
            )

    print(f"  BM25 mining done in {time.time()-t1:.1f}s", flush=True)
    return neg_idx, query_idx


# ---------------------------------------------------------------------------
# AUC + P@1 with pre-mined hard negatives
# ---------------------------------------------------------------------------
def auc_precision_at_1_bm25(
    anchor_emb: np.ndarray,
    gold_positive_emb: np.ndarray,
    neg_pool_emb: np.ndarray,
    neg_idx: np.ndarray,
) -> dict:
    """Compute AUC and P@1 using pre-mined BM25 negatives.

    Args:
        anchor_emb:       (Q, D) embeddings for the queried anchors.
        gold_positive_emb:(Q, D) aligned gold positives for the queried anchors.
        neg_pool_emb:     (N, D) full positive pool — neg_idx references this.
        neg_idx:          (Q, k) global indices into neg_pool_emb.
    """
    n, n_negatives = neg_idx.shape
    pos_sims = np.sum(anchor_emb * gold_positive_emb, axis=1)  # (Q,) — aligned
    neg_pos  = neg_pool_emb[neg_idx]                            # (Q, k, D)
    neg_sims = np.einsum("id,ind->in", anchor_emb, neg_pos) # (Q, k)

    p_at_1 = float(np.mean(pos_sims > neg_sims.max(axis=1)))

    scores = np.concatenate([pos_sims, neg_sims.ravel()])
    labels = np.concatenate([
        np.ones(n,  dtype=np.int8),
        np.zeros(n * n_negatives, dtype=np.int8),
    ])
    auc = float(roc_auc_score(labels, scores))

    return {
        "auc": auc,
        "precision_at_1": p_at_1,
        "n_anchors": n,
        "n_negatives_per_anchor": n_negatives,
    }


# ---------------------------------------------------------------------------
# Per-(model, dataset) evaluation
# ---------------------------------------------------------------------------
def evaluate_model_on_dataset(
    model_key: str,
    model_cfg: dict,
    dataset_name: str,
    dataset_path: Path,
    candidate_pool_size: int,
    n_negatives: int,
    bm25_corpus_size: int | None,
    max_bm25_queries: int | None,
    k_values: list[int],
    seed: int,
    retriever: PretrainedRetriever,
) -> dict:
    print(f"\n[{model_key} × {dataset_name}]")

    pairs = load_pairs(dataset_path)
    if len(pairs) < 2:
        print(f"  SKIP: only {len(pairs)} valid pairs in {dataset_path}")
        return {"model": model_key, "dataset": dataset_name, "error": "not enough pairs"}

    raw_anchors   = [a for a, _ in pairs]
    raw_positives = [p for _, p in pairs]
    n = len(pairs)
    print(f"  loaded {n} pairs")

    # Apply per-model prefixes for encoding (BM25 mining uses raw text below).
    anchor_prefix   = model_cfg.get("anchor_prefix",   "")
    positive_prefix = model_cfg.get("positive_prefix", "")
    anchors   = [anchor_prefix   + a for a in raw_anchors]
    positives = [positive_prefix + p for p in raw_positives]
    if anchor_prefix or positive_prefix:
        print(f"  prefixes: anchor='{anchor_prefix}'  positive='{positive_prefix}'")

    # Encode anchors + positives
    t0 = time.time()
    a_emb = retriever.encode_anchors(anchors)
    p_emb = retriever.encode_candidates(positives)
    enc_time = time.time() - t0
    print(f"  encoded {n * 2} texts in {enc_time:.1f}s (dim={a_emb.shape[1]})")

    # ---- BM25 hard-negative pair metrics ----
    # Use the RAW texts for BM25 indexing (prefixes are encoder-only).
    neg_idx, query_idx = mine_bm25_negatives(
        raw_anchors, raw_positives,
        n_negatives=n_negatives,
        max_corpus_size=bm25_corpus_size,
        max_queries=max_bm25_queries,
        seed=seed,
    )
    # Align anchor/positive embeddings to the queried subset
    qa_emb = a_emb[query_idx]   # (Q, D)
    qp_emb = p_emb[query_idx]   # (Q, D)  gold positives for queried anchors

    pair_metrics = auc_precision_at_1_bm25(qa_emb, qp_emb, p_emb, neg_idx)

    # ---- Retrieval metrics (random pool, same as evaluate_inference.py) ----
    t1 = time.time()
    retrieval_metrics = random_pool_retrieval_metrics(
        a_emb, p_emb,
        pool_size=candidate_pool_size,
        k_values=k_values,
        seed=seed,
    )
    ret_time = time.time() - t1

    print(
        f"  BM25 AUC: {pair_metrics['auc']:.4f}    "
        f"BM25 P@1 (vs {n_negatives} hard negs): {pair_metrics['precision_at_1']:.4f}"
    )
    print(
        f"  MRR: {retrieval_metrics['mrr']:.4f}    "
        f"Mean rank: {retrieval_metrics['mean_rank']:.2f}    "
        f"Median rank: {retrieval_metrics['median_rank']:.1f}"
    )
    for k in k_values:
        print(
            f"  Recall@{k:<3}{retrieval_metrics['recall_at_k'][k]:.4f}"
            f"    Hit@{k:<3}{retrieval_metrics['hit_at_k'][k]:.4f}"
        )

    return {
        "model":                  model_key,
        "model_hf_id":            model_cfg["hf_id"],
        "dataset":                dataset_name,
        "n_queries":              retrieval_metrics["n_queries"],
        "n_bm25_queries":         int(len(query_idx)),
        "bm25_corpus_size":       int(min(n, bm25_corpus_size) if bm25_corpus_size else n),
        "n_candidates_per_query": retrieval_metrics["n_candidates_per_query"],
        "encoding_seconds":       enc_time,
        "ranking_seconds":        ret_time,
        "bm25_auc":               pair_metrics["auc"],
        "bm25_precision_at_1":    pair_metrics["precision_at_1"],
        "n_negatives":            n_negatives,
        "mrr":                    retrieval_metrics["mrr"],
        "mean_rank":              retrieval_metrics["mean_rank"],
        "median_rank":            retrieval_metrics["median_rank"],
        "recall_at_k":            retrieval_metrics["recall_at_k"],
        "hit_at_k":               retrieval_metrics["hit_at_k"],
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], k_values: list[int]) -> None:
    print("\n" + "=" * 130)
    print("SUMMARY  (BM25 hard negatives for AUC/P@1 · random pool for retrieval)")
    print("=" * 130)
    header = (
        f"{'model':<22} {'dataset':<14} {'N':>7} {'bm25Q':>6} "
        f"{'BM25 AUC':>9} {'BM25 P@1':>9} {'MRR':>7} "
        f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'mean':>7} {'median':>7}"
    )
    print(header)
    print("-" * 130)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<22} {r['dataset']:<14}  ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:<22} {r['dataset']:<14} {r['n_queries']:>7d} "
            f"{r['n_bm25_queries']:>6d} "
            f"{r['bm25_auc']:>9.4f} {r['bm25_precision_at_1']:>9.4f} "
            f"{r['mrr']:>7.4f} "
            f"{r['recall_at_k'][1]:>7.4f} {r['recall_at_k'][5]:>7.4f} "
            f"{r['recall_at_k'][10]:>7.4f} "
            f"{r['mean_rank']:>7.2f} {r['median_rank']:>7.1f}"
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
    ap.add_argument("--candidate-pool-size", type=int, default=1000,
                    help="Random candidate pool per anchor for retrieval metrics.")
    ap.add_argument("--n-negatives", type=int, default=4,
                    help="BM25 hard negatives per anchor for AUC / P@1.")
    ap.add_argument("--bm25-corpus-size", type=int, default=10_000,
                    help="Max positives in BM25 index (sampled for large datasets).")
    ap.add_argument("--max-bm25-queries", type=int, default=10_000,
                    help="Max anchors queried per dataset for BM25 AUC/P@1. "
                         "Retrieval metrics always use the full split.")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(ROOT / "evaluation_6/bm25negative/results_bm25.json"),
        help="Output JSON path.",
    )
    args = ap.parse_args()

    k_values = sorted(set(args.k))

    print(f"Models:              {args.models}")
    print(f"Datasets:            {args.datasets}")
    print(f"Candidate pool size: {args.candidate_pool_size}")
    print(f"BM25 negatives:      {args.n_negatives}")
    print(f"BM25 corpus size:    {args.bm25_corpus_size}")
    print(f"Max BM25 queries:    {args.max_bm25_queries}")
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
                    bm25_corpus_size=args.bm25_corpus_size,
                    max_bm25_queries=args.max_bm25_queries,
                    k_values=k_values,
                    seed=args.seed,
                    retriever=retriever,
                )
                results.append(row)
            except Exception as e:
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
            "bm25_corpus_size":    args.bm25_corpus_size,
            "max_bm25_queries":    args.max_bm25_queries,
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
