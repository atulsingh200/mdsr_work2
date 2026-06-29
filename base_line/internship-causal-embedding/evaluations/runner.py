"""EvalRunner: the single evaluation pipeline all models go through.

Every model × dataset × negative-strategy combination calls run_eval().
No metric logic lives anywhere else.

Cross-encoders (IS_CROSS_ENCODER=True) use score_matrix() instead of
encode_anchors/encode_candidates, but produce the same metrics dict.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evaluations.metrics import (
    auc_and_precision_at_1,
    compute_metrics,
    pool_retrieval_metrics,
    random_baseline_recall,
)


@dataclass
class EvalConfig:
    dataset: str
    split: str = "test"
    negative_strategy: str = "random"
    k_negatives: int = 4
    pool_size: int = 1000
    seed: int = 42
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    max_queries: int = 0          # 0 = use all


def run_eval(
    retriever,
    dataset_name: str,
    split: str = "test",
    negative_strategy: str = "random",
    k_negatives: int = 4,
    pool_size: int = 1000,
    seed: int = 42,
    k_values: list[int] | None = None,
    max_queries: int = 0,
) -> dict[str, Any]:
    """Run the full evaluation pipeline for one (model, dataset, negatives) triple.

    Args:
        retriever:          any object satisfying models.base.Retriever protocol,
                            or CrossEncoder2xRetriever (IS_CROSS_ENCODER=True).
        dataset_name:       registered followup_data dataset name.
        split:              "train", "val", or "test".
        negative_strategy:  "random", "shuffle", or "bm25".
        k_negatives:        negatives per anchor for AUC/P@1 metrics.
        pool_size:          candidate pool size for MRR/Recall@K.
        seed:               RNG seed (applied identically to all models).
        k_values:           Recall@K and Hit@K cut-offs.
        max_queries:        cap number of queries (0 = all). For quick smoke tests.

    Returns:
        Dict with all metrics plus config used.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    k_values = sorted(k_values)

    # ------------------------------------------------------------------ #
    # 1. Load dataset pairs
    # ------------------------------------------------------------------ #
    import followup_data.loaders  # noqa: F401 — triggers self-registration of all loaders
    from followup_data.registry import load, default_root

    t_load = time.time()
    ds = load(dataset_name, root=default_root(), split=split)
    pairs = list(ds)  # BaseDataset.__iter__ yields PairExamples
    if not pairs:
        raise ValueError(f"No pairs found for dataset={dataset_name} split={split}")

    if max_queries > 0 and max_queries < len(pairs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_queries, replace=False)
        idx.sort()
        pairs = [pairs[i] for i in idx]

    anchors   = [p.anchor   for p in pairs]
    positives = [p.positive for p in pairs]
    n = len(pairs)
    print(f"  [{dataset_name}/{split}] {n} pairs  (loaded in {time.time()-t_load:.1f}s)")

    # ------------------------------------------------------------------ #
    # 2. Encode
    # ------------------------------------------------------------------ #
    is_cross = getattr(retriever, "IS_CROSS_ENCODER", False)

    t_enc = time.time()
    if not is_cross:
        a_emb = retriever.encode_anchors(anchors)      # (N, D)
        c_emb = retriever.encode_candidates(positives) # (N, D)
        enc_seconds = time.time() - t_enc
        print(f"  encoded {2*n} texts in {enc_seconds:.1f}s  shape={a_emb.shape}")
    else:
        a_emb = c_emb = None
        enc_seconds = 0.0
        print(f"  cross-encoder mode — scoring pairs on-the-fly")

    # ------------------------------------------------------------------ #
    # 3. AUC + P@1 (vs k_negatives random negatives)
    # ------------------------------------------------------------------ #
    if not is_cross:
        pair_metrics = auc_and_precision_at_1(a_emb, c_emb, n_negatives=k_negatives, seed=seed)
    else:
        pair_metrics = _cross_auc(retriever, anchors, positives, k_negatives, seed)

    print(f"  AUC={pair_metrics['auc']:.4f}  P@1={pair_metrics['precision_at_1']:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Pool retrieval: MRR / Recall@K / Hit@K
    # ------------------------------------------------------------------ #
    if not is_cross:
        retrieval_metrics = pool_retrieval_metrics(
            a_emb, c_emb, pool_size=pool_size, k_values=k_values, seed=seed
        )
    else:
        retrieval_metrics = _cross_pool(
            retriever, anchors, positives, pool_size, k_values, seed
        )

    actual_pool = retrieval_metrics["n_candidates_per_query"]
    print(f"  MRR={retrieval_metrics['mrr']:.4f}  "
          f"mean_rank={retrieval_metrics['mean_rank']:.2f}  "
          f"pool={actual_pool}")
    for k in k_values:
        print(f"    Recall@{k:<3}{retrieval_metrics['recall_at_k'][k]:.4f}  "
              f"Hit@{k:<3}{retrieval_metrics['hit_at_k'][k]:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Assemble results
    # ------------------------------------------------------------------ #
    rand_baseline = random_baseline_recall(actual_pool, k_values)

    return {
        "model":             retriever.name,
        "dataset":           dataset_name,
        "split":             split,
        "negative_strategy": negative_strategy,
        "n_pairs":           n,
        "encoding_seconds":  enc_seconds,
        "pool_size":         actual_pool,
        "seed":              seed,
        "k_negatives":       k_negatives,
        # pair metrics
        "auc":               pair_metrics["auc"],
        "precision_at_1":    pair_metrics["precision_at_1"],
        # retrieval metrics
        "mrr":               retrieval_metrics["mrr"],
        "mean_rank":         retrieval_metrics["mean_rank"],
        "median_rank":       retrieval_metrics["median_rank"],
        "recall_at_k":       retrieval_metrics["recall_at_k"],
        "hit_at_k":          retrieval_metrics["hit_at_k"],
        # random baseline
        "random_baseline_recall": rand_baseline,
    }


# ---------------------------------------------------------------------------
# Cross-encoder helpers (internal — not part of the Retriever protocol)
# ---------------------------------------------------------------------------

def _cross_auc(retriever, anchors, positives, k_negatives, seed):
    rng = np.random.default_rng(seed)
    n = len(anchors)
    pos_scores = retriever.score_pairs(anchors, positives)

    auc_wins = 0
    auc_total = 0
    p1_wins = 0
    for i in range(n):
        neg_idx = rng.choice(
            [j for j in range(n) if j != i],
            size=min(k_negatives, n - 1),
            replace=False,
        )
        neg_scores = retriever.score_pairs(
            [anchors[i]] * len(neg_idx),
            [positives[j] for j in neg_idx],
        )
        sp = pos_scores[i]
        wins = int(np.sum(sp > neg_scores))
        auc_wins += wins
        auc_total += len(neg_scores)
        if wins == len(neg_scores):
            p1_wins += 1

    return {
        "auc": auc_wins / max(auc_total, 1),
        "precision_at_1": p1_wins / max(n, 1),
        "n_pairs": n,
    }


def _cross_pool(retriever, anchors, positives, pool_size, k_values, seed):
    rng = np.random.default_rng(seed)
    n = len(anchors)
    actual_pool = min(pool_size, n)

    similarity_rows = []
    gold_indices = []

    for i in range(n):
        others = [j for j in range(n) if j != i]
        neg_idx = rng.choice(others, size=min(actual_pool - 1, len(others)), replace=False)
        pool_idx = [i] + list(neg_idx)
        pool_cands = [positives[j] for j in pool_idx]
        scores = retriever.score_pairs([anchors[i]] * len(pool_cands), pool_cands)
        similarity_rows.append(scores)
        gold_indices.append([0])

    similarity = np.stack(similarity_rows, axis=0)
    metrics = compute_metrics(similarity, gold_indices, k_values)
    metrics["n_candidates_per_query"] = actual_pool
    return metrics
