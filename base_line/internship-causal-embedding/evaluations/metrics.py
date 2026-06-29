"""Single source of truth for all retrieval metrics.

Covers:
  - MRR, Recall@K, Hit@K, mean/median rank  (from src/evaluation/metrics.py)
  - AUC, Precision@1 vs random negatives     (from evaluation_6/metrics_extra.py)

Every model goes through these functions. Never copy-paste metric logic elsewhere.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Core ranking metrics
# ---------------------------------------------------------------------------

def _rank_lists(
    similarity: np.ndarray,
    gold_indices: list[list[int]],
) -> list[list[int]]:
    """For each query return the 1-based ranks of its gold candidates, sorted ascending."""
    n_q = similarity.shape[0]
    if len(gold_indices) != n_q:
        raise ValueError(f"gold_indices length {len(gold_indices)} != n_queries {n_q}")
    order = np.argsort(-similarity, axis=1)
    n_c = similarity.shape[1]
    rank = np.empty_like(order)
    cols = np.arange(n_c)
    for i in range(n_q):
        rank[i, order[i]] = cols + 1
    out: list[list[int]] = []
    for i, golds in enumerate(gold_indices):
        out.append(sorted(int(rank[i, g]) for g in golds))
    return out


def compute_metrics(
    similarity: np.ndarray,
    gold_indices: list[list[int]],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict:
    """Compute retrieval metrics from a similarity matrix.

    Args:
        similarity:   (n_queries, n_candidates) array. Higher = more similar.
        gold_indices: per-query list of gold candidate indices. len == n_queries.
        k_values:     cut-offs for Recall@K and Hit@K.

    Returns:
        Dict with mrr, mean_rank, median_rank, recall_at_k, hit_at_k,
        n_queries, n_queries_with_gold.
    """
    k_values = sorted(int(k) for k in k_values)
    ranks_per_query = _rank_lists(similarity, gold_indices)

    best_ranks: list[int] = []
    recall_sums = {k: 0.0 for k in k_values}
    hit_sums = {k: 0 for k in k_values}
    mrr_sum = 0.0
    n_queries = len(gold_indices)

    for ranks in ranks_per_query:
        if not ranks:
            best_ranks.append(0)
            continue
        best = ranks[0]
        best_ranks.append(best)
        mrr_sum += 1.0 / best
        n_gold = len(ranks)
        for k in k_values:
            n_in_top_k = sum(1 for r in ranks if r <= k)
            recall_sums[k] += n_in_top_k / n_gold
            if n_in_top_k > 0:
                hit_sums[k] += 1

    valid_ranks = [r for r in best_ranks if r > 0] or [0]

    return {
        "n_queries": n_queries,
        "n_queries_with_gold": sum(1 for g in gold_indices if g),
        "mrr": mrr_sum / max(n_queries, 1),
        "mean_rank": float(np.mean(valid_ranks)),
        "median_rank": float(np.median(valid_ranks)),
        "recall_at_k": {k: recall_sums[k] / max(n_queries, 1) for k in k_values},
        "hit_at_k": {k: hit_sums[k] / max(n_queries, 1) for k in k_values},
    }


def random_baseline_recall(
    n_candidates: int,
    k_values: Iterable[int] = (1, 3, 5, 10),
    n_gold: int = 1,
) -> dict:
    """Analytical random-baseline Recall@K = K * n_gold / n_candidates, capped at 1."""
    return {int(k): min(int(k) * n_gold / max(n_candidates, 1), 1.0) for k in k_values}


# ---------------------------------------------------------------------------
# Pair-level metrics (AUC + Precision@1 vs random negatives)
# ---------------------------------------------------------------------------

def auc_and_precision_at_1(
    anchor_embs: np.ndarray,
    positive_embs: np.ndarray,
    n_negatives: int = 4,
    seed: int = 42,
) -> dict:
    """Compute AUC and Precision@1 for each anchor vs n_negatives random negatives.

    For each pair (a_i, p_i), sample n_negatives random positives p_j (j != i)
    as negatives. Score = dot product (assumes L2-normalized embeddings).

    AUC: fraction of (pos, neg) pairs where score(a, pos) > score(a, neg).
    P@1: fraction of anchors where score(a, pos) > all n_negatives neg scores.

    Returns:
        Dict with auc, precision_at_1, n_pairs.
    """
    rng = np.random.default_rng(seed)
    n = len(anchor_embs)
    scores_pos = np.einsum("id,id->i", anchor_embs, positive_embs)  # (n,)

    auc_wins = 0
    auc_total = 0
    p1_wins = 0

    for i in range(n):
        neg_idx = rng.choice(
            [j for j in range(n) if j != i],
            size=min(n_negatives, n - 1),
            replace=False,
        )
        neg_scores = anchor_embs[i] @ positive_embs[neg_idx].T  # (n_neg,)
        sp = scores_pos[i]
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


# ---------------------------------------------------------------------------
# Pool-based retrieval metrics (capped candidate pool)
# ---------------------------------------------------------------------------

def pool_retrieval_metrics(
    anchor_embs: np.ndarray,
    positive_embs: np.ndarray,
    pool_size: int = 1000,
    k_values: Iterable[int] = (1, 3, 5, 10),
    seed: int = 42,
) -> dict:
    """MRR/Recall@K/Hit@K over a randomly sampled candidate pool.

    For each anchor i, the pool is {positive_i} ∪ (pool_size-1 random other positives).
    The gold index within the pool is always 0 (positive_i is inserted first).

    Args:
        anchor_embs:   (N, D) L2-normalized.
        positive_embs: (N, D) L2-normalized, index-aligned with anchor_embs.
        pool_size:     total candidates per query (including the gold).
        k_values:      Recall@K and Hit@K cut-offs.
        seed:          RNG seed for reproducibility.

    Returns:
        Dict with all compute_metrics fields plus n_candidates_per_query.
    """
    k_values = sorted(int(k) for k in k_values)
    rng = np.random.default_rng(seed)
    n = len(anchor_embs)
    actual_pool = min(pool_size, n)

    similarity_rows: list[np.ndarray] = []
    gold_indices: list[list[int]] = []

    for i in range(n):
        others = [j for j in range(n) if j != i]
        neg_idx = rng.choice(others, size=min(actual_pool - 1, len(others)), replace=False)
        pool_idx = np.concatenate([[i], neg_idx])          # gold always at position 0
        pool_embs = positive_embs[pool_idx]                # (pool_size, D)
        sims = anchor_embs[i] @ pool_embs.T               # (pool_size,)
        similarity_rows.append(sims)
        gold_indices.append([0])

    similarity = np.stack(similarity_rows, axis=0)         # (N, pool_size)
    metrics = compute_metrics(similarity, gold_indices, k_values)
    metrics["n_candidates_per_query"] = actual_pool
    return metrics
