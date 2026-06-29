"""Retrieval metrics: MRR, Recall@K, Hit@K, mean / median rank.

These metric functions handle the general case of one or more gold
candidates per query. They take a similarity matrix and a list of gold
indices per query, and return aggregate metrics over all queries.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _rank_lists(
    similarity: np.ndarray,
    gold_indices: list[list[int]],
) -> list[list[int]]:
    """For each query, return the *sorted ranks* of its gold candidates.

    Ranks are 1-based: 1 is the best (highest similarity) candidate.
    Returns one list per query, sorted ascending (best rank first).
    """
    n_q = similarity.shape[0]
    if len(gold_indices) != n_q:
        raise ValueError(f"gold_indices length {len(gold_indices)} != n_queries {n_q}")
    # argsort once per row (descending similarity).
    # `.argsort(-sim)` gives indices in order from highest to lowest similarity.
    order = np.argsort(-similarity, axis=1)
    # rank[i, j] = 1-based rank of candidate j for query i
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
        similarity:    (n_queries, n_candidates) array. Higher = more similar.
        gold_indices:  per query, the indices (in the candidate pool) of the
                       gold / relevant candidates. Length must equal n_queries.
        k_values:      cut-offs for Recall@K and Hit@K.

    Returns:
        Dict with:
          mrr           Mean Reciprocal Rank of the *best-ranked* gold per query.
          mean_rank     Mean of best-gold rank across queries.
          median_rank   Median of best-gold rank across queries.
          recall_at_k   Per-K: fraction of golds found in top-K, averaged over queries
                       (so for 1 gold/query reduces to standard Recall@K).
          hit_at_k      Per-K: fraction of queries with >=1 gold in top-K.

    Notes:
        * `recall_at_k` and `hit_at_k` coincide when every query has exactly one gold.
        * Empty gold lists are skipped in averages but counted as zero recall/hit.
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
            best_ranks.append(0)  # unknown; will be filtered for rank stats
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

    valid_ranks = [r for r in best_ranks if r > 0]
    if not valid_ranks:
        valid_ranks = [0]  # avoid div-zero

    return {
        "n_queries": n_queries,
        "n_queries_with_gold": sum(1 for g in gold_indices if g),
        "mrr": mrr_sum / max(n_queries, 1),
        "mean_rank": float(np.mean(valid_ranks)),
        "median_rank": float(np.median(valid_ranks)),
        "recall_at_k": {k: recall_sums[k] / max(n_queries, 1) for k in k_values},
        "hit_at_k": {k: hit_sums[k] / max(n_queries, 1) for k in k_values},
    }


def random_baseline_recall(n_candidates: int, k_values: Iterable[int] = (1, 3, 5, 10), n_gold: int = 1) -> dict:
    """Analytical random-baseline Recall@K: K * n_gold / n_candidates, capped at 1."""
    return {
        int(k): min(int(k) * n_gold / max(n_candidates, 1), 1.0)
        for k in k_values
    }
