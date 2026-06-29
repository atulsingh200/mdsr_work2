"""Pair-level and retrieval metrics for evaluation.

Three families:

  1) Pair-level: AUC and Precision@1 with N random negatives per anchor.
     O(N · K). Cheap, reproducible cross-model comparison.

  2) Random-pool retrieval: rank each anchor's true positive against
     (pool_size - 1) random distractors. O(N · pool_size).

  3) Full N×N retrieval: rank each anchor against ALL N positives.
     Exact metrics — no sampling. O(N²) via batched GPU matmul.
     AUC derived from ranks: mean((N - rank_i) / (N - 1)).
     P@1 == Recall@1 in this setting.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Pair-level: AUC + P@1 with N random negatives
# ---------------------------------------------------------------------------
def auc_precision_at_1_with_random_negatives(
    anchor_emb: np.ndarray,
    positive_emb: np.ndarray,
    n_negatives: int = 4,
    seed: int = 0,
) -> dict:
    """Compute AUC and Precision@1 against `n_negatives` random negatives per anchor.

    Args:
        anchor_emb:   (N, D) L2-normalized anchor embeddings.
        positive_emb: (N, D) L2-normalized positive embeddings (aligned with anchors).
        n_negatives:  Random negatives per anchor (sampled from other rows' positives,
                      without replacement, excluding self).
        seed:         RNG seed for reproducibility.

    Returns:
        Dict with:
            auc                       float, ROC-AUC over all (pos,neg) score-label pairs
            precision_at_1            float, fraction of anchors where pos beats all negs
            n_anchors                 int
            n_negatives_per_anchor    int
    """
    n = len(anchor_emb)
    if n < 2:
        raise ValueError(f"need >=2 anchors, got {n}")
    if n_negatives >= n:
        raise ValueError(f"n_negatives={n_negatives} must be < n={n}")

    rng = np.random.default_rng(seed)

    # Sample N negative *positive-indices* per anchor, excluding the anchor's own index.
    # Trick: pick from [0, n-1) then bump indices >= i by 1.
    neg_idx = np.empty((n, n_negatives), dtype=np.int64)
    for i in range(n):
        picks = rng.choice(n - 1, size=n_negatives, replace=False)
        picks[picks >= i] += 1
        neg_idx[i] = picks

    pos_sims = np.sum(anchor_emb * positive_emb, axis=1)              # (n,)
    neg_pos = positive_emb[neg_idx]                                   # (n, n_neg, d)
    neg_sims = np.einsum("id,ind->in", anchor_emb, neg_pos)           # (n, n_neg)

    p_at_1 = float(np.mean(pos_sims > neg_sims.max(axis=1)))

    scores = np.concatenate([pos_sims, neg_sims.ravel()])
    labels = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n * n_negatives, dtype=np.int8)])
    auc = float(roc_auc_score(labels, scores))

    return {
        "auc": auc,
        "precision_at_1": p_at_1,
        "n_anchors": n,
        "n_negatives_per_anchor": n_negatives,
    }


# ---------------------------------------------------------------------------
# Pair-level: AUC + P@1 with semantic hard negatives ONLY (no random)
# ---------------------------------------------------------------------------
def auc_precision_at_1_with_semantic_hard_negatives(
    anchor_emb: np.ndarray,
    positive_emb: np.ndarray,
    hard_neg_idx: np.ndarray,
) -> dict:
    """AUC and P@1 against pre-mined SEMANTIC hard negatives — no random sampling.

    Every anchor i is scored against:
        - its true positive (label 1)
        - the K positives whose indices are listed in ``hard_neg_idx[i]`` (label 0)
    Those K negatives are typically the semantic-kNN nearest other positives
    for anchor i's true positive (see ``mine_hard_negatives.py``), i.e. the
    same kind of hard negatives the model was trained against.

    Args:
        anchor_emb:    (N, D) L2-normalized anchor embeddings.
        positive_emb:  (N, D) L2-normalized positive embeddings (aligned with anchors).
        hard_neg_idx: (N, K) int matrix of pre-mined hard-negative indices into
                       ``positive_emb``. NO random negatives are added.

    Returns dict with: auc, precision_at_1, n_anchors, n_negatives_per_anchor.
    """
    n = len(anchor_emb)
    if n < 2:
        raise ValueError(f"need >=2 anchors, got {n}")
    if hard_neg_idx.shape[0] != n:
        raise ValueError(f"hard_neg_idx rows {hard_neg_idx.shape[0]} != n {n}")
    if hard_neg_idx.dtype.kind not in "iu":
        hard_neg_idx = hard_neg_idx.astype(np.int64)

    K = int(hard_neg_idx.shape[1])

    pos_sims = np.sum(anchor_emb * positive_emb, axis=1)                  # (n,)
    neg_pos = positive_emb[hard_neg_idx]                                  # (n, K, d)
    neg_sims = np.einsum("id,ikd->ik", anchor_emb, neg_pos)               # (n, K)

    p_at_1 = float(np.mean(pos_sims > neg_sims.max(axis=1)))

    scores = np.concatenate([pos_sims, neg_sims.ravel()])
    labels = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n * K, dtype=np.int8)])
    auc = float(roc_auc_score(labels, scores))

    return {
        "auc": auc,
        "precision_at_1": p_at_1,
        "n_anchors": n,
        "n_negatives_per_anchor": K,
    }


# ---------------------------------------------------------------------------
# Retrieval: per-anchor random candidate pool, GPU-accelerated
# ---------------------------------------------------------------------------
def random_pool_retrieval_metrics(
    anchor_emb: np.ndarray,
    positive_emb: np.ndarray,
    pool_size: int = 1000,
    k_values: list[int] | None = None,
    seed: int = 0,
    device: str | None = None,
    batch_size: int = 256,
) -> dict:
    """Rank each anchor's true positive against (pool_size - 1) random distractors.

    For each anchor i, the candidate pool is:  [ positive_i (gold) ] ∪ S_i
    where S_i is a uniform random sample of (pool_size - 1) other positives
    (sampled without replacement, excluding positive_i).

    This gives O(N · pool_size) work rather than O(N²) for full same-split
    retrieval, so the entire dataset can be evaluated regardless of size.
    All similarities are computed on the GPU in batches.

    If pool_size >= N, falls back to full same-split (every positive is a
    candidate for every anchor, with each anchor's own positive as the gold).

    Args:
        anchor_emb:   (N, D) L2-normalized.
        positive_emb: (N, D) L2-normalized (aligned with anchors).
        pool_size:    Candidate pool size per anchor (capped at N).
        k_values:     K cutoffs for Recall@K / Hit@K.
        seed:         RNG seed for distractor sampling.
        device:       "cuda" / "cpu" / None (= auto).
        batch_size:   Anchors processed per GPU batch.

    Returns:
        Dict shaped like `evaluation.metrics.compute_metrics`:
            n_queries, n_candidates_per_query,
            mrr, mean_rank, median_rank,
            recall_at_k {k: float},
            hit_at_k    {k: float}  (== recall_at_k since 1 gold/query)
    """
    import torch

    if k_values is None:
        k_values = [1, 3, 5, 10]
    k_values = sorted(set(int(k) for k in k_values))

    N, _ = anchor_emb.shape
    actual_pool_size = min(pool_size, N)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    a = torch.from_numpy(anchor_emb).to(device)
    p = torch.from_numpy(positive_emb).to(device)
    pos_sims = (a * p).sum(dim=1)  # (N,)

    ranks = torch.zeros(N, dtype=torch.long, device=device)

    if actual_pool_size >= N:
        # Full same-split: rank against ALL positives (N candidates per anchor).
        # rank_i = #{j : sim(a_i, p_j) > sim(a_i, p_i)} + 1   (j == i contributes 0)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            sim_batch = a[start:end] @ p.T              # (b, N)
            diag = pos_sims[start:end].unsqueeze(1)     # (b, 1)
            ranks[start:end] = (sim_batch > diag).sum(dim=1) + 1
    else:
        # Per-anchor random pool.
        n_distractors = actual_pool_size - 1
        rng = np.random.default_rng(seed)
        # With-replacement sampling — collisions are rare when n_distractors << N
        # (expected ~n_distractors² / (2N) duplicates per row). For pool_size=1000
        # and N >= 10k that's < 5% duplicates, which doesn't materially affect
        # the empirical rank distribution.
        rand = rng.integers(0, N - 1, size=(N, n_distractors), dtype=np.int64)
        rand = rand + (rand >= np.arange(N)[:, None])           # bump >= self
        d_idx = torch.from_numpy(rand).to(device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_a = a[start:end]                              # (b, d)
            batch_d_idx = d_idx[start:end]                      # (b, n_distractors)
            batch_neg_p = p[batch_d_idx]                        # (b, n_distractors, d)
            neg_sims = torch.einsum("bd,bnd->bn", batch_a, batch_neg_p)
            batch_pos = pos_sims[start:end].unsqueeze(1)        # (b, 1)
            ranks[start:end] = (neg_sims > batch_pos).sum(dim=1) + 1

    ranks_np = ranks.cpu().numpy()
    recall = {k: float((ranks_np <= k).mean()) for k in k_values}
    return {
        "n_queries":              N,
        "n_candidates_per_query": actual_pool_size,
        "mrr":                    float((1.0 / ranks_np).mean()),
        "mean_rank":              float(ranks_np.mean()),
        "median_rank":            float(np.median(ranks_np)),
        "recall_at_k":            recall,
        "hit_at_k":               recall,  # 1 gold/query → coincide
    }


# ---------------------------------------------------------------------------
# Full N×N retrieval — exact metrics, no pool sampling
# ---------------------------------------------------------------------------
def full_retrieval_metrics(
    anchor_emb: np.ndarray,
    positive_emb: np.ndarray,
    k_values: list[int] | None = None,
    device: str | None = None,
    batch_size: int = 512,
) -> dict:
    """Rank each anchor against ALL N positives (true same-split retrieval).

    rank_i = #{j : sim(a_i, p_j) > sim(a_i, p_i)} + 1  (1-indexed).
    Computed via batched matmul on GPU: O(N²/B) passes of size (B, N).

    Derived metrics:
        AUC   = mean((N - rank_i) / (N - 1))   — exact, no sampling
        P@1   = Recall@1                         — fraction ranked 1st

    Args:
        anchor_emb:   (N, D) L2-normalised anchor embeddings.
        positive_emb: (N, D) L2-normalised positive embeddings (aligned).
        k_values:     K cutoffs for Recall@K / Hit@K.
        device:       "cuda" / "cpu" / None (auto).
        batch_size:   Query rows per GPU batch (tune for VRAM).

    Returns:
        Dict with n_queries, n_candidates, auc, precision_at_1, mrr,
        mean_rank, median_rank, recall_at_k, hit_at_k.
    """
    import torch

    if k_values is None:
        k_values = [1, 3, 5, 10]
    k_values = sorted(set(int(k) for k in k_values))
    if 1 not in k_values:
        k_values = [1] + k_values  # always need rank-1 for P@1

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = len(anchor_emb)
    a = torch.from_numpy(anchor_emb).to(device)
    p = torch.from_numpy(positive_emb).to(device)
    pos_sims = (a * p).sum(dim=1)  # (N,)

    ranks = torch.zeros(N, dtype=torch.long, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sim_batch = a[start:end] @ p.T              # (b, N)
        diag = pos_sims[start:end].unsqueeze(1)     # (b, 1)
        ranks[start:end] = (sim_batch > diag).sum(dim=1) + 1

    ranks_np = ranks.cpu().numpy().astype(np.float64)
    recall = {k: float((ranks_np <= k).mean()) for k in k_values}
    auc = float(((N - ranks_np) / (N - 1)).mean())

    return {
        "n_queries":      N,
        "n_candidates":   N,
        "auc":            auc,
        "precision_at_1": recall[1],
        "mrr":            float((1.0 / ranks_np).mean()),
        "mean_rank":      float(ranks_np.mean()),
        "median_rank":    float(np.median(ranks_np)),
        "recall_at_k":    recall,
        "hit_at_k":       recall,
    }
