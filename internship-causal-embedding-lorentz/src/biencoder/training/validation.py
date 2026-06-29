"""In-training validation utilities.

These functions are intentionally lightweight: they compute MRR / Recall@K /
mean & median rank over the validation set using the model that is currently
in memory and the same dataloader as training. They are NOT a substitute for
the standalone evaluation harness (a separate `src/evaluation/` module is
planned for that). Use these only between training epochs to track progress
and drive early stopping.
"""

from __future__ import annotations

from typing import Iterable

import torch

from ..losses import InfoNCELoss
from ..model import BiEncoder


def compute_retrieval_metrics(
    anchor_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    recall_k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict:
    """Compute MRR, mean / median rank, and Recall@K over a similarity matrix.

    Assumes the i-th anchor matches the i-th positive (diagonal). For each
    anchor we count how many positives score >= the diagonal entry; that count
    is the rank (1-based).

    Args:
        anchor_embeds:   (N, D) L2-normalized.
        positive_embeds: (N, D) L2-normalized.
        recall_k_values: cut-offs at which to compute Recall@K.

    Returns:
        dict with keys: mrr, mean_rank, median_rank, recall@<K> for each K.
    """
    sim = anchor_embeds @ positive_embeds.T               # (N, N)
    diag = sim.diag().unsqueeze(1)                        # (N, 1)
    # rank = number of items with similarity >= the gold pair's
    ranks = (sim >= diag).sum(dim=1).float()              # (N,)

    metrics: dict[str, float] = {
        "mrr": (1.0 / ranks).mean().item(),
        "mean_rank": ranks.mean().item(),
        "median_rank": ranks.median().item(),
    }
    for k in recall_k_values:
        metrics[f"recall@{k}"] = (ranks <= k).float().mean().item()
    return metrics


@torch.no_grad()
def validate(
    model: BiEncoder,
    dataloader,
    loss_fn: InfoNCELoss,
    device: torch.device,
    recall_k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict:
    """Run validation: average loss + retrieval metrics over the val set.

    The model is encoded in batches; embeddings are concatenated; retrieval
    metrics are then computed over the full N×N similarity matrix.
    """
    model.eval()
    all_a: list[torch.Tensor] = []
    all_p: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    for a_ids, a_mask, p_ids, p_mask in dataloader:
        a_ids = a_ids.to(device, non_blocking=True)
        a_mask = a_mask.to(device, non_blocking=True)
        p_ids = p_ids.to(device, non_blocking=True)
        p_mask = p_mask.to(device, non_blocking=True)
        a, p = model(a_ids, a_mask, p_ids, p_mask)
        loss, _ = loss_fn(a, p)
        total_loss += loss.item()
        n_batches += 1
        all_a.append(a.cpu())
        all_p.append(p.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    anchor_embeds = torch.cat(all_a, dim=0)
    positive_embeds = torch.cat(all_p, dim=0)

    metrics = compute_retrieval_metrics(anchor_embeds, positive_embeds, recall_k_values)
    metrics["loss"] = avg_loss
    return metrics
