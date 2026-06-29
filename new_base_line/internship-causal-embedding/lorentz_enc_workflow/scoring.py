"""Asymmetric causal similarity functions for LorentzEnc.

sim(a, b) = α · cos(space_a, space_b)
          + β · (1/D_t) · Σ_k σ(τ · (time_b_k − time_a_k))

The second term is asymmetric: it is high when t_b > t_a in every
dimension (b is "temporally after" a) and low in the reverse direction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def steep_sigmoid_mean(
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    tau: float = 10.0,
) -> torch.Tensor:
    """Per-pair time-ordering score in (0, 1).

    High when t_b_k > t_a_k across all time dimensions.
    t_a, t_b: (..., time_dim)  — returns (...,)
    """
    return torch.sigmoid(tau * (t_b - t_a)).mean(dim=-1)


def causal_similarity(
    space_a: torch.Tensor,
    time_a: torch.Tensor,
    space_b: torch.Tensor,
    time_b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 10.0,
) -> torch.Tensor:
    """Element-wise asymmetric similarity for aligned pairs.

    space_a, space_b: (B, D_s)  L2-normalised
    time_a,  time_b:  (B, D_t)
    Returns: (B,)
    """
    cos = (space_a * space_b).sum(dim=-1)
    time_score = steep_sigmoid_mean(time_a, time_b, tau=tau)
    return alpha * cos + beta * time_score


def causal_similarity_matrix(
    space_q: torch.Tensor,
    time_q: torch.Tensor,
    space_k: torch.Tensor,
    time_k: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 10.0,
) -> torch.Tensor:
    """All-pairs asymmetric similarity.

    space_q, time_q: (Q, D_s/D_t)
    space_k, time_k: (K, D_s/D_t)
    Returns: (Q, K)
    """
    cos = space_q @ space_k.t()                                   # (Q, K)
    diff = time_k.unsqueeze(0) - time_q.unsqueeze(1)              # (Q, K, D_t)
    time_score = torch.sigmoid(tau * diff).mean(dim=-1)           # (Q, K)
    return alpha * cos + beta * time_score
