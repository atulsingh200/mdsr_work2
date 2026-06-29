import torch
import torch.nn.functional as F


def steep_sigmoid_mean(t_a: torch.Tensor, t_b: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    """Asymmetric time-precedence score: high when t_b is 'after' t_a in every dim."""
    diff = t_b - t_a
    return torch.sigmoid(tau * diff).mean(dim=-1)


def causal_similarity(
    space_a: torch.Tensor, time_a: torch.Tensor,
    space_b: torch.Tensor, time_b: torch.Tensor,
    alpha: float = 1.0, beta: float = 0.5, tau: float = 10.0,
) -> torch.Tensor:
    """sim(a,b) = α · cos(space) + β · SteepSigmoid_mean(time)"""
    cos = (space_a * space_b).sum(dim=-1)
    time_score = steep_sigmoid_mean(time_a, time_b, tau=tau)
    return alpha * cos + beta * time_score


def causal_similarity_matrix(
    space_q: torch.Tensor, time_q: torch.Tensor,
    space_k: torch.Tensor, time_k: torch.Tensor,
    alpha: float = 1.0, beta: float = 0.5, tau: float = 10.0,
) -> torch.Tensor:
    """All-pairs causal similarity. Returns (Q, K)."""
    cos = space_q @ space_k.t()
    diff = time_k.unsqueeze(0) - time_q.unsqueeze(1)
    time_score = torch.sigmoid(tau * diff).mean(dim=-1)
    return alpha * cos + beta * time_score
