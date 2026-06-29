"""CausalDensityEmbedding (CDE) model.

Each text is mapped to a Gaussian distribution N(mu, diag(sigma^2)) in
proj_dim-dimensional space, plus a causal drift vector delta.

Scoring A -> B (directional):
  1. Encode A: (mu_A, log_sigma_A, delta_A)
  2. Transport A: mu_T = mu_A + delta_A
                  log_sigma_T = log_sigma_A + gamma/2   (gamma is learned)
  3. Encode B: (mu_B, log_sigma_B, _)
  4. Score = -KL(N_B || T(N_A))

Vectorised helpers (kl_matrix, kl_pointwise) are used for evaluation and
the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class CausalDensityEmbedding(nn.Module):
    def __init__(self, backbone: str = "google-bert/bert-base-uncased", proj_dim: int = 128):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.proj_dim = proj_dim

        self.mu_head = nn.Linear(hidden, proj_dim)
        self.log_sigma_head = nn.Linear(hidden, proj_dim)
        self.delta_head = nn.Linear(hidden, proj_dim)

        # Learned scalar: variance inflation per transport step.
        self.gamma = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.delta_head.weight, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.log_sigma_head.weight, std=1e-3)
        nn.init.zeros_(self.log_sigma_head.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, log_sigma, delta) — shape (B, proj_dim) each."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        h = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 5.0)
        delta = self.delta_head(h)
        return mu, log_sigma, delta

    def transport(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply causal transport: shift mean by delta, inflate variance by gamma/2."""
        return mu + delta, log_sigma + 0.5 * self.gamma

    @staticmethod
    def kl_pointwise(
        mu_p: torch.Tensor,
        log_sigma_p: torch.Tensor,
        mu_q: torch.Tensor,
        log_sigma_q: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N_p || N_q) for N independent pairs — shape (N,).

        KL(p||q) = sum_d [ log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2)/(2 sigma_q^2) - 0.5 ]
        """
        sigma_p_sq = (2 * log_sigma_p).exp()
        sigma_q_sq = (2 * log_sigma_q).exp().clamp(min=1e-9)
        log_ratio = log_sigma_q - log_sigma_p
        mean_term = (sigma_p_sq + (mu_p - mu_q).pow(2)) / (2 * sigma_q_sq)
        return (log_ratio + mean_term - 0.5).sum(dim=-1)

    @staticmethod
    def kl_matrix(
        mu_q: torch.Tensor,
        log_sigma_q: torch.Tensor,
        mu_c: torch.Tensor,
        log_sigma_c: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N_c[j] || N_q[i]) for all (i,j) — shape (N_q, N_c).

        KL(c||q) = sum_d [ log(sigma_q[i,d]/sigma_c[j,d])
                           + (sigma_c[j,d]^2 + (mu_c[j,d] - mu_q[i,d])^2) / (2*sigma_q[i,d]^2)
                           - 0.5 ]
        """
        D = mu_q.shape[-1]
        sigma_q_sq = (2 * log_sigma_q).exp().clamp(min=1e-9)   # (N_q, D)
        sigma_c_sq = (2 * log_sigma_c).exp()                    # (N_c, D)
        inv_sigma_q_sq = 1.0 / sigma_q_sq                       # (N_q, D)

        # Term 1: sum_d(log_sigma_q[i,d]) - sum_d(log_sigma_c[j,d])
        t1 = log_sigma_q.sum(-1).unsqueeze(1) - log_sigma_c.sum(-1).unsqueeze(0)  # (N_q, N_c)

        # Term 2: sum_d [ sigma_c_sq[j,d] / (2*sigma_q_sq[i,d]) ]
        t2 = 0.5 * (inv_sigma_q_sq @ sigma_c_sq.T)  # (N_q, N_c)

        # Term 3: sum_d [ (mu_c[j,d] - mu_q[i,d])^2 / (2*sigma_q_sq[i,d]) ]
        # = sum_d[ mu_c^2 * inv_q/2 ] - sum_d[ mu_c * mu_q * inv_q ] + sum_d[ mu_q^2 * inv_q/2 ]
        half_inv = 0.5 * inv_sigma_q_sq                          # (N_q, D)
        t3 = (
            half_inv @ (mu_c.pow(2)).T                           # (N_q, N_c)
            - (mu_q * inv_sigma_q_sq) @ mu_c.T                  # (N_q, N_c)
            + (mu_q.pow(2) * half_inv).sum(-1, keepdim=True)    # (N_q, 1)
        )

        return t1 + t2 + t3 - 0.5 * D  # (N_q, N_c)

    def score(
        self,
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score(A->B) = -KL(N_B || T(N_A)), shape (B,)."""
        mu_a, log_sigma_a, delta_a = self.encode(a_ids, a_mask)
        mu_b, log_sigma_b, _ = self.encode(b_ids, b_mask)
        mu_ta, log_sigma_ta = self.transport(mu_a, log_sigma_a, delta_a)
        return -self.kl_pointwise(mu_b, log_sigma_b, mu_ta, log_sigma_ta)

    def score_matrix(
        self,
        mu_ta: torch.Tensor,
        log_sigma_ta: torch.Tensor,
        mu_b: torch.Tensor,
        log_sigma_b: torch.Tensor,
    ) -> torch.Tensor:
        """Score matrix S[i,j] = Score(A_i -> B_j), shape (N_q, N_c)."""
        return -self.kl_matrix(mu_ta, log_sigma_ta, mu_b, log_sigma_b)

    def forward_features(
        self,
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
    ) -> dict:
        """Full forward pass returning intermediate tensors for multi-task loss."""
        mu_a, log_sigma_a, delta_a = self.encode(a_ids, a_mask)
        mu_b, log_sigma_b, delta_b = self.encode(b_ids, b_mask)
        mu_ta, log_sigma_ta = self.transport(mu_a, log_sigma_a, delta_a)
        mu_tb, log_sigma_tb = self.transport(mu_b, log_sigma_b, delta_b)
        return {
            "mu_a": mu_a, "log_sigma_a": log_sigma_a, "delta_a": delta_a,
            "mu_b": mu_b, "log_sigma_b": log_sigma_b, "delta_b": delta_b,
            "mu_ta": mu_ta, "log_sigma_ta": log_sigma_ta,
            "mu_tb": mu_tb, "log_sigma_tb": log_sigma_tb,
        }
