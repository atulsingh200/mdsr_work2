"""Loss functions for CDEv2 training.

Three-phase multi-task loss:
  Phase A (warmup)    : InfoNCE only
  Phase B (main)      : InfoNCE + entropy ordering + antisymmetry + sigma_lb
  Phase C (AUC phase) : Phase B losses + BPR ranking loss

BPR (Bayesian Personalized Ranking) loss directly maximises the probability
that each positive pair outranks its hard negatives — this is a differentiable
proxy for AUC and directly addresses v1's weakness on that metric.

Sigma lower-bound regularisation prevents variance collapse (log_sigma → -5)
which was the root cause of the unbounded KL scores in v1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDELossV2(nn.Module):

    def __init__(
        self,
        temperature_init: float = 0.05,
        temperature_min: float = 0.01,
        temperature_max: float = 0.5,
        entropy_margin: float = 0.5,
        antisym_margin: float = 2.0,
        sigma_lb: float = -3.0,
        lambda_entropy: float = 0.0,
        lambda_antisym: float = 0.0,
        lambda_bpr: float = 0.0,
        lambda_sigma_lb: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature_init).log())
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.entropy_margin = entropy_margin
        self.antisym_margin = antisym_margin
        self.sigma_lb = sigma_lb
        # Updated by trainer per phase.
        self.lambda_entropy = lambda_entropy
        self.lambda_antisym = lambda_antisym
        self.lambda_bpr = lambda_bpr
        self.lambda_sigma_lb = lambda_sigma_lb

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    # ── Individual losses ────────────────────────────────────────────────────

    def infonce(
        self,
        score_matrix: torch.Tensor,       # (B, B)
        score_hardneg: torch.Tensor | None,  # (B, K) or None
    ) -> torch.Tensor:
        B = score_matrix.size(0)
        tau = self.temperature
        if score_hardneg is not None and score_hardneg.numel() > 0:
            logits = torch.cat([score_matrix, score_hardneg], dim=1) / tau  # (B, B+K)
        else:
            logits = score_matrix / tau
        labels = torch.arange(B, device=score_matrix.device)
        return F.cross_entropy(logits, labels)

    def entropy_ordering(
        self,
        log_sigma_a: torch.Tensor,   # (B, D)  cause log_sigma
        log_sigma_b: torch.Tensor,   # (B, D)  effect log_sigma
    ) -> torch.Tensor:
        """H(effect) > H(cause) + margin: effects are more uncertain than causes."""
        H_a = log_sigma_a.sum(-1)
        H_b = log_sigma_b.sum(-1)
        return F.relu(H_a + self.entropy_margin - H_b).mean()

    def antisymmetry(
        self,
        score_ab: torch.Tensor,  # (B,) forward scores
        score_ba: torch.Tensor,  # (B,) reverse scores
    ) -> torch.Tensor:
        """Penalise score(B→A) being close to or higher than score(A→B)."""
        return F.relu(score_ba - score_ab + self.antisym_margin).mean()

    def bpr_loss(
        self,
        score_pos: torch.Tensor,    # (B,) positive pair scores
        score_neg: torch.Tensor,    # (B, K) hard negative scores
    ) -> torch.Tensor:
        """Bayesian Personalised Ranking: -mean log sigmoid(score_pos - score_neg).

        Directly optimises that positives rank above hard negatives — a
        differentiable AUC proxy.  Applied in Phase C.
        """
        # score_pos: (B,) → (B, 1) for broadcast
        diff = score_pos.unsqueeze(1) - score_neg        # (B, K)
        return -F.logsigmoid(diff).mean()

    def sigma_lower_bound(
        self,
        log_sigma: torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        """Penalise log_sigma falling below sigma_lb — prevents variance collapse."""
        return F.relu(self.sigma_lb - log_sigma).mean()

    # ── Combined forward ─────────────────────────────────────────────────────

    def forward(
        self,
        score_matrix: torch.Tensor,           # (B, B)
        score_hardneg: torch.Tensor | None,   # (B, K)
        log_sigma_a: torch.Tensor,            # cause log_sigma
        log_sigma_b: torch.Tensor,            # effect log_sigma
        score_ab: torch.Tensor,               # (B,) forward scores
        score_ba: torch.Tensor | None,        # (B,) or None
    ) -> tuple[torch.Tensor, dict]:
        L_info = self.infonce(score_matrix, score_hardneg)
        total = L_info
        stats: dict = {
            "loss_infonce": L_info.item(),
            "temperature": self.temperature.item(),
        }

        if self.lambda_entropy > 0:
            L_ent = self.entropy_ordering(log_sigma_a, log_sigma_b)
            total = total + self.lambda_entropy * L_ent
            stats["loss_entropy"] = L_ent.item()

        if self.lambda_antisym > 0 and score_ba is not None:
            L_anti = self.antisymmetry(score_ab, score_ba)
            total = total + self.lambda_antisym * L_anti
            stats["loss_antisym"] = L_anti.item()

        if self.lambda_sigma_lb > 0:
            L_slb = self.sigma_lower_bound(log_sigma_a) + self.sigma_lower_bound(log_sigma_b)
            total = total + self.lambda_sigma_lb * L_slb
            stats["loss_sigma_lb"] = L_slb.item()

        if self.lambda_bpr > 0 and score_hardneg is not None and score_hardneg.numel() > 0:
            L_bpr = self.bpr_loss(score_ab, score_hardneg)
            total = total + self.lambda_bpr * L_bpr
            stats["loss_bpr"] = L_bpr.item()

        stats["loss_total"] = total.item()
        return total, stats
