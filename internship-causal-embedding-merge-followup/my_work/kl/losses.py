"""Loss functions for CDE training.

Multi-task loss with phased schedule:
  Phase A  (warmup)  : InfoNCE only
  Phase B  (main)    : InfoNCE + entropy ordering + antisymmetry

InfoNCE is in-batch (B×B score matrix) + K explicit hard negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDELoss(nn.Module):
    """Combines InfoNCE + entropy ordering + antisymmetry."""

    def __init__(
        self,
        temperature_init: float = 0.05,
        temperature_min: float = 0.01,
        temperature_max: float = 0.5,
        entropy_margin: float = 0.5,
        antisym_margin: float = 2.0,
        lambda_entropy: float = 0.0,
        lambda_antisym: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature_init).log())
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.entropy_margin = entropy_margin
        self.antisym_margin = antisym_margin
        # These are updated by the trainer according to the phase schedule.
        self.lambda_entropy = lambda_entropy
        self.lambda_antisym = lambda_antisym

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    def infonce(
        self,
        score_matrix: torch.Tensor,     # (B, B) in-batch scores
        score_hardneg: torch.Tensor | None,  # (B, K) or None
    ) -> torch.Tensor:
        """In-batch InfoNCE with optional hard negatives in the denominator.

        Each row i: positive is column i (diagonal), negatives are all
        other columns plus the K hard-negative scores.
        """
        B = score_matrix.size(0)
        tau = self.temperature
        if score_hardneg is not None and score_hardneg.numel() > 0:
            logits = torch.cat([score_matrix, score_hardneg], dim=1) / tau  # (B, B+K)
        else:
            logits = score_matrix / tau                                       # (B, B)
        labels = torch.arange(B, device=score_matrix.device)
        return F.cross_entropy(logits, labels)

    def entropy_ordering(
        self,
        log_sigma_a: torch.Tensor,  # (B, D)
        log_sigma_b: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        """Encourage H(B) > H(A) + margin — cause is more certain than effect."""
        H_a = log_sigma_a.sum(-1)
        H_b = log_sigma_b.sum(-1)
        return F.relu(H_a + self.entropy_margin - H_b).mean()

    def antisymmetry(
        self,
        score_ab: torch.Tensor,  # (B,) forward scores
        score_ba: torch.Tensor,  # (B,) reverse scores
    ) -> torch.Tensor:
        """Penalise score(B->A) being close to or higher than score(A->B)."""
        return F.relu(score_ba - score_ab + self.antisym_margin).mean()

    def forward(
        self,
        score_matrix: torch.Tensor,
        score_hardneg: torch.Tensor | None,
        log_sigma_a: torch.Tensor,
        log_sigma_b: torch.Tensor,
        score_ab: torch.Tensor,
        score_ba: torch.Tensor | None,
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

        stats["loss_total"] = total.item()
        return total, stats
