"""Contrastive losses for bi-encoder training.

The default is InfoNCE (NT-Xent) with in-batch negatives: for each anchor a_i
in a batch of N pairs, the positive is p_i and all other p_j (j != i) act
as negatives. The loss pushes a_i towards p_i and away from every p_j.

This is the asymmetric anchor → positive direction only; the symmetric
variant adds the reverse direction term, which is not used here because
the relation we want to learn is directional by design.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """In-batch InfoNCE loss with optional learnable temperature.

    For an N×N similarity matrix S (anchors × positives), the loss is

        L = (1/N) Σ_i −log( exp(S_ii / τ) / Σ_j exp(S_ij / τ) )

    where τ is the temperature. Temperature is stored as log(τ) for
    unconstrained optimization and clamped on access.

    Args:
        temperature_init: Initial temperature value (default 0.05).
        learnable:        If True, temperature is an nn.Parameter.
        temperature_min:  Lower clamp on temperature.
        temperature_max:  Upper clamp on temperature.
    """

    def __init__(
        self,
        temperature_init: float = 0.05,
        learnable: bool = True,
        temperature_min: float = 0.01,
        temperature_max: float = 0.5,
    ) -> None:
        super().__init__()
        log_tau = torch.tensor(temperature_init).log()
        if learnable:
            self.log_temperature = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_temperature", log_tau)
        self.learnable = learnable
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    def forward(
        self,
        anchor_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute InfoNCE loss + diagnostic stats.

        Args:
            anchor_embeds:   (B, D) L2-normalized.
            positive_embeds: (B, D) L2-normalized.

        Returns:
            loss:  scalar tensor.
            stats: dict with temperature, mean positive/negative similarity,
                   and the similarity gap (useful for logging).
        """
        sim = anchor_embeds @ positive_embeds.T            # (B, B)
        tau = self.temperature
        logits = sim / tau

        batch_size = anchor_embeds.size(0)
        labels = torch.arange(batch_size, device=anchor_embeds.device)
        loss = nn.functional.cross_entropy(logits, labels)

        with torch.no_grad():
            pos_sim = sim.diag().mean().item()
            off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=sim.device)
            neg_sim = sim[off_diag].mean().item()

        stats = {
            "temperature": tau.item(),
            "mean_pos_sim": pos_sim,
            "mean_neg_sim": neg_sim,
            "sim_gap": pos_sim - neg_sim,
        }
        return loss, stats
