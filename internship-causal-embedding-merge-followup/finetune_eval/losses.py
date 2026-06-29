"""InfoNCE loss extended with explicit hard negatives.

For each anchor a_i in a batch of B pairs, the candidate set is:

    [ p_0, p_1, ..., p_{B-1},   n_i_1, n_i_2, ..., n_i_K ]
       ↑                            ↑
       in-batch positives           K anchor-specific hard negatives
       (the correct one is p_i)     (semantically similar to p_i but ≠ p_i)

Anchor i's softmax target is index i (its own positive in the in-batch
block). All other in-batch positives plus the K hard negatives contribute
to the denominator → harder, more informative gradients than vanilla
in-batch InfoNCE.

Returns the standard InfoNCE cross-entropy and diagnostic similarity stats.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HardNegativeInfoNCELoss(nn.Module):
    """InfoNCE with B-1 in-batch + K explicit hard negatives per anchor."""

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
        anchor_embeds: torch.Tensor,         # (B, D)
        positive_embeds: torch.Tensor,       # (B, D)
        hardneg_embeds: torch.Tensor | None, # (B, K, D) or None
    ) -> tuple[torch.Tensor, dict]:
        B = anchor_embeds.size(0)
        device = anchor_embeds.device
        tau = self.temperature

        # In-batch similarity matrix: (B, B)
        sim_pos = anchor_embeds @ positive_embeds.T

        if hardneg_embeds is None or hardneg_embeds.numel() == 0:
            logits = sim_pos / tau
            labels = torch.arange(B, device=device)
            loss = nn.functional.cross_entropy(logits, labels)
            with torch.no_grad():
                pos_sim = sim_pos.diag().mean().item()
                off = ~torch.eye(B, dtype=torch.bool, device=device)
                neg_sim = sim_pos[off].mean().item()
                hn_sim = 0.0
        else:
            # Anchor-specific hard negatives: anchor_i · hardneg_i_k → (B, K)
            sim_hn = (anchor_embeds.unsqueeze(1) * hardneg_embeds).sum(dim=-1)
            # Concatenate so each row has B in-batch + K hard-neg scores.
            logits = torch.cat([sim_pos, sim_hn], dim=1) / tau           # (B, B + K)
            labels = torch.arange(B, device=device)                       # target = own positive
            loss = nn.functional.cross_entropy(logits, labels)
            with torch.no_grad():
                pos_sim = sim_pos.diag().mean().item()
                off = ~torch.eye(B, dtype=torch.bool, device=device)
                neg_sim = sim_pos[off].mean().item()
                hn_sim = sim_hn.mean().item()

        stats = {
            "temperature":     tau.item(),
            "mean_pos_sim":    pos_sim,
            "mean_inbatch_neg": neg_sim,
            "mean_hardneg_sim": hn_sim,
            "pos_minus_hn":    pos_sim - hn_sim,
        }
        return loss, stats
