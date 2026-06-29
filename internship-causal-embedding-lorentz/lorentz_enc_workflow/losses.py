"""Three-term loss for LorentzEnc training.

Term 1 — Causal InfoNCE:
    For anchor a_i, positive is b_i.
    Negatives: (B-1) in-batch positives  +  K semantic hard negatives.
    Scores computed with the ASYMMETRIC causal_similarity_matrix.
    Learnable temperature.

Term 2 — Ordering loss on time dimensions:
    Push  (time_pos_k − time_a_k)  > margin   (positive is "after" anchor)
    Push  (time_neg_k − time_a_k)  < −slack    (hard neg is "before" anchor)

Term 3 — Reverse pair penalty:
    For each pair (a_i, b_i) in the batch, penalise high sim(b_i, a_i).
    loss = relu(sim(b_i, a_i) − margin_rev).mean()

Combined: L = λ_nce·L_nce + λ_ord·L_ord + λ_rev·L_rev
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scoring import causal_similarity, causal_similarity_matrix


class LorentzLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        tau_score: float = 10.0,
        temperature_init: float = 0.05,
        temperature_min: float = 0.01,
        temperature_max: float = 0.5,
        lambda_nce: float = 1.0,
        lambda_ord: float = 0.5,
        lambda_rev: float = 1.0,
        ord_margin: float = 0.5,
        ord_slack: float = 0.1,
        margin_rev: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau_score = tau_score
        self.lambda_nce = lambda_nce
        self.lambda_ord = lambda_ord
        self.lambda_rev = lambda_rev
        self.ord_margin = ord_margin
        self.ord_slack = ord_slack
        self.margin_rev = margin_rev
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.log_temperature = nn.Parameter(torch.tensor(temperature_init).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    def forward(
        self,
        space_a: torch.Tensor,   # (B, D_s)
        time_a: torch.Tensor,    # (B, D_t)
        space_p: torch.Tensor,   # (B, D_s)
        time_p: torch.Tensor,    # (B, D_t)
        space_n: torch.Tensor | None,  # (B*K, D_s) or None
        time_n: torch.Tensor | None,   # (B*K, D_t) or None
    ) -> tuple[torch.Tensor, dict]:
        B = space_a.size(0)
        tau = self.temperature
        device = space_a.device

        # ---- Term 1: Causal InfoNCE ----
        # In-batch similarity matrix (B, B): sim(a_i, p_j)
        sim_inbatch = causal_similarity_matrix(
            space_a, time_a, space_p, time_p,
            alpha=self.alpha, beta=self.beta, tau=self.tau_score,
        )

        if space_n is not None and space_n.numel() > 0:
            K = space_n.size(0) // B
            space_n_3d = space_n.view(B, K, -1)
            time_n_3d = time_n.view(B, K, -1)
            # (B, K): sim(a_i, hn_k) for each anchor's K hard negs
            diff_t = time_n_3d - time_a.unsqueeze(1)               # (B, K, D_t)
            hn_time = torch.sigmoid(self.tau_score * diff_t).mean(-1)  # (B, K)
            hn_cos = (space_a.unsqueeze(1) * space_n_3d).sum(-1)       # (B, K)
            sim_hn = self.alpha * hn_cos + self.beta * hn_time          # (B, K)
            logits = torch.cat([sim_inbatch, sim_hn], dim=1) / tau      # (B, B+K)
        else:
            K = 0
            sim_hn = None
            logits = sim_inbatch / tau                                   # (B, B)

        labels = torch.arange(B, device=device)
        l_nce = F.cross_entropy(logits, labels)

        # ---- Term 2: Ordering loss ----
        # Positive: want time_p_k − time_a_k > ord_margin for all k
        diff_pos = time_p - time_a                                  # (B, D_t)
        l_ord_pos = F.relu(self.ord_margin - diff_pos).mean()

        l_ord_neg = torch.zeros(1, device=device)
        if space_n is not None and space_n.numel() > 0:
            # Hard neg: want time_neg_k − time_a_k < −slack for all k
            diff_neg = time_n_3d - time_a.unsqueeze(1)             # (B, K, D_t)
            l_ord_neg = F.relu(self.ord_slack + diff_neg).mean()

        l_ord = l_ord_pos + l_ord_neg

        # ---- Term 3: Reverse pair penalty ----
        # sim(b_i, a_i) should be low — penalise when it exceeds margin_rev
        sim_rev = causal_similarity(
            space_p, time_p, space_a, time_a,
            alpha=self.alpha, beta=self.beta, tau=self.tau_score,
        )                                                           # (B,)
        l_rev = F.relu(sim_rev - self.margin_rev).mean()

        total = (
            self.lambda_nce * l_nce
            + self.lambda_ord * l_ord
            + self.lambda_rev * l_rev
        )

        with torch.no_grad():
            pos_sim = sim_inbatch.diag().mean().item()
            off = ~torch.eye(B, dtype=torch.bool, device=device)
            inbatch_neg_sim = sim_inbatch[off].mean().item()
            hn_sim_val = sim_hn.mean().item() if sim_hn is not None else 0.0

        stats = {
            "temperature":      tau.item(),
            "loss_nce":         l_nce.item(),
            "loss_ord":         l_ord.item(),
            "loss_rev":         l_rev.item(),
            "mean_pos_sim":     pos_sim,
            "mean_inbatch_neg": inbatch_neg_sim,
            "mean_hardneg_sim": hn_sim_val,
            "mean_rev_sim":     sim_rev.mean().item(),
        }
        return total, stats
