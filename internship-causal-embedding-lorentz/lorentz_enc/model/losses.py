import torch
import torch.nn as nn
import torch.nn.functional as F
from .scoring import causal_similarity_matrix, steep_sigmoid_mean


def info_nce_loss(
    space_a, time_a, space_pos, time_pos, space_neg, time_neg,
    neg_mask, alpha=1.0, beta=0.5, tau_score=10.0,
    tau_nce=0.05, margin=0.02,
):
    """InfoNCE with additive margin on positive and explicit hard negatives."""
    B = space_a.size(0)
    N = neg_mask.size(1)
    space_neg = space_neg.view(B, N, -1)
    time_neg = time_neg.view(B, N, -1)

    # Positive score (B,)
    pos_score = (
        alpha * (space_a * space_pos).sum(-1)
        + beta * steep_sigmoid_mean(time_a, time_pos, tau=tau_score)
    )

    # Negative scores (B, N)
    diff_t = time_neg - time_a.unsqueeze(1)
    neg_time = torch.sigmoid(tau_score * diff_t).mean(-1)
    neg_cos = (space_a.unsqueeze(1) * space_neg).sum(-1)
    neg_score = alpha * neg_cos + beta * neg_time

    # In-batch negatives (B, B), diagonal removed
    in_batch = causal_similarity_matrix(
        space_a, time_a, space_pos, time_pos,
        alpha=alpha, beta=beta, tau=tau_score,
    )
    diag = torch.eye(B, dtype=torch.bool, device=space_a.device)
    in_batch = in_batch.masked_fill(diag, float("-inf"))

    pos_logit = (pos_score - margin) / tau_nce
    neg_logit = neg_score / tau_nce
    neg_logit = neg_logit.masked_fill(~neg_mask, float("-inf"))
    in_batch_logit = in_batch / tau_nce

    logits = torch.cat([pos_logit.unsqueeze(1), neg_logit, in_batch_logit], dim=1)
    target = torch.zeros(B, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def ordering_loss(
    time_a, time_pos, time_neg, neg_mask,
    margin=0.5, slack=0.1,
):
    """Push t_pos > t_a by margin; push t_neg < t_a by slack."""
    B, D = time_a.shape
    N = neg_mask.size(1)

    diff_pos = time_pos - time_a
    loss_pos = F.relu(margin - diff_pos).mean(-1)

    diff_neg = time_neg.view(B, N, D) - time_a.unsqueeze(1)
    per_neg = F.relu(slack + diff_neg).mean(-1)
    per_neg = per_neg.masked_fill(~neg_mask, 0.0)
    n_valid = neg_mask.sum(-1).clamp(min=1).float()
    loss_neg = per_neg.sum(-1) / n_valid

    return (loss_pos + loss_neg).mean()


def cawai_regulariser(space_a: torch.Tensor, space_frozen_a: torch.Tensor):
    """In-batch InfoNCE pulling space rep toward frozen-encoder twin."""
    logits = space_a @ space_frozen_a.t()
    target = torch.arange(space_a.size(0), device=space_a.device)
    return F.cross_entropy(logits / 0.05, target)


def mds_distillation_loss(
    space_pred: torch.Tensor, time_pred: torch.Tensor,
    space_target: torch.Tensor, time_target: torch.Tensor,
):
    return (
        F.mse_loss(space_pred, space_target) +
        F.mse_loss(time_pred, time_target)
    )


class Stage3Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        space_a, time_a, space_pos, time_pos, space_neg, time_neg,
        space_frozen_a, neg_mask,
    ):
        l_nce = info_nce_loss(
            space_a, time_a, space_pos, time_pos, space_neg, time_neg,
            neg_mask,
            alpha=self.cfg.scoring.alpha, beta=self.cfg.scoring.beta,
            tau_score=self.cfg.scoring.tau,
            tau_nce=self.cfg.stage3.nce_temperature, margin=self.cfg.stage3.nce_margin,
        )
        l_ord = ordering_loss(
            time_a, time_pos, time_neg, neg_mask,
            margin=self.cfg.stage3.ord_margin, slack=self.cfg.stage3.ord_slack,
        )
        l_reg = cawai_regulariser(space_a, space_frozen_a)
        total = (
            self.cfg.stage3.lambda_nce * l_nce
            + self.cfg.stage3.lambda_ord * l_ord
            + self.cfg.stage3.lambda_reg * l_reg
        )
        return total, {"nce": l_nce.item(), "ord": l_ord.item(), "reg": l_reg.item()}
