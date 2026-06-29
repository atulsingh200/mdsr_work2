import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Tuple


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: x + Dropout(W2 GELU(W1 LN(x)))."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + h


class TimeHead(nn.Module):
    """Expressive time head: projects pooled rep into a `time_dim` space that
    encodes causal precedence.

    Architecture (deeper than a single MLP so it can model nonlinear temporal
    relations):
        Linear(hidden -> time_hidden) -> GELU -> Dropout
        [ResidualBlock(time_hidden)] x n_layers
        Linear(time_hidden -> time_dim) -> LayerNorm(time_dim)

    The output is layer-normalised (not L2): magnitude carries temporal-order
    information, which the SteepSigmoid scoring term reads as "b is after a".
    """

    def __init__(self, hidden, time_dim, time_hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(hidden, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(time_hidden, dropout) for _ in range(max(0, n_layers))
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(time_hidden, time_dim),
            nn.LayerNorm(time_dim),
        )

    def forward(self, h):
        x = self.input_proj(h)
        for blk in self.blocks:
            x = blk(x)
        return self.output_proj(x)


class LorentzEncoder(nn.Module):
    """
    Untied two-tower Lorentz encoder:
      - Separate anchor_backbone + anchor heads
      - Separate positive_backbone + positive heads

    Space output: L2-normalized (cosine retrieval, symmetric/topical).
    Time output:  deeper MLP, LayerNorm'd (asymmetric causal-ordering signal).

    When space_dim == hidden_dim (no bottleneck), the space head is just an
    L2-norm of the pooled output — identical to a vanilla bi-encoder for
    space-only evaluation.
    """

    def __init__(
        self,
        backbone_name: str = "intfloat/e5-base-v2",
        space_dim: int = 768,
        time_dim: int = 20,
        space_hidden: int = 768,
        time_hidden: int = 256,
        time_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
        tied: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.time_layers = time_layers
        self.tied = tied

        self.anchor_backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.anchor_backbone.config.hidden_size

        def make_space_head():
            if space_dim == hidden:
                return None  # L2-norm pooled output directly
            return nn.Sequential(
                nn.Linear(hidden, space_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(space_hidden, space_dim),
            )

        self.anchor_space_head = make_space_head()
        self.anchor_time_head = TimeHead(hidden, time_dim, time_hidden, time_layers, dropout)

        if tied:
            self.positive_backbone = self.anchor_backbone
            self.positive_space_head = self.anchor_space_head
            self.positive_time_head = self.anchor_time_head
        else:
            self.positive_backbone = AutoModel.from_pretrained(backbone_name)
            self.positive_space_head = make_space_head()
            self.positive_time_head = TimeHead(hidden, time_dim, time_hidden, time_layers, dropout)

    def _pool(self, last_hidden, attention_mask):
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _encode(self, backbone, space_head, time_head, input_ids, attention_mask):
        out = backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self._pool(out.last_hidden_state, attention_mask)
        space = F.normalize(h if space_head is None else space_head(h), dim=-1, p=2)
        time = time_head(h)
        return space, time

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.encode_anchor(input_ids, attention_mask)

    def encode_anchor(self, input_ids, attention_mask, **kwargs):
        return self._encode(self.anchor_backbone, self.anchor_space_head,
                            self.anchor_time_head, input_ids, attention_mask)

    def encode_positive(self, input_ids, attention_mask, **kwargs):
        return self._encode(self.positive_backbone, self.positive_space_head,
                            self.positive_time_head, input_ids, attention_mask)

    # --- fine-grained access for the directional loss ---------------------- #
    def pooled(self, which, input_ids, attention_mask):
        """Return the pooled backbone representation (pre-head) for one tower."""
        bk = self.anchor_backbone if which == "anchor" else self.positive_backbone
        out = bk(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def space_from_pooled(self, which, h):
        head = self.anchor_space_head if which == "anchor" else self.positive_space_head
        return F.normalize(h if head is None else head(h), dim=-1, p=2)

    def time_from_pooled(self, which, h):
        head = self.anchor_time_head if which == "anchor" else self.positive_time_head
        return head(h)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask, **kwargs):
        self.eval()
        s, t = self.encode_anchor(input_ids, attention_mask)
        return {"space": s, "time": t}
