"""Directional classifier: dual-encoder + classification head.

Architecture:
  - Two pretrained encoders (source + target, untied weights).
  - CLS-token pooling on each, followed by L2 normalization.
  - Concatenate [A; B; A-B; A*B] -> 4d feature vector.
  - Pluggable classification head (Linear or MLP) -> single logit.
  - Optional auxiliary tier-prediction heads branching off the MLP's
    penultimate hidden layer (regularizer for tier-aware embeddings).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .heads import build_head


class DirectionalClassifier(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        head_cfg: dict | None = None,
        n_tiers: int = 0,
    ):
        super().__init__()
        self.src = AutoModel.from_pretrained(base_model_name)
        self.tgt = AutoModel.from_pretrained(base_model_name)
        d = self.src.config.hidden_size  # 384 for bge-small
        self.hidden_size = d

        # Pluggable head: defaults to LinearHead.
        self.head_cfg = head_cfg or {"head_type": "linear"}
        self.head = build_head(self.head_cfg, in_dim=4 * d)

        # Optional auxiliary tier-prediction heads (regularizer).
        # Branch off the MLP's penultimate hidden layer.
        self.n_tiers = n_tiers
        if n_tiers > 0:
            if not hasattr(self.head, "net"):
                raise ValueError(
                    "--tier-aux-weight requires --head-type mlp "
                    "(cannot split a linear head)"
                )
            # Split MLP into backbone (up to last hidden) + final projection.
            modules = list(self.head.net)
            # Last 2 modules are LayerNorm + Linear(hidden, 1)
            self.head_backbone = nn.Sequential(*modules[:-2])
            self.head_final = nn.Sequential(*modules[-2:])
            last_hidden = modules[-1].in_features  # e.g. 128
            self.tier_head_1 = nn.Linear(last_hidden, n_tiers)
            self.tier_head_2 = nn.Linear(last_hidden, n_tiers)
        else:
            self.head_backbone = None
            self.head_final = None
            self.tier_head_1 = None
            self.tier_head_2 = None

    @staticmethod
    def _encode(encoder: nn.Module, enc: dict) -> torch.Tensor:
        out = encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]             # CLS pooling
        cls = nn.functional.normalize(cls, p=2, dim=-1)  # L2 normalize
        return cls

    def forward(
        self, enc1: dict, enc2: dict
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        a = self._encode(self.src, enc1)       # (B, d)
        b = self._encode(self.tgt, enc2)       # (B, d)
        feat = torch.cat([a, b, a - b, a * b], dim=-1)  # (B, 4d)

        if self.head_backbone is not None:
            hidden = self.head_backbone(feat)                    # (B, last_hidden)
            dir_logit = self.head_final(hidden).squeeze(-1)      # (B,)
            tier_logits_1 = self.tier_head_1(hidden)             # (B, n_tiers)
            tier_logits_2 = self.tier_head_2(hidden)             # (B, n_tiers)
        else:
            dir_logit = self.head(feat)                          # (B,)
            tier_logits_1 = None
            tier_logits_2 = None

        return dir_logit, tier_logits_1, tier_logits_2
