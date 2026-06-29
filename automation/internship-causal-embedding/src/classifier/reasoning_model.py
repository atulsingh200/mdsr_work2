"""Reasoning-augmented directional classifier.

Architecture:
  - Two trainable encoders (src, tgt) with dual-encoder feature fusion [A;B;A-B;A*B].
  - One frozen encoder (reason_encoder) that embeds ground-truth explanations.
  - Shared MLP backbone -> two branches:
      * Prediction head: Linear(H, 1) -> BCEWithLogitsLoss
      * Reasoning head:  Linear(H, d) -> CosineEmbeddingLoss vs frozen explanation emb

At train time both losses are combined: total = dir_loss + w * reason_loss.
At val/test time enc_reason=None -> reasoning head is skipped; only dir_logit returned.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .heads import MLPHead, build_head


class ReasoningClassifier(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        head_cfg: dict | None = None,
    ):
        super().__init__()
        # Trainable dual encoders
        self.src = AutoModel.from_pretrained(base_model_name)
        self.tgt = AutoModel.from_pretrained(base_model_name)
        d = self.src.config.hidden_size
        self.hidden_size = d

        # Frozen encoder for explanation embeddings (shares weights architecture,
        # but is fully detached from the optimizer).
        self.reason_encoder = AutoModel.from_pretrained(base_model_name)
        for p in self.reason_encoder.parameters():
            p.requires_grad = False

        # Build shared MLP backbone — must be mlp type to allow branching.
        cfg = dict(head_cfg or {})
        if cfg.get("head_type", "linear") != "mlp":
            raise ValueError(
                "ReasoningClassifier requires --head-type mlp "
                "(backbone must have a penultimate hidden layer to branch from)"
            )
        self.head_cfg = cfg

        full_head: MLPHead = build_head(cfg, in_dim=4 * d)  # type: ignore[assignment]
        if not hasattr(full_head, "net"):
            raise ValueError("Expected MLPHead with .net attribute")

        # Split MLPHead into backbone (all layers except final LayerNorm+Linear)
        # and final projection, mirroring the tier-aux pattern in DirectionalClassifier.
        modules = list(full_head.net)
        # Last 2 modules: LayerNorm + Linear(H, 1)
        self.backbone = nn.Sequential(*modules[:-2])
        self.pred_final = nn.Sequential(*modules[-2:])  # -> (B, 1)
        last_hidden: int = modules[-1].in_features      # e.g. 128

        # Reasoning projection: hidden -> explanation embedding space
        self.reason_proj = nn.Linear(last_hidden, d)
        nn.init.xavier_uniform_(self.reason_proj.weight)
        nn.init.zeros_(self.reason_proj.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _encode(encoder: nn.Module, enc: dict) -> torch.Tensor:
        out = encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return nn.functional.normalize(cls, p=2, dim=-1)

    def forward(
        self,
        enc1: dict,
        enc2: dict,
        enc_reason: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            dir_logit      (B,)          — raw logit for BCE loss
            reason_pred    (B, d) | None — projected hidden for cosine loss
            reason_target  (B, d) | None — frozen explanation embedding (detached)
        """
        a = self._encode(self.src, enc1)
        b = self._encode(self.tgt, enc2)
        feat = torch.cat([a, b, a - b, a * b], dim=-1)  # (B, 4d)

        hidden = self.backbone(feat)                     # (B, H)
        dir_logit = self.pred_final(hidden).squeeze(-1)  # (B,)

        if enc_reason is not None:
            with torch.no_grad():
                reason_target = self._encode(self.reason_encoder, enc_reason)
            reason_pred = nn.functional.normalize(self.reason_proj(hidden), p=2, dim=-1)
            return dir_logit, reason_pred, reason_target.detach()

        return dir_logit, None, None
