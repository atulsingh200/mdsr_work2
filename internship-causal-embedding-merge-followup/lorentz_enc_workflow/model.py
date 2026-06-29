"""LorentzBiEncoder: BGE-M3 backbone with space and time projection heads.

Produces two outputs per text:
  space: (B, space_dim)  L2-normalised — semantic/topical similarity
  time:  (B, time_dim)   LayerNorm'd, unbounded — causal precedence

Total output dim = space_dim + time_dim (default 1004+20 = 1024 = hidden_size).
A single shared encoder is used for both anchors and positives; asymmetry
comes entirely from the time component in the scoring function.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Tuple


class LorentzBiEncoder(nn.Module):

    def __init__(
        self,
        backbone_name: str = "BAAI/bge-m3",
        space_dim: int = 1004,
        time_dim: int = 20,
        space_hidden: int = 1024,
        time_hidden: int = 512,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.pooling = pooling
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.hidden_size = hidden

        self.space_head = nn.Sequential(
            nn.Linear(hidden, space_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(space_hidden, space_dim),
        )
        self.time_head = nn.Sequential(
            nn.Linear(hidden, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, time_dim),
            nn.LayerNorm(time_dim),
        )

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self._pool(out.last_hidden_state, attention_mask)
        space = F.normalize(self.space_head(h), dim=-1, p=2)
        time = self.time_head(h)
        return space, time

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        self.eval()
        space, time = self.forward(input_ids, attention_mask)
        return {"space": space, "time": time}


def get_tokenizer(backbone_name: str):
    return AutoTokenizer.from_pretrained(backbone_name)


def tokenize(tokenizer, texts: list[str], max_length: int, device: torch.device):
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
