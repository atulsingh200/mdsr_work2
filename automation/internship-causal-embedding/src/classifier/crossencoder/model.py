"""Cross-encoder directional classifier.

A single pretrained encoder jointly attends over  text_1 [SEP] text_2  and a
linear head maps the [CLS] representation to one directional logit:

    P(text_1 causally precedes text_2) = sigmoid(logit)

BCEWithLogitsLoss is applied externally (head returns a raw logit), matching the
convention of the dual-encoder heads in src/classifier/heads.py.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class CrossEncoderClassifier(nn.Module):
    def __init__(self, base_model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        d = self.encoder.config.hidden_size
        self.hidden_size = d
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, enc: dict) -> torch.Tensor:
        """enc = joint tokenization of (text_1, text_2). Returns (B,) logits."""
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] pooling
        return self.classifier(self.dropout(cls)).squeeze(-1)
