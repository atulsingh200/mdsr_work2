"""Cross-encoder for causal pair scoring.

Architecture (vs. the two-tower BiEncoder):

    [CLS] anchor [SEP] positive [SEP]   ─►  BERT  ─►  CLS-vector  ─►  Linear(1)
                                                                       │
                                                              logit s(A,B) ∈ ℝ

A single BERT sees both texts jointly. This permits token-level cross-attention
between anchor and candidate — strictly more expressive than two-tower scoring
(at the cost of O(N · pool_size) BERT forwards at retrieval time vs O(N + pool)
for embedding-based retrievers).

Output is a scalar logit per pair. Train with BCE-with-logits (positive=1,
negative=0). At eval time, the logit (or its sigmoid) is the similarity score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "google-bert/bert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape (B,)."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = outputs.last_hidden_state[:, 0, :]      # (B, H)
        cls = self.dropout(cls)
        return self.classifier(cls).squeeze(-1)        # (B,)


def get_tokenizer(name: str = "google-bert/bert-base-uncased"):
    return AutoTokenizer.from_pretrained(name)


def tokenize_pairs(
    tokenizer,
    text_a: list[str],
    text_b: list[str],
    max_length: int = 256,
) -> dict[str, torch.Tensor]:
    """Tokenize parallel (A, B) text lists into one [CLS] A [SEP] B [SEP] tensor each."""
    enc = tokenizer(
        text_a,
        text_b,
        padding=True,
        truncation="longest_first",
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids"),
    }
