"""Bi-encoder model: two independent encoders for anchor and positive texts.

Both encoders share the same architecture / initialization (a HuggingFace
pretrained encoder, e.g. BAAI/bge-small-en-v1.5) but have untied weights so
they can diverge to capture asymmetric / directional relations:

    anchor_encoder(A)    "what tends to be followed by ..."
    positive_encoder(B)  "what tends to follow ..."

The asymmetric framing matters when the (A, B) relation is non-symmetric —
e.g. follow-up question retrieval, causal prerequisite/next-step linking,
or sequential interactions.

Embeddings are L2-normalized so dot products equal cosine similarities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MeanPooling(nn.Module):
    """Mean pooling over token embeddings, respecting the attention mask."""

    def forward(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeds = model_output.last_hidden_state           # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()             # (B, T, 1)
        summed = (token_embeds * mask).sum(dim=1)               # (B, D)
        counts = mask.sum(dim=1).clamp(min=1e-9)                # (B, 1)
        return summed / counts


class CLSPooling(nn.Module):
    """CLS-token pooling: take the [CLS] position representation."""

    def forward(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        return model_output.last_hidden_state[:, 0]


def get_pooling(strategy: str) -> nn.Module:
    if strategy == "mean":
        return MeanPooling()
    if strategy == "cls":
        return CLSPooling()
    raise ValueError(f"unknown pooling strategy: {strategy!r} (use 'mean' or 'cls')")


class BiEncoder(nn.Module):
    """Two-tower encoder with untied anchor / positive encoders.

    Args:
        model_name:      HuggingFace model id for both towers' initialization.
        pooling_strategy: "mean" or "cls".
        anchor_prefix:   Optional string prepended to anchor texts at tokenize-time
                         (some models like Arctic-Embed require "query: ").
        positive_prefix: Optional string prepended to positive texts.
    """

    def __init__(
        self,
        model_name: str,
        pooling_strategy: str = "mean",
        anchor_prefix: str = "",
        positive_prefix: str = "",
    ) -> None:
        super().__init__()
        self.anchor_encoder = AutoModel.from_pretrained(model_name)
        self.positive_encoder = AutoModel.from_pretrained(model_name)
        self.pooling = get_pooling(pooling_strategy)
        self.anchor_prefix = anchor_prefix
        self.positive_prefix = positive_prefix
        self.hidden_size: int = self.anchor_encoder.config.hidden_size

    def _encode(
        self,
        encoder: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeds = self.pooling(out, attention_mask)
        return nn.functional.normalize(embeds, p=2, dim=1)

    def encode_anchor(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encode(self.anchor_encoder, input_ids, attention_mask)

    def encode_positive(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encode(self.positive_encoder, input_ids, attention_mask)

    def forward(
        self,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of anchor / positive pairs.

        Returns:
            anchor_embeds:   (B, D) L2-normalized
            positive_embeds: (B, D) L2-normalized
        """
        a = self.encode_anchor(anchor_input_ids, anchor_attention_mask)
        p = self.encode_positive(positive_input_ids, positive_attention_mask)
        return a, p


def get_tokenizer(model_name: str):
    """Load the HuggingFace tokenizer for the given backbone."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a list of texts into (input_ids, attention_mask) tensors."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]
