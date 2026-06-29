"""Generic torch Dataset wrapper over any `followup_data.BaseDataset`.

Materializes a `BaseDataset` (which yields `PairExample`) into a list of
(anchor, positive) string pairs. The `collate_fn` factory takes a tokenizer
and produces a function that turns a batch of pairs into the four tensors
the bi-encoder expects (anchor_ids, anchor_mask, positive_ids, positive_mask).

Both pieces are dataset-agnostic — they work on any registered loader
(`aep_causal`, `qrecc`, `clariq`, ...). Only the abstract PairExample interface
is assumed.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

from followup_data.base import BaseDataset, PairExample

from ..model import tokenize_texts


class PairDataset(Dataset):
    """Torch Dataset that exposes (anchor, positive) string tuples.

    Args:
        source:       Either a BaseDataset (which is iterated) or any iterable
                      of PairExample (already materialized).
        max_examples: Optional cap on number of examples (useful for smoke tests).
    """

    def __init__(
        self,
        source: BaseDataset | Iterable[PairExample],
        max_examples: int | None = None,
    ) -> None:
        self.pairs: list[tuple[str, str]] = []
        for ex in source:
            if max_examples is not None and len(self.pairs) >= max_examples:
                break
            anchor = (ex.anchor or "").strip()
            positive = (ex.positive or "").strip()
            if not anchor or not positive:
                continue
            self.pairs.append((anchor, positive))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def make_collate_fn(
    tokenizer,
    max_length: int,
    anchor_prefix: str = "",
    positive_prefix: str = "",
) -> Callable[[list[tuple[str, str]]], tuple[torch.Tensor, ...]]:
    """Build a collate function that tokenizes (anchor, positive) batches.

    The returned function maps a list of (anchor_text, positive_text) tuples
    to a tuple of four tensors:
        (anchor_input_ids, anchor_attention_mask,
         positive_input_ids, positive_attention_mask)

    Args:
        tokenizer:       HuggingFace tokenizer.
        max_length:      Truncation length per side.
        anchor_prefix:   Optional string prepended to anchor texts.
        positive_prefix: Optional string prepended to positive texts.
    """

    def collate(batch: list[tuple[str, str]]) -> tuple[torch.Tensor, ...]:
        anchors, positives = zip(*batch)
        a_texts = [anchor_prefix + a for a in anchors] if anchor_prefix else list(anchors)
        p_texts = [positive_prefix + p for p in positives] if positive_prefix else list(positives)
        a_ids, a_mask = tokenize_texts(tokenizer, a_texts, max_length)
        p_ids, p_mask = tokenize_texts(tokenizer, p_texts, max_length)
        return a_ids, a_mask, p_ids, p_mask

    return collate
