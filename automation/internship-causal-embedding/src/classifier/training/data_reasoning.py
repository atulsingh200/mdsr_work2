"""Dataset and collate classes for the reasoning-augmented classifier.

ReasoningPairDataset extends PairDataset to also expose the `explanation` field.
ReasoningCollate extends Collate to tokenize explanations into enc_reason.

When the explanation is absent (val/test splits), enc_reason is set to None
so the training script can skip the reasoning loss branch cleanly.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .data import Collate, PairDataset


class ReasoningPairDataset(PairDataset):
    """Adds `explanation` field on top of standard PairDataset."""

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        item["explanation"] = self.rows[i].get("explanation", "")
        return item


class ReasoningCollate(Collate):
    """Tokenizes text_1, text_2, and explanation (when present)."""

    def __call__(self, batch: list[dict]) -> dict:
        result = super().__call__(batch)

        explanations = [b.get("explanation", "") for b in batch]
        has_any = any(e for e in explanations)

        if has_any:
            # Replace empty strings with a single space so the tokenizer
            # doesn't choke — these rows simply won't contribute to the loss
            # because we only apply reason_loss when the full batch has explanations.
            texts = [e if e else " " for e in explanations]
            enc_reason = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            result["enc_reason"] = enc_reason
        else:
            result["enc_reason"] = None

        return result
