"""Dataset and collation for directional classification training.

Reads JSONL files directly (each row: text_1, text_2, label, tier_1, tier_2,
sub_1, sub_2, url_1, url_2). This avoids coupling the training loop to the
followup_data loader abstraction while keeping the same data format.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """In-memory dataset of labelled text pairs from a JSONL file."""

    def __init__(self, path: Path):
        self.rows: list[dict] = []
        with open(path) as f:
            for ln in f:
                self.rows.append(json.loads(ln))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {
            "text_1": r["text_1"],
            "text_2": r["text_2"],
            "label": float(r["label"]),
            "tier_1": r.get("tier_1") or 1,
            "tier_2": r.get("tier_2") or 1,
            "idx": i,
        }


class Collate:
    """Tokenize and batch text pairs for the directional classifier."""

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict:
        t1 = [b["text_1"] for b in batch]
        t2 = [b["text_2"] for b in batch]
        enc1 = self.tok(
            t1, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        enc2 = self.tok(
            t2, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        # Tier labels as 0-indexed tensors (tier 1..15 -> index 0..14)
        tier_1 = torch.tensor([b["tier_1"] - 1 for b in batch], dtype=torch.long)
        tier_2 = torch.tensor([b["tier_2"] - 1 for b in batch], dtype=torch.long)
        return {
            "enc1": enc1,
            "enc2": enc2,
            "labels": labels,
            "tier_1": tier_1,
            "tier_2": tier_2,
            "idx": [b["idx"] for b in batch],
        }
