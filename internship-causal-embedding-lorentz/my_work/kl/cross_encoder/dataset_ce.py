"""Pair-classification dataset for the cross-encoder.

Each training example is a single (anchor, candidate, label) tuple where label
is 1 for the true positive and 0 for a negative. For each positive pair, we
emit:

    1 × (anchor_i, positive_i,                                       label=1)
    K × (anchor_i, positives[hard_negatives[i, k]],                  label=0)

so a single base pair expands to (1 + K) training examples. With K=4 this turns
5,487 anchor/positive pairs into 5,487 × 5 = 27,435 training rows.

The collate function batches these into a single forward pass through BERT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = (row.get("anchor") or "").strip()
            p = (row.get("positive") or "").strip()
            if a and p:
                pairs.append((a, p))
    return pairs


class PairLabelDataset(Dataset):
    """Flattened (anchor, candidate, label) view of pairs + hard negatives."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        hard_negatives: np.ndarray | None,
    ) -> None:
        self.anchors = [a for a, _ in pairs]
        self.positives = [p for _, p in pairs]
        self.hard_negatives = hard_negatives          # (N, K) or None

        rows: list[tuple[int, int, int]] = []         # (anchor_idx, cand_idx, label)
        N = len(pairs)
        for i in range(N):
            rows.append((i, i, 1))                    # positive
            if hard_negatives is not None:
                for j in hard_negatives[i]:
                    rows.append((i, int(j), 0))       # hard negative
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        anchor_i, cand_i, label = self.rows[idx]
        return self.anchors[anchor_i], self.positives[cand_i], label


def make_collate_fn(tokenizer, max_length: int):
    """Collate (a, b, label) triples into a single tokenized batch."""

    def collate(batch: list[tuple[str, str, int]]):
        anchors = [b[0] for b in batch]
        candidates = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float)
        enc = tokenizer(
            anchors,
            candidates,
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
            "labels": labels,
        }

    return collate
