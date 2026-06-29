"""Dataset utilities for CDE training.

Reads (anchor, positive) JSONL pairs and a pre-mined hard_negatives.npy
matrix of shape (N, K) — same format as finetune_eval/data.py — and
yields triples for the collate function.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_pairs(path: Path, cap: int | None = None, seed: int = 0) -> list[tuple[str, str]]:
    """Read (anchor, positive) pairs from JSONL.

    The file must have 'anchor' and 'positive' keys per row.
    """
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
    if cap is not None and len(pairs) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=cap, replace=False)
        idx.sort()
        pairs = [pairs[i] for i in idx]
    return pairs


class TripleDataset(Dataset):
    """Yields (anchor_text, positive_text, [hard_neg_text...]).

    hard_negatives is an (N, K) int matrix indexing into the full
    positives list.  Splitting the active rows via active_indices
    does not invalidate the precomputed mining indices.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        hard_negatives: np.ndarray | None,
        active_indices: list[int] | None = None,
    ) -> None:
        self.anchors = [a for a, _ in pairs]
        self.positives = [p for _, p in pairs]
        if hard_negatives is not None and hard_negatives.shape[0] != len(pairs):
            raise ValueError(
                f"hard_negatives rows {hard_negatives.shape[0]} != pairs {len(pairs)}"
            )
        self.hard_negatives = hard_negatives
        self.active_indices = (
            list(active_indices) if active_indices is not None
            else list(range(len(pairs)))
        )

    def __len__(self) -> int:
        return len(self.active_indices)

    @property
    def n_hard_negatives(self) -> int:
        return 0 if self.hard_negatives is None else int(self.hard_negatives.shape[1])

    def __getitem__(self, idx: int) -> tuple[str, str, list[str]]:
        real = self.active_indices[idx]
        hardnegs: list[str] = []
        if self.hard_negatives is not None:
            hardnegs = [self.positives[int(j)] for j in self.hard_negatives[real]]
        return self.anchors[real], self.positives[real], hardnegs


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(tokenizer, texts: list[str], max_length: int):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def make_collate_fn(tokenizer, max_length: int):
    """Collate triples into (a_ids, a_mask, p_ids, p_mask, neg_ids, neg_mask).

    neg_ids / neg_mask are (B*K, L) — flattened hard negatives.
    If K==0 both are empty tensors.
    """
    def collate(batch):
        anchors  = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        hardnegs_rows = [b[2] for b in batch]
        K = len(hardnegs_rows[0]) if hardnegs_rows else 0

        a_ids, a_mask = tokenize_batch(tokenizer, anchors, max_length)
        p_ids, p_mask = tokenize_batch(tokenizer, positives, max_length)

        if K == 0:
            empty = torch.zeros((0, 1), dtype=torch.long)
            return a_ids, a_mask, p_ids, p_mask, empty, empty

        flat_negs = [t for row in hardnegs_rows for t in row]
        n_ids, n_mask = tokenize_batch(tokenizer, flat_negs, max_length)
        return a_ids, a_mask, p_ids, p_mask, n_ids, n_mask

    return collate
