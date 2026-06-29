"""Triple dataset wrapper (anchor, positive, [hard_negatives]).

Reads (anchor, positive) JSONL pairs and an aligned `hard_negatives.npy`
matrix of shape `(N, K)` whose entry `[i, k]` is the index of the k-th
hard negative *positive* for anchor i (i.e. the `k`-th nearest other
positive in semantic-similarity space).

Yields tuples `(anchor_text, positive_text, [hardneg_text_1, ..., hardneg_text_K])`
which the collate function tokenizes into encoder-ready tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from biencoder.model import tokenize_texts  # noqa: E402


def load_pairs(path: Path, cap: int | None = None, seed: int = 0) -> list[tuple[str, str]]:
    """Read (anchor, positive) pairs from JSONL.

    If `cap` is set and the file is larger, take a deterministic random sample.
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
    """Each item: (anchor_text, positive_text, list[hardneg_text]).

    hard_negatives is an (N, K) int matrix whose entries index into the
    *full* positives list (not the active subset) so that splitting
    train/val by index doesn't invalidate any precomputed mining.

    Args:
        pairs:           full set of (anchor, positive) pairs.
        hard_negatives:  (N, K) int indices into pairs (or None).
        active_indices:  rows to iterate over; defaults to all rows. Use
                         this to expose a train/val subset without trimming
                         the positives pool — hard-neg indices stay valid.
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
        self.hard_negatives = hard_negatives  # (N, K) or None
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
        anchor = self.anchors[real]
        positive = self.positives[real]
        if self.hard_negatives is None:
            hardnegs: list[str] = []
        else:
            hardnegs = [self.positives[int(j)] for j in self.hard_negatives[real]]
        return anchor, positive, hardnegs


def make_collate_fn(tokenizer, max_length: int):
    """Build a collate function that tokenizes triples into tensors.

    Returns a function producing six tensors:
        anchor_ids, anchor_mask           — (B, T_a)
        positive_ids, positive_mask       — (B, T_p)
        hardneg_ids, hardneg_mask         — (B*K, T_n)  flattened; K = n_hard_negatives
    If K == 0, the hardneg tensors are empty.
    """

    def collate(batch):
        anchors = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        hardnegs_per_row = [b[2] for b in batch]
        K = len(hardnegs_per_row[0]) if hardnegs_per_row else 0

        a_ids, a_mask = tokenize_texts(tokenizer, anchors, max_length)
        p_ids, p_mask = tokenize_texts(tokenizer, positives, max_length)

        if K == 0:
            empty = torch.zeros((0, 1), dtype=torch.long)
            return a_ids, a_mask, p_ids, p_mask, empty, empty

        flat_negs = [t for row in hardnegs_per_row for t in row]   # B*K texts
        n_ids, n_mask = tokenize_texts(tokenizer, flat_negs, max_length)
        return a_ids, a_mask, p_ids, p_mask, n_ids, n_mask

    return collate
