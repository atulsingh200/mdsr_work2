"""Dataset + collation for contrastive cross-encoder training.

Each item in GroupedDataset is one complete group:
  - 1 positive pair  (neg_type == "positive", always at index 0)
  - 1-5 negative pairs (hard / easy / reverse)

Group sizes are VARIABLE (2-6): the strict tier<tier(x) constraint on hard/easy
negatives means sparse lower tiers yield fewer negatives. ContrastiveCollate
flattens B groups into sum(group_sizes) jointly-tokenized pairs, and the
group_sizes tensor lets the InfoNCE loss reconstruct per-group logit vectors.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class GroupedDataset(Dataset):
    """Each __getitem__ returns one complete group (list of dicts, positive first)."""

    def __init__(self, path: Path):
        path = Path(path)
        rows_by_group: OrderedDict[str, list[dict]] = OrderedDict()
        with open(path) as f:
            for ln in f:
                r = json.loads(ln)
                gid = r["group_id"]
                if gid not in rows_by_group:
                    rows_by_group[gid] = []
                rows_by_group[gid].append(r)

        self.groups: list[list[dict]] = []
        for gid, group_rows in rows_by_group.items():
            # Sort: positive first, then negatives (stable within each class)
            sorted_rows = sorted(
                group_rows,
                key=lambda r: (0 if r["neg_type"] == "positive" else 1),
            )
            # Validate
            n_pos = sum(1 for r in sorted_rows if r["neg_type"] == "positive")
            assert n_pos == 1, f"Group {gid} has {n_pos} positives (expected 1)"
            self.groups.append(sorted_rows)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> list[dict]:
        return self.groups[i]


class ContrastiveCollate:
    """Flatten B groups into a single batch of sum(group_sizes) tokenized pairs.

    Returns (let N = sum(group_sizes)):
      enc           – tokenizer output for all N pairs
      group_sizes   – (B,) int tensor of per-group sizes (variable)
      labels        – (N,) float tensor (1.0 for positive, 0.0 for negatives)
      tier_1        – (N,) long tensor
      tier_2        – (N,) long tensor
    """

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[list[dict]]) -> dict:
        all_rows = [row for group in batch for row in group]
        t1 = [r["text_1"] for r in all_rows]
        t2 = [r["text_2"] for r in all_rows]
        enc = self.tok(
            t1, t2,
            padding=True,
            truncation="longest_first",
            max_length=self.max_len,
            return_tensors="pt",
        )
        group_sizes = torch.tensor([len(g) for g in batch], dtype=torch.long)
        labels = torch.tensor([float(r["label"]) for r in all_rows], dtype=torch.float32)
        tier_1 = torch.tensor([r["tier_1"] for r in all_rows], dtype=torch.long)
        tier_2 = torch.tensor([r["tier_2"] for r in all_rows], dtype=torch.long)
        return {
            "enc": enc,
            "group_sizes": group_sizes,
            "labels": labels,
            "tier_1": tier_1,
            "tier_2": tier_2,
        }
