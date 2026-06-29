"""Dataset + collation for the cross-encoder directional classifier.

Mirrors src/classifier/training/data.py, but the collate JOINTLY tokenizes the
pair  (text_1, text_2)  into a single sequence  [CLS] text_1 [SEP] text_2 [SEP]
instead of producing two separate encodings.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _parse_step(v) -> int | None:
    """Normalise a tier/step value to int.  Handles ints, 'tN' strings, and None."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        digits = "".join(c for c in v if c.isdigit())
        return int(digits) if digits else None
    return None


class CEPairDataset(Dataset):
    """In-memory dataset of labelled text pairs from a JSONL file.

    Parameters
    ----------
    path      : path to a JSONL file
    step_key  : field-name prefix for the step/tier columns.
                ``"tier"`` → reads ``tier_1`` / ``tier_2`` (integers).
                ``"step"`` → reads ``step_1`` / ``step_2`` (e.g. ``"t2"`` strings).
    """

    def __init__(self, path: Path, step_key: str = "tier"):
        self.step_key = step_key
        self.rows: list[dict] = []
        skipped = 0
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    self.rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            warnings.warn(f"{path.name}: skipped {skipped} malformed line(s)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        k1, k2 = f"{self.step_key}_1", f"{self.step_key}_2"
        return {
            "text_1": r["text_1"],
            "text_2": r["text_2"],
            "label": float(r["label"]),
            "step_1": _parse_step(r.get(k1)),
            "step_2": _parse_step(r.get(k2)),
            "idx": i,
        }


class CECollate:
    """Joint-tokenize text pairs for the cross-encoder."""

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict:
        t1 = [b["text_1"] for b in batch]
        t2 = [b["text_2"] for b in batch]
        enc = self.tok(
            t1, t2, padding=True, truncation="longest_first",
            max_length=self.max_len, return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        step_1 = torch.tensor([b["step_1"] - 1 if b["step_1"] is not None else -1 for b in batch], dtype=torch.long)
        step_2 = torch.tensor([b["step_2"] - 1 if b["step_2"] is not None else -1 for b in batch], dtype=torch.long)
        return {
            "enc": enc,
            "labels": labels,
            "step_1": step_1,
            "step_2": step_2,
            "idx": [b["idx"] for b in batch],
        }
