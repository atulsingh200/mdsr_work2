"""FollowupQG (IJCNLP 2023): info-seeking follow-up question generation.

The dataset is mirrored on HuggingFace at Vivian12300/FollowupQG with
{train,valid,test}.json — each a list of records with fields:
  id, question, answer, follow-up, relation

(anchor, positive) mapping:
    anchor   = "<question>\n\n<answer>"
    positive = "<follow-up>"
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file
from ..registry import register

_BASE = "https://huggingface.co/datasets/Vivian12300/FollowupQG/resolve/main"


@register
class FollowupQG(BaseDataset):
    name = "followupqg"
    description = (
        "FollowupQG: 3,790 (question, answer, follow-up) tuples from Reddit ELI5. "
        "Most-cited follow-up question generation benchmark."
    )
    homepage = "https://huggingface.co/datasets/Vivian12300/FollowupQG"
    citation = "Meng et al., IJCNLP 2023 (arXiv 2309.05007)"
    license = "MIT"
    splits = ("train", "valid", "test")

    _FILES = {"train": "train.json", "valid": "valid.json", "test": "test.json"}

    def _file(self) -> Path:
        return self.data_dir / self._FILES[self.split]

    def is_downloaded(self) -> bool:
        return self._file().exists() and self._file().stat().st_size > 0

    def download(self) -> None:
        for split, fname in self._FILES.items():
            dest = self.data_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            download_file(f"{_BASE}/{fname}", dest)

    def _iter_examples(self) -> Iterator[PairExample]:
        with self._file().open() as f:
            rows = json.load(f)
        for r in rows:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            fu = (r.get("follow-up") or r.get("followup") or "").strip()
            if not q or not fu:
                continue
            anchor = f"{q}\n\n{a}" if a else q
            yield PairExample(
                anchor=anchor,
                positive=fu,
                dataset=self.name,
                metadata={"id": r.get("id"), "relation": r.get("relation")},
            )
