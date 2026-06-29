"""ClariQ: Clarifying Questions in Conversational Search.

https://github.com/aliannejadi/ClariQ

train.tsv / dev.tsv layout (tab-separated):
    topic_id  initial_request  topic_desc  facet_id  facet_desc
    question_id  question  clarification_need

We emit:

    anchor   = initial_request          (the user's ambiguous query)
    positive = question                 (the chosen clarifying question)
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file
from ..registry import register

_BASE = "https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data"


@register
class ClariQ(BaseDataset):
    name = "clariq"
    description = (
        "ClariQ: open-domain conversational search clarifications. "
        "(query -> clarifying question) pairs."
    )
    homepage = "https://github.com/aliannejadi/ClariQ"
    citation = "Aliannejadi et al., 2020"
    license = "MIT"
    splits = ("train", "dev")

    _FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
    }

    def _files(self) -> list[Path]:
        return [self.data_dir / self._FILES[self.split]]

    def is_downloaded(self) -> bool:
        return all(p.exists() and p.stat().st_size > 0 for p in self._files())

    def download(self) -> None:
        for split, fname in self._FILES.items():
            dest = self.data_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            download_file(f"{_BASE}/{fname}", dest)

    def _iter_examples(self) -> Iterator[PairExample]:
        path = self.data_dir / self._FILES[self.split]
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                anchor = (row.get("initial_request") or "").strip()
                positive = (row.get("question") or "").strip()
                if not anchor or not positive or positive == "-":
                    continue
                yield PairExample(
                    anchor=anchor,
                    positive=positive,
                    dataset=self.name,
                    metadata={
                        "topic_id": row.get("topic_id"),
                        "facet_id": row.get("facet_id"),
                        "clarification_need": row.get("clarification_need"),
                    },
                )
