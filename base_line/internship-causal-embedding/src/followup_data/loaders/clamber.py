"""CLAMBER (ACL 2024): identifying and clarifying ambiguous information needs.

Single-file dataset hosted on GitHub:
    https://github.com/SCUNLP/CLAMBER/blob/main/clamber_benchmark.jsonl

Each line is a JSON-encoded *string* (i.e. quoted) — `"{\"question\": ...}"` —
so each line needs `json.loads(json.loads(line))`.

(anchor, positive) mapping:
    anchor   = question
    positive = clarifying_question

We skip rows where `require_clarification == 0` since there is no canonical
follow-up to learn against. Use `split="all"` to keep them too.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file
from ..registry import register

_URL = (
    "https://raw.githubusercontent.com/SCUNLP/CLAMBER/main/clamber_benchmark.jsonl"
)


@register
class CLAMBER(BaseDataset):
    name = "clamber"
    description = (
        "CLAMBER: 12K examples across 3 ambiguity dimensions; ambiguity "
        "detection + clarification-question generation."
    )
    homepage = "https://github.com/SCUNLP/CLAMBER"
    citation = "Zhang et al., ACL 2024 (arXiv 2405.12063)"
    license = "Research-only (see homepage)"
    splits = ("train", "all")  # "train" = require_clarification==1 only

    def _file(self) -> Path:
        return self.data_dir / "clamber_benchmark.jsonl"

    def is_downloaded(self) -> bool:
        return self._file().exists() and self._file().stat().st_size > 0

    def download(self) -> None:
        if self.is_downloaded():
            return
        download_file(_URL, self._file())

    def _iter_examples(self) -> Iterator[PairExample]:
        keep_unclear_only = self.split == "train"
        with self._file().open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Lines are quoted JSON strings.
                outer = json.loads(line)
                row = json.loads(outer) if isinstance(outer, str) else outer
                req = row.get("require_clarification")
                if keep_unclear_only and not req:
                    continue
                anchor = (row.get("question") or "").strip()
                positive = (row.get("clarifying_question") or "").strip()
                if not anchor or not positive:
                    continue
                ctx = row.get("context") or ""
                yield PairExample(
                    anchor=anchor,
                    positive=positive,
                    dataset=self.name,
                    context=(ctx,) if ctx else (),
                    metadata={
                        "category": row.get("category"),
                        "subclass": row.get("subclass"),
                        "require_clarification": req,
                    },
                )
