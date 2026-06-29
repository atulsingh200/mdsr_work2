"""AEP Causal Pairs: Claude-judged causal document pairs from Adobe
Experience Platform / Journey Optimizer documentation.

Each row is a (source_doc, target_doc) pair where the source doc causally
precedes the target doc — typically as a prerequisite ("read this first")
or as a recommended next step ("do this next"). Pairs were judged by Claude
on the AEP docs corpus.

(anchor, positive) mapping:
    anchor   = text_1_causal     # source-doc chunk (cause)
    positive = text_2_target     # target-doc chunk (effect / follow-up)

Splits: train (5,487), val (562), test (562). The data ships with the repo
under `data/aep_causal/{train,val,test}.jsonl`, so `download()` only verifies
file presence — there is no public URL to fetch from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, DatasetNotDownloaded, PairExample
from ..registry import register


@register
class AEPCausal(BaseDataset):
    name = "aep_causal"
    description = (
        "AEP Causal Pairs: 6,611 Claude-judged prerequisite / next-step pairs "
        "between Adobe Experience Platform documentation pages. "
        "5,487 train / 562 val / 562 test."
    )
    homepage = "internal"
    citation = "Internal — Adobe Experience Platform documentation corpus (v4+v5)"
    license = "Internal use only"
    splits = ("train", "val", "test")

    _FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}

    def _file(self) -> Path:
        return self.data_dir / self._FILES[self.split]

    def is_downloaded(self) -> bool:
        return all(
            (self.data_dir / fname).exists()
            and (self.data_dir / fname).stat().st_size > 0
            for fname in self._FILES.values()
        )

    def download(self) -> None:
        """No public URL — data is committed alongside the repo.

        This method only validates that the expected files exist locally.
        """
        if self.is_downloaded():
            return
        missing = [
            fname
            for fname in self._FILES.values()
            if not (self.data_dir / fname).exists()
            or (self.data_dir / fname).stat().st_size == 0
        ]
        raise DatasetNotDownloaded(
            f"{self.name}: missing files {missing} in {self.data_dir}. "
            f"Place {list(self._FILES.values())} under {self.data_dir} "
            f"(they should ship with the repo)."
        )

    def _iter_examples(self) -> Iterator[PairExample]:
        with self._file().open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                anchor = (r.get("text_1_causal") or "").strip()
                positive = (r.get("text_2_target") or "").strip()
                if not anchor or not positive:
                    continue
                yield PairExample(
                    anchor=anchor,
                    positive=positive,
                    dataset=self.name,
                    metadata={
                        "pair_id": r.get("pair_id"),
                        "source_doc_id": r.get("source_doc_id"),
                        "target_doc_id": r.get("target_doc_id"),
                        "source_doc_title": r.get("source_doc_title"),
                        "target_doc_title": r.get("target_doc_title"),
                        "anchor_text": r.get("anchor_text"),
                        "causal_type": r.get("causal_type"),
                        "context_strategy": r.get("context_strategy"),
                        "confidence": r.get("confidence"),
                        "reasoning": r.get("reasoning"),
                        "text_overlap_ratio": r.get("text_overlap_ratio"),
                        "text_1_word_count": r.get("text_1_word_count"),
                        "text_2_word_count": r.get("text_2_word_count"),
                    },
                )
