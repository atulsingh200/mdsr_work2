"""AEP Causal Classification: tier-based directional classification pairs
from Adobe Experience Platform / Journey Optimizer documentation.

Each row is a (text_1, text_2, label) triple where:
  - label=1 means text_1's tier causally precedes text_2's tier
  - label=0 means the pair is not in a causal (prerequisite) relationship

Both texts are sentence-based segments (2–5 sentences) drawn from AJO tutorial
transcripts, assigned to a 15-tier causal dependency hierarchy. Pairs span all
C(15,2)=105 tier combinations, single-direction, capped at 1500 per tier-pair.

Schema per JSONL row:
    text_1, text_2    : segment texts
    label             : 1 (causal) or 0 (not causal)
    tier_1, tier_2    : integer tier IDs (1–15)
    sub_1, sub_2      : sub-chapter codes (e.g. "2a", "9h")
    url_1, url_2      : source tutorial URLs

Splits: train (110,775), val (6,004), test (6,004).

Data is NOT shipped with the repo due to size (~180 MB). Place the JSONL files
under `data/aep_causal_classification/` before using this loader. See README
for procurement instructions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..classification_base import BaseClassificationDataset, ClassificationExample
from ..classification_registry import register_classification


@register_classification
class AEPCausalClassification(BaseClassificationDataset):
    name = "aep_causal_classification"
    description = (
        "AEP Causal Classification: 122,783 tier-based directional pairs from "
        "Adobe Experience Platform documentation. Sentence-based segments across "
        "a 15-tier causal dependency hierarchy. "
        "110,775 train / 6,004 val / 6,004 test."
    )
    homepage = "internal"
    citation = "Internal — Adobe Experience Platform documentation corpus (v6, sentence-based)"
    license = "Internal use only"
    splits = ("train", "val", "test")

    _FILES = {
        "train": "directional_train.jsonl",
        "val": "directional_val.jsonl",
        "test": "directional_test.jsonl",
    }

    def _file(self) -> Path:
        return self.data_dir / self._FILES[self.split]

    def is_downloaded(self) -> bool:
        return all(
            (self.data_dir / fname).exists()
            and (self.data_dir / fname).stat().st_size > 0
            for fname in self._FILES.values()
        )

    def download(self) -> None:
        """No public URL — data must be procured separately.

        Validates that the expected files exist locally.
        """
        if self.is_downloaded():
            return
        missing = [
            fname
            for fname in self._FILES.values()
            if not (self.data_dir / fname).exists()
            or (self.data_dir / fname).stat().st_size == 0
        ]
        from ..base import DatasetNotDownloaded

        raise DatasetNotDownloaded(
            f"{self.name}: missing files {missing} in {self.data_dir}. "
            f"Place {list(self._FILES.values())} under {self.data_dir}. "
            f"See README for data procurement instructions."
        )

    def _iter_examples(self) -> Iterator[ClassificationExample]:
        with self._file().open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                yield ClassificationExample(
                    text_1=r["text_1"],
                    text_2=r["text_2"],
                    label=int(r["label"]),
                    dataset=self.name,
                    metadata={
                        "tier_1": r.get("tier_1"),
                        "tier_2": r.get("tier_2"),
                        "sub_1": r.get("sub_1"),
                        "sub_2": r.get("sub_2"),
                        "url_1": r.get("url_1"),
                        "url_2": r.get("url_2"),
                    },
                )
