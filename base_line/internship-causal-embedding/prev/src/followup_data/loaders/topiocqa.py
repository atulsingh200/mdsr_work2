"""TopiOCQA: Open-Domain Conversational QA with Topic Switches.

https://github.com/McGill-NLP/topiocqa

Each conversation contains turns with: Question, Answer, Topic, Sub-Topic, etc.
We emit (answer_t -> question_{t+1}) pairs as in QReCC. Topic switches make
this a particularly hard test of follow-up coherence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file
from ..registry import register


@register
class TopiOCQA(BaseDataset):
    name = "topiocqa"
    description = (
        "TopiOCQA: 3,920 dialogues / 50K Q-A turns with explicit topic shifts. "
        "(answer t -> user question t+1) pairs across topic switches."
    )
    homepage = "https://github.com/McGill-NLP/topiocqa"
    citation = "Adlakha et al., TACL 2022"
    license = "CC BY-SA 4.0"
    splits = ("train", "dev")

    # Direct files mirrored from the official Zenodo / Drive release.
    # Source: download_data.py in the repo.
    _URLS = {
        "train": "https://zenodo.org/records/7709644/files/topiocqa_train.json",
        "dev": "https://zenodo.org/records/7709644/files/topiocqa_dev.json",
    }

    def _file(self) -> Path:
        return self.data_dir / f"topiocqa_{self.split}.json"

    def is_downloaded(self) -> bool:
        return self._file().exists() and self._file().stat().st_size > 0

    def download(self) -> None:
        if self.is_downloaded():
            return
        download_file(self._URLS[self.split], self._file())

    def _iter_examples(self) -> Iterator[PairExample]:
        with self._file().open() as f:
            rows = json.load(f)
        rows.sort(key=lambda r: (r["Conversation_no"], r["Turn_no"]))
        prev_conv: int | None = None
        history: list[tuple[str, str]] = []
        for r in rows:
            conv = r["Conversation_no"]
            if conv != prev_conv:
                history = []
                prev_conv = conv
            q = (r.get("Question") or "").strip()
            a = (r.get("Answer") or "").strip()
            if history and a and q:
                _, prev_a = history[-1]
                if prev_a:
                    yield PairExample(
                        anchor=prev_a,
                        positive=q,
                        dataset=self.name,
                        context=tuple(t for pair in history for t in pair),
                        metadata={
                            "conversation_no": conv,
                            "turn_no": r["Turn_no"],
                            "topic": r.get("Topic"),
                            "subtopic": r.get("Topic_section"),
                        },
                    )
            history.append((q, a))
