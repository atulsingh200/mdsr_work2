"""QReCC: Open-Domain Question Rewriting in Conversational Contexts.

https://github.com/apple/ml-qrecc

Each conversation is a sequence of turns (Question, Truth_answer, Truth_rewrite).
We emit one PairExample per consecutive turn pair within a conversation:

    anchor   = Truth_answer at turn t   (the system's response)
    positive = Question      at turn t+1 (the user's actual follow-up)

Context = the up-to-t-1 question/answer history.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file, extract
from ..registry import register


@register
class QReCC(BaseDataset):
    name = "qrecc"
    description = (
        "QReCC: 14K conversations, 81K open-domain QA turns. Used here as a "
        "(system answer t -> user follow-up question t+1) pair source."
    )
    homepage = "https://github.com/apple/ml-qrecc"
    citation = "Anantha et al., NAACL 2021"
    license = "CC BY-SA 4.0"
    splits = ("train", "test")

    _ZIP_URL = (
        "https://github.com/apple/ml-qrecc/raw/main/dataset/qrecc_data.zip"
    )

    def is_downloaded(self) -> bool:
        return (self.data_dir / "qrecc_train.json").exists() and (
            self.data_dir / "qrecc_test.json"
        ).exists()

    def download(self) -> None:
        if self.is_downloaded():
            return
        zip_path = self.data_dir / "qrecc_data.zip"
        download_file(self._ZIP_URL, zip_path)
        extract(zip_path, self.data_dir)
        # The archive may extract into a nested folder; flatten if needed.
        for nested in self.data_dir.rglob("qrecc_train.json"):
            if nested.parent != self.data_dir:
                nested.replace(self.data_dir / "qrecc_train.json")
        for nested in self.data_dir.rglob("qrecc_test.json"):
            if nested.parent != self.data_dir:
                nested.replace(self.data_dir / "qrecc_test.json")

    def _file(self) -> Path:
        return self.data_dir / f"qrecc_{self.split}.json"

    def _iter_examples(self) -> Iterator[PairExample]:
        with self._file().open() as f:
            rows = json.load(f)
        # Group by Conversation_no preserving turn order.
        rows.sort(key=lambda r: (r["Conversation_no"], r["Turn_no"]))
        prev_conv: int | None = None
        history: list[tuple[str, str]] = []  # (question, answer)
        for r in rows:
            conv = r["Conversation_no"]
            if conv != prev_conv:
                history = []
                prev_conv = conv
            q = (r.get("Rewrite") or r.get("Question") or "").strip()
            a = (r.get("Answer") or "").strip()
            if history and a and q:
                # The previous turn's answer -> this turn's user question.
                anchor_q, anchor_a = history[-1]
                if anchor_a:
                    yield PairExample(
                        anchor=anchor_a,
                        positive=q,
                        dataset=self.name,
                        context=tuple(t for pair in history for t in pair),
                        metadata={
                            "conversation_no": conv,
                            "turn_no": r["Turn_no"],
                            "source": r.get("Conversation_source"),
                        },
                    )
            history.append((q, a))
