"""mtRAG (TACL 2025): multi-turn human-authored RAG conversations.

https://github.com/IBM/mt-rag-benchmark

Each conversation has a `messages` list with {speaker, text, timestamp,
enrichments}. We emit (text turn t -> text turn t+1) pairs across the
user/agent boundary, exactly as for QReCC/TopiOCQA. Enrichment metadata
(Question Type, Multi-Turn, Answerability) is preserved for slicing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file
from ..registry import register

_URL = (
    "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/"
    "mtrag-human/conversations/conversations.json"
)


@register
class MTRAG(BaseDataset):
    name = "mtrag"
    description = (
        "mtRAG: 110 human-authored multi-turn RAG conversations across 4 "
        "domains (~7.7 turns each). (turn t -> turn t+1) pairs."
    )
    homepage = "https://github.com/IBM/mt-rag-benchmark"
    citation = "Katsis et al., TACL 2025"
    license = "Apache-2.0"
    splits = ("train",)  # single benchmark split

    def _file(self) -> Path:
        return self.data_dir / "conversations.json"

    def is_downloaded(self) -> bool:
        return self._file().exists() and self._file().stat().st_size > 0

    def download(self) -> None:
        if self.is_downloaded():
            return
        download_file(_URL, self._file())

    def _iter_examples(self) -> Iterator[PairExample]:
        with self._file().open() as f:
            convs = json.load(f)
        for conv in convs:
            messages = conv.get("messages") or []
            history: list[str] = []
            for i in range(len(messages) - 1):
                m_a = messages[i]
                m_b = messages[i + 1]
                anchor = (m_a.get("text") or "").strip()
                positive = (m_b.get("text") or "").strip()
                if not anchor or not positive:
                    continue
                yield PairExample(
                    anchor=anchor,
                    positive=positive,
                    dataset=self.name,
                    context=tuple(history),
                    metadata={
                        "author": conv.get("author"),
                        "domain": conv.get("domain"),
                        "anchor_speaker": m_a.get("speaker"),
                        "positive_speaker": m_b.get("speaker"),
                        "anchor_enrichments": m_a.get("enrichments"),
                        "positive_enrichments": m_b.get("enrichments"),
                    },
                )
                history.append(anchor)
