"""MultiWOZ — task-oriented dialog with action labels.

The original `multi_woz_v22` HuggingFace dataset uses a loading script which
HF no longer permits. We use Brendan/multiwoz_turns_v24, a turn-level parquet
mirror with explicit `user` and `system_response` columns — exactly the
(anchor → positive) framing we want.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file
from ..registry import register

_BASE = "https://huggingface.co/datasets/Brendan/multiwoz_turns_v24/resolve/main/data"
_FILES = {
    "train": "train-00000-of-00001-d27f1cce8ef2d445.parquet",
    "validation": "validation-00000-of-00001-e2d540d091777ea4.parquet",
    "test": "test-00000-of-00001-98ae416deade52a1.parquet",
}


@register
class MultiWOZ(BaseDataset):
    name = "multiwoz_v24"
    description = (
        "MultiWOZ v2.4 turn-level mirror: ~10K task-oriented multi-domain "
        "dialogues; (user utterance -> system response) pairs."
    )
    homepage = "https://huggingface.co/datasets/Brendan/multiwoz_turns_v24"
    citation = "Zang et al., 2020 (Multi-Domain Wizard-of-Oz 2.2)"
    license = "MIT"
    splits = ("train", "validation", "test")

    def _file(self) -> Path:
        return self.data_dir / _FILES[self.split]

    def is_downloaded(self) -> bool:
        return self._file().exists() and self._file().stat().st_size > 0

    def download(self) -> None:
        for split, fname in _FILES.items():
            dest = self.data_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            download_file(f"{_BASE}/{fname}", dest)

    def _iter_examples(self) -> Iterator[PairExample]:
        import pandas as pd

        df = pd.read_parquet(self._file())
        for _, row in df.iterrows():
            anchor = (row.get("user") or "").strip()
            positive = (row.get("system_response") or "").strip()
            if not anchor or not positive:
                continue
            history = row.get("history")
            ctx: tuple[str, ...] = ()
            if history is not None and len(history):
                ctx = tuple(str(h) for h in history)
            yield PairExample(
                anchor=anchor,
                positive=positive,
                dataset=self.name,
                context=ctx,
                metadata={
                    "dialogue_id": row.get("dialogue_id"),
                    "turn_id": int(row["turn_id"]) if "turn_id" in row else None,
                    "degenerate_user": bool(row.get("degenerate_user", False)),
                },
            )
