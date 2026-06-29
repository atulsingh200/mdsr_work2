"""Core abstractions for classification datasets.

Parallel to base.py (which handles retrieval pairs), this module provides
ClassificationExample and BaseClassificationDataset for binary classification
tasks where each row is a labelled (text_1, text_2, label) triple.

This separation is intentional: retrieval datasets yield PairExample(anchor,
positive) where every row is a positive pair and negatives are sampled
externally. Classification datasets carry their own labels — both positive
and negative rows — so a different abstraction is needed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Iterator


@dataclass(frozen=True, slots=True)
class ClassificationExample:
    text_1: str
    text_2: str
    label: int
    dataset: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseClassificationDataset(ABC):
    """Common interface for classification datasets.

    Mirrors BaseDataset but yields ClassificationExample instead of
    PairExample. Subclasses set class-level metadata and implement
    download / _iter_examples.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    homepage: ClassVar[str]
    citation: ClassVar[str] = ""
    license: ClassVar[str | None] = None
    splits: ClassVar[tuple[str, ...]] = ("train",)

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).expanduser().resolve()
        if split not in self.splits:
            raise ValueError(f"{self.name}: split={split!r} not in {self.splits}")
        self.split = split
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self.root / self.name

    @abstractmethod
    def is_downloaded(self) -> bool: ...

    @abstractmethod
    def download(self) -> None: ...

    @abstractmethod
    def _iter_examples(self) -> Iterator[ClassificationExample]: ...

    def __iter__(self) -> Iterator[ClassificationExample]:
        if not self.is_downloaded():
            from .base import DatasetNotDownloaded

            raise DatasetNotDownloaded(
                f"{self.name} not present in {self.data_dir}. "
                f"Place the required files under {self.data_dir}."
            )
        yield from self._iter_examples()

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def head(self, n: int = 5) -> list[ClassificationExample]:
        out: list[ClassificationExample] = []
        for ex in self:
            out.append(ex)
            if len(out) >= n:
                break
        return out

    def to_jsonl(self, out_path: Path | str) -> int:
        out_path = Path(out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w") as f:
            for ex in self:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
                n += 1
        return n
