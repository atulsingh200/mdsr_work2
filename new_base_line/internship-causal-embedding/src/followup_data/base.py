"""Core abstractions: PairExample, TripleExample, BaseDataset.

Every dataset normalizes to PairExample(anchor, positive) — i.e. a piece of text
followed by the next piece of text. Negative samples are produced by Negatives
sampling strategies in negatives.py, yielding TripleExample(anchor, positive,
negatives) suitable for contrastive / ranking objectives.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable, Iterator


@dataclass(frozen=True, slots=True)
class PairExample:
    anchor: str
    positive: str
    dataset: str
    context: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TripleExample:
    anchor: str
    positive: str
    negatives: tuple[str, ...]
    dataset: str
    context: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetNotDownloaded(RuntimeError):
    pass


class BaseDataset(ABC):
    """Common interface for every dataset.

    Subclasses set the class-level metadata and implement download / __iter__.
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
    def _iter_examples(self) -> Iterator[PairExample]: ...

    def __iter__(self) -> Iterator[PairExample]:
        if not self.is_downloaded():
            raise DatasetNotDownloaded(
                f"{self.name} not present in {self.data_dir}. "
                f"Run `.download()` or `followup-data download {self.name}`."
            )
        yield from self._iter_examples()

    def __len__(self) -> int:
        # Override when cheap to compute. Default: count by iterating.
        return sum(1 for _ in self)

    def head(self, n: int = 5) -> list[PairExample]:
        out: list[PairExample] = []
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


def chain(datasets: Iterable[BaseDataset]) -> Iterator[PairExample]:
    """Iterate examples across multiple datasets in sequence."""
    for ds in datasets:
        yield from ds
