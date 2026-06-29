"""Dataset registry. Loaders self-register on import."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseDataset

_REGISTRY: dict[str, type["BaseDataset"]] = {}


def register(cls: type["BaseDataset"]) -> type["BaseDataset"]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"Dataset {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> type["BaseDataset"]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {list_datasets()}"
        ) from e


def load(name: str, root: Path | str, split: str = "train") -> "BaseDataset":
    return get(name)(root=root, split=split)


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data"


def default_root() -> Path:
    return DEFAULT_ROOT
