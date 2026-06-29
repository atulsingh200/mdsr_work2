"""Classification dataset registry. Loaders self-register on import.

Separate from the retrieval registry (registry.py) because classification
datasets use a different base class and example type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classification_base import BaseClassificationDataset

_REGISTRY: dict[str, type["BaseClassificationDataset"]] = {}


def register_classification(
    cls: type["BaseClassificationDataset"],
) -> type["BaseClassificationDataset"]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"Classification dataset {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def list_classification_datasets() -> list[str]:
    return sorted(_REGISTRY)


def get_classification(name: str) -> type["BaseClassificationDataset"]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown classification dataset {name!r}. "
            f"Available: {list_classification_datasets()}"
        ) from e


def load_classification(
    name: str, root: Path | str, split: str = "train",
) -> "BaseClassificationDataset":
    return get_classification(name)(root=root, split=split)
