"""NegativeSampler protocol — the only interface the eval runner uses for negatives.

To add a new negative strategy:
1. Create evaluations/negatives/<name>.py, implement this protocol.
2. Register it in evaluations/negatives/registry.py.
That's it — no other file changes needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from followup_data.base import PairExample, TripleExample


@runtime_checkable
class NegativeSampler(Protocol):
    """Wraps an iterable of PairExamples and yields TripleExamples with negatives.

    Attributes:
        name: short identifier shown in result filenames ("random", "bm25", ...).
    """

    name: str

    def __call__(
        self,
        pairs: Iterable[PairExample],
        k: int = 1,
    ) -> Iterator[TripleExample]:
        """Yield one TripleExample per PairExample with k negatives each."""
        ...
