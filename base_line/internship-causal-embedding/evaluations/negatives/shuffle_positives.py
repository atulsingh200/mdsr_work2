"""Shuffle-positives negative sampler (streaming / memory-efficient).

Maintains a small ring buffer of recently seen positives and samples negatives
from it. Avoids materializing the full corpus — suitable for large datasets.

Tradeoff: slightly weaker negatives than RandomCorpusNegatives (narrower pool)
but runs in O(buffer_size) memory instead of O(N).
"""

from __future__ import annotations

import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from collections.abc import Iterable, Iterator

from followup_data.base import PairExample, TripleExample


class ShufflePositiveNegatives:
    """Stream pairs through a ring buffer; sample negatives from the buffer."""

    name = "shuffle"

    def __init__(
        self,
        k: int = 1,
        buffer_size: int = 1024,
        seed: int = 0,
    ) -> None:
        self.k = k
        self.buffer_size = buffer_size
        self._rng = random.Random(seed)

    def __call__(
        self,
        pairs: Iterable[PairExample],
        k: int | None = None,
    ) -> Iterator[TripleExample]:
        k = k if k is not None else self.k
        buf: list[str] = []
        for ex in pairs:
            if len(buf) >= k + 1:
                negs: list[str] = []
                seen = {ex.positive}
                attempts = 0
                while len(negs) < k and attempts < k * 10:
                    cand = buf[self._rng.randrange(len(buf))]
                    if cand not in seen:
                        negs.append(cand)
                        seen.add(cand)
                    attempts += 1
                yield TripleExample(
                    anchor=ex.anchor,
                    positive=ex.positive,
                    negatives=tuple(negs),
                    dataset=ex.dataset,
                    context=ex.context,
                    metadata=ex.metadata,
                )
            buf.append(ex.positive)
            if len(buf) > self.buffer_size:
                buf.pop(self._rng.randrange(len(buf)))
