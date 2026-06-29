"""Random corpus negative sampler.

Pre-loads the full pool of positives, then samples k negatives per anchor
uniformly at random (excluding the anchor's own positive).

Fast and simple — good for sanity checks. Weak negatives: easy to distinguish
in most embedding spaces. Use bm25.py or dense_mined.py for harder eval.
"""

from __future__ import annotations

import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from collections.abc import Iterable, Iterator

from followup_data.base import PairExample, TripleExample


class RandomCorpusNegatives:
    """Sample k negatives per anchor uniformly from the corpus positive pool."""

    name = "random"

    def __init__(
        self,
        pairs: Iterable[PairExample],
        k: int = 1,
        seed: int = 0,
    ) -> None:
        self._pool: list[str] = [p.positive for p in pairs]
        if not self._pool:
            raise ValueError("RandomCorpusNegatives: empty pair pool")
        self.k = k
        self._rng = random.Random(seed)

    def __call__(
        self,
        pairs: Iterable[PairExample],
        k: int | None = None,
    ) -> Iterator[TripleExample]:
        k = k if k is not None else self.k
        n = len(self._pool)
        for ex in pairs:
            negs: list[str] = []
            seen = {ex.positive}
            attempts = 0
            while len(negs) < k and attempts < k * 10:
                cand = self._pool[self._rng.randrange(n)]
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
