"""Negative sampling strategies that turn PairExamples into TripleExamples.

The framing of the project is causal embedding:
- (anchor, positive)  -> high score   (positive is the true continuation)
- (anchor, negative)  -> low score    (negative is unrelated text)

These samplers wrap an iterable of PairExamples and yield TripleExamples
with k negatives each.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from typing import Literal

from .base import PairExample, TripleExample

NegMode = Literal["random_corpus", "in_batch", "shuffle_positives"]


class RandomCorpusNegatives:
    """Pre-loads a pool of positives, then samples k negatives per anchor.

    Uniform over the pool, excluding the anchor's own positive. Simple and fast,
    fine for sanity checks but trivially distinguishable in many embedding spaces.
    Use BM25Negatives or hard mining for stronger evaluation.
    """

    def __init__(self, pairs: Iterable[PairExample], k: int = 1, seed: int = 0) -> None:
        self._pool: list[str] = [p.positive for p in pairs]
        if not self._pool:
            raise ValueError("Empty pair pool")
        self.k = k
        self._rng = random.Random(seed)

    def __call__(self, pairs: Iterable[PairExample]) -> Iterator[TripleExample]:
        n = len(self._pool)
        for ex in pairs:
            negs: list[str] = []
            seen = {ex.positive}
            attempts = 0
            while len(negs) < self.k and attempts < self.k * 10:
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


class ShufflePositiveNegatives:
    """Streaming variant: keeps a small ring buffer of recent positives and
    samples negatives from it. Avoids loading the whole corpus into memory.
    """

    def __init__(self, k: int = 1, buffer_size: int = 1024, seed: int = 0) -> None:
        self.k = k
        self.buffer_size = buffer_size
        self._rng = random.Random(seed)

    def __call__(self, pairs: Iterable[PairExample]) -> Iterator[TripleExample]:
        buf: list[str] = []
        for ex in pairs:
            if len(buf) >= self.k + 1:
                negs: list[str] = []
                seen = {ex.positive}
                attempts = 0
                while len(negs) < self.k and attempts < self.k * 10:
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


def with_random_negatives(
    pairs: Iterable[PairExample], k: int = 1, seed: int = 0
) -> Iterator[TripleExample]:
    """Convenience: materialize pairs once, then sample uniformly from the pool."""
    pairs_list = list(pairs)
    sampler = RandomCorpusNegatives(pairs_list, k=k, seed=seed)
    return sampler(pairs_list)
