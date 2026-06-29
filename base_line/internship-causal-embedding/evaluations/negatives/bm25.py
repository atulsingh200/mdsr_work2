"""BM25 hard negative sampler.

Builds a BM25Okapi index over the positive pool. Per anchor, queries the index
and returns the top-k scoring positives that aren't the anchor's own gold.
These are lexically similar to the anchor, making them harder negatives than
random sampling.

Requires: pip install rank-bm25   (already in pyproject.toml [hard-negatives])
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from collections.abc import Iterable, Iterator

from followup_data.base import PairExample, TripleExample


class BM25Negatives:
    """BM25-mined hard negatives from the positive corpus."""

    name = "bm25"

    def __init__(
        self,
        pairs: Iterable[PairExample],
        k: int = 1,
        oversample: int = 4,
        tokenize=None,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25Negatives requires rank-bm25: pip install '.[hard-negatives]'"
            ) from exc

        self._tokenize = tokenize or (lambda s: s.lower().split())
        self._pool: list[str] = [p.positive for p in pairs]
        if not self._pool:
            raise ValueError("BM25Negatives: empty pair pool")
        self._bm25 = BM25Okapi([self._tokenize(p) for p in self._pool])
        self.k = k
        self.oversample = oversample

    def __call__(
        self,
        pairs: Iterable[PairExample],
        k: int | None = None,
    ) -> Iterator[TripleExample]:
        k = k if k is not None else self.k
        fetch_n = max(k * self.oversample, k + 1)
        for ex in pairs:
            scores = self._bm25.get_scores(self._tokenize(ex.anchor))
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:fetch_n]
            negs: list[str] = []
            seen = {ex.positive}
            for idx in top:
                cand = self._pool[idx]
                if cand in seen:
                    continue
                negs.append(cand)
                seen.add(cand)
                if len(negs) >= k:
                    break
            yield TripleExample(
                anchor=ex.anchor,
                positive=ex.positive,
                negatives=tuple(negs),
                dataset=ex.dataset,
                context=ex.context,
                metadata=ex.metadata,
            )
