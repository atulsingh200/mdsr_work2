"""Negative sampler registry: maps strategy name → sampler class.

Usage in runner.py:
    sampler = get_sampler("bm25", pairs=train_pairs, k=4)

Adding a new strategy:
    1. Create evaluations/negatives/<name>.py
    2. Import it here and add an entry to _REGISTRY.
"""

from __future__ import annotations

from typing import Any


def get_sampler(name: str, **kwargs: Any):
    """Instantiate a NegativeSampler by name.

    Args:
        name:     registered strategy name.
        **kwargs: passed to the sampler constructor (e.g. pairs=..., k=..., seed=...).

    Returns:
        A callable that satisfies the NegativeSampler protocol.
    """
    if name == "random":
        from evaluations.negatives.random_corpus import RandomCorpusNegatives
        return RandomCorpusNegatives(**kwargs)

    if name == "shuffle":
        from evaluations.negatives.shuffle_positives import ShufflePositiveNegatives
        return ShufflePositiveNegatives(**kwargs)

    if name == "bm25":
        from evaluations.negatives.bm25 import BM25Negatives
        return BM25Negatives(**kwargs)

    raise ValueError(
        f"Unknown negative strategy '{name}'. "
        f"Valid strategies: random, shuffle, bm25."
    )
