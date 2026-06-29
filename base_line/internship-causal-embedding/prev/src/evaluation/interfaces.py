"""Abstract interfaces for evaluation.

Eval scripts in this module take models that conform to one of these
interfaces, so the same script works across different model approaches
(bi-encoder, classifier, cross-encoder, LLM, ...) without modification.

Each model module is expected to ship its own adapter that conforms to
one of these protocols — see e.g. `biencoder/evaluation_adapter.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Retriever(Protocol):
    """A retriever encodes anchors and candidates into a shared vector space.

    Implementations need only produce L2-normalized embeddings; ranking is
    done by the eval scripts via dot product (= cosine similarity).

    Attributes:
        name: short identifier shown in result files / CLI output.
    """

    name: str

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        """Encode anchor / query texts. Returns (N, D) L2-normalized."""
        ...

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        """Encode candidate / document texts. Returns (M, D) L2-normalized."""
        ...
