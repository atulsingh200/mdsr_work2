"""Retriever protocol — the only interface the eval pipeline knows about.

Every model adapter (BiEncoder, CDE, CrossEncoder2x, Lorentz, Pretrained)
must implement this two-method protocol. The eval runner calls nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Retriever(Protocol):
    """Encodes anchors and candidates into a shared L2-normalized vector space.

    Attributes:
        name: short identifier shown in result tables and filenames.
    """

    name: str

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        """Encode anchor/query texts. Returns (N, D) float32, L2-normalized."""
        ...

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        """Encode candidate/document texts. Returns (M, D) float32, L2-normalized."""
        ...
