"""Directional classifier for causal dependency pairs.

Architecture: dual pretrained encoders (untied) with CLS pooling, L2
normalization, [A; B; A-B; A*B] concatenation, and a pluggable
classification head (Linear or MLP). Optionally adds auxiliary
tier-prediction heads as a regularizer.

Submodules:
    classifier.model           DirectionalClassifier
    classifier.heads           LinearHead, MLPHead, build_head
    classifier.training        Training loop + data loading
"""

from .heads import LinearHead, MLPHead, build_head
from .model import DirectionalClassifier

__all__ = [
    "DirectionalClassifier",
    "LinearHead",
    "MLPHead",
    "build_head",
]
