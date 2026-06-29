"""Training pipeline for the bi-encoder.

Submodules:
    biencoder.training.data        PairDataset wrapper over any BaseDataset
    biencoder.training.validation  In-training metrics (MRR, Recall@K)
    biencoder.training.train       Training loop + CLI entry point
"""

from .data import PairDataset, make_collate_fn
from .validation import compute_retrieval_metrics, validate

__all__ = [
    "PairDataset",
    "make_collate_fn",
    "compute_retrieval_metrics",
    "validate",
]
