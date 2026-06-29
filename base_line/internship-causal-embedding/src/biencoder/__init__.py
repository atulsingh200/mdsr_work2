"""Bi-encoder for follow-up causal embeddings.

A bi-encoder learns two separate encoders — one for anchors, one for
positives (follow-up text) — that diverge during training to capture
asymmetric / directional relations. The two encoders are initialized
from the same pretrained backbone (e.g. BGE-small) and trained with
InfoNCE contrastive loss using in-batch negatives.

Submodules:
    biencoder.model     BiEncoder, pooling layers
    biencoder.losses    InfoNCELoss
    biencoder.config    MODEL_CONFIGS (per-backbone hyperparameters)
    biencoder.training  Training pipeline (PairDataset, train, validation)
"""

from .losses import InfoNCELoss
from .model import BiEncoder, CLSPooling, MeanPooling, get_pooling
from .config import MODEL_CONFIGS, get_model_config

__all__ = [
    "BiEncoder",
    "MeanPooling",
    "CLSPooling",
    "get_pooling",
    "InfoNCELoss",
    "MODEL_CONFIGS",
    "get_model_config",
]
