"""Cross-encoder directional classifier + two-miner ensemble.

A cross-encoder jointly encodes (text_1 [SEP] text_2) with full cross-attention
and predicts a single directional logit from the [CLS] representation. Two such
models are trained on different hard-negative flavors (semantic vs in-domain) and
blended at inference.

Submodules:
    crossencoder.model          CrossEncoderClassifier
    crossencoder.data           CEPairDataset, CECollate
    crossencoder.train_ce       training loop + per-tier-pair test eval
    crossencoder.ensemble_eval  blend two trained CEs on the test set
"""

from .model import CrossEncoderClassifier

__all__ = ["CrossEncoderClassifier"]
