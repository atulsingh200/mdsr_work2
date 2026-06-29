"""followup_data: middleware for follow-up causal embedding datasets.

Quick start:

    from followup_data import load, list_datasets, with_random_negatives

    ds = load("followupqg")
    ds.download()                      # idempotent
    for ex in ds.head(3):
        print(ex.anchor, "->", ex.positive)

    for triple in with_random_negatives(ds, k=4, seed=0):
        # triple.anchor, triple.positive, triple.negatives
        ...
"""

from .base import (
    BaseDataset,
    DatasetNotDownloaded,
    PairExample,
    TripleExample,
    chain,
)
from .classification_base import (
    BaseClassificationDataset,
    ClassificationExample,
)
from .classification_registry import (
    get_classification,
    list_classification_datasets,
    load_classification,
    register_classification,
)
from .negatives import (
    RandomCorpusNegatives,
    ShufflePositiveNegatives,
    with_random_negatives,
)
from .registry import default_root, get, list_datasets, load, register

# Importing the loaders package triggers self-registration.
from . import loaders  # noqa: F401, E402

__all__ = [
    # Retrieval
    "BaseDataset",
    "DatasetNotDownloaded",
    "PairExample",
    "TripleExample",
    "RandomCorpusNegatives",
    "ShufflePositiveNegatives",
    "chain",
    "default_root",
    "get",
    "list_datasets",
    "load",
    "register",
    "with_random_negatives",
    # Classification
    "BaseClassificationDataset",
    "ClassificationExample",
    "get_classification",
    "list_classification_datasets",
    "load_classification",
    "register_classification",
]
