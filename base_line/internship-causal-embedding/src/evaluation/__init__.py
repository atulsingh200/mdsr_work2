"""Model-agnostic evaluation harness for follow-up retrieval tasks.

The core abstraction is `Retriever` — a protocol that encodes anchors and
candidates into a shared vector space. Any model approach (bi-encoder,
classifier, cross-encoder, pretrained baseline, TF-IDF) can implement it,
and the same eval scripts apply.

Submodules:
    evaluation.interfaces       Retriever Protocol
    evaluation.metrics          MRR, Recall@K, Hit@K (multi-gold aware)
    evaluation.retrievers       Built-in baselines (pretrained, tfidf)
    evaluation.retrieval_eval   Unified `evaluate-retrieval` CLI
    evaluation.llm_baseline     LLM ranking baseline (separate CLI)
"""

from .interfaces import Retriever
from .metrics import compute_metrics, random_baseline_recall
from .retrievers import PretrainedRetriever, TfidfRetriever

__all__ = [
    "Retriever",
    "compute_metrics",
    "random_baseline_recall",
    "PretrainedRetriever",
    "TfidfRetriever",
]
