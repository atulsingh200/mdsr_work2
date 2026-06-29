"""Loader modules — importing each one self-registers via @register.

Classification loaders use a separate registry (classification_registry.py)
but are imported here alongside retrieval loaders for convenience.
"""

from . import (  # noqa: F401
    _manual,
    aep_causal,
    aep_causal_classification,
    clamber,
    clariq,
    clarq_llm,
    followupqg,
    infoquest,
    movielens,
    mtrag,
    multiwoz,
    proactive_agent,
    qrecc,
    topiocqa,
)
