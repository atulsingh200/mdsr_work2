"""Model registry: maps model-type strings to loader functions.

Usage in run_eval.py:
    retriever = load_retriever("biencoder", checkpoint="runs/best.pt")
    retriever = load_retriever("pretrained", hf_id="BAAI/bge-small-en-v1.5")
    retriever = load_retriever("tfidf")

Adding a new model: import its adapter and add an entry to _LOADERS.
"""

from __future__ import annotations

from typing import Any


def load_retriever(model_type: str, **kwargs: Any):
    """Instantiate a retriever by type name.

    Args:
        model_type: one of the registered keys (biencoder, cde,
                    cross_encoder_2x, lorentz, pretrained, tfidf).
        **kwargs:   passed through to the adapter constructor.

    Returns:
        An object that satisfies the Retriever protocol.
    """
    # Lazy imports so torch is only loaded when actually needed.
    if model_type == "biencoder":
        from models.biencoder.adapter import BiEncoderRetriever
        return BiEncoderRetriever(**kwargs)

    if model_type == "cde":
        from models.cde.adapter import CDERetriever
        return CDERetriever(**kwargs)

    if model_type == "cross_encoder_2x":
        from models.cross_encoder_2x.adapter import CrossEncoder2xRetriever
        return CrossEncoder2xRetriever(**kwargs)

    if model_type == "lorentz":
        from models.lorentz.adapter import LorentzRetriever
        return LorentzRetriever(**kwargs)

    if model_type == "pretrained":
        from models.pretrained.adapter import PretrainedRetriever
        return PretrainedRetriever(**kwargs)

    if model_type == "tfidf":
        from models.pretrained.adapter import TfidfRetriever
        return TfidfRetriever(**kwargs)

    raise ValueError(
        f"Unknown model type '{model_type}'. "
        f"Valid types: biencoder, cde, cross_encoder_2x, lorentz, pretrained, tfidf."
    )
