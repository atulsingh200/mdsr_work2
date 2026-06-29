"""Per-backbone configurations for the bi-encoder.

Each entry encodes the model id, max sequence length, the pooling strategy
that matches the model's pretraining, and any prefix the model expects on
query / document inputs.
"""

from __future__ import annotations

from typing import TypedDict


class ModelConfig(TypedDict):
    model_name: str
    embedding_dim: int
    max_seq_length: int
    pooling: str            # "mean" or "cls"
    anchor_prefix: str
    positive_prefix: str
    suggested_batch_size: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "max_seq_length": 512,
        "pooling": "mean",
        "anchor_prefix": "",
        "positive_prefix": "",
        "suggested_batch_size": 64,
    },
    "arctic-embed-l": {
        "model_name": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "embedding_dim": 1024,
        "max_seq_length": 8192,
        "pooling": "cls",
        "anchor_prefix": "query: ",
        "positive_prefix": "",
        "suggested_batch_size": 16,
    },
}

DEFAULT_MODEL_KEY = "bge-small"


def get_model_config(key: str | None = None) -> ModelConfig:
    """Look up a backbone config by short key (e.g. 'bge-small')."""
    key = key or DEFAULT_MODEL_KEY
    if key not in MODEL_CONFIGS:
        raise ValueError(
            f"unknown model key {key!r}. available: {list(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[key]
