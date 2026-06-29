"""BiEncoderRetriever: wraps a trained BiEncoder checkpoint as a Retriever.

Loads from checkpoint path. Implements encode_anchors / encode_candidates
using the model's untied towers. Satisfies models.base.Retriever protocol.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pathlib import Path

import numpy as np
import torch

from biencoder.config import get_model_config
from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts


class BiEncoderRetriever:
    """Retriever-protocol wrapper around a trained BiEncoder checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        backbone: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        backbone_key = backbone or cfg.get("backbone") or "bge-small"
        mcfg = get_model_config(backbone_key)
        model_name = cfg.get("model_name") or mcfg["model_name"]
        pooling = cfg.get("pooling") or mcfg["pooling"]
        self.anchor_prefix = cfg.get("anchor_prefix", mcfg["anchor_prefix"])
        self.positive_prefix = cfg.get("positive_prefix", mcfg["positive_prefix"])
        self.max_seq_length = cfg.get("max_seq_length", mcfg["max_seq_length"])

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = BiEncoder(
            model_name=model_name,
            pooling_strategy=pooling,
            anchor_prefix=self.anchor_prefix,
            positive_prefix=self.positive_prefix,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = get_tokenizer(model_name)
        self.batch_size = batch_size

        ckpt_name = Path(checkpoint).parent.name
        self.name = f"biencoder_{ckpt_name}"

    @torch.no_grad()
    def _encode_with(self, encode_fn, texts: list[str], prefix: str) -> np.ndarray:
        texts = [prefix + t for t in texts] if prefix else texts
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            input_ids, attention_mask = tokenize_texts(
                self.tokenizer, batch, self.max_seq_length
            )
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            embeds = encode_fn(input_ids, attention_mask)
            out.append(embeds.cpu().numpy())
        if not out:
            return np.zeros((0, self.model.hidden_size), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        return self._encode_with(self.model.encode_anchor, texts, self.anchor_prefix)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        return self._encode_with(self.model.encode_positive, texts, self.positive_prefix)
