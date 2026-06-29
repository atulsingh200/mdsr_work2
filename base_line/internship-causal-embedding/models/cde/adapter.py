"""CDERetriever: wraps a CausalDensityEmbedding checkpoint as a Retriever.

CDE scores via -KL(N_B || T(N_A)), which is not a simple dot product.
For the Retriever protocol we need flat embeddings for matrix similarity.

Strategy: encode anchors as their *transported* mean vector (mu_A + delta_A),
encode candidates as their plain mean vector (mu_B). Similarity = dot product
of transported anchor mean vs candidate mean (approximates the -KL score when
sigma is roughly constant). This lets us use the same pool-retrieval pipeline
as every other model for fair comparison.

For precise KL-based scoring the full evaluate_cde.py in prev/ remains available.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'my_work', 'kl'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


class CDERetriever:
    """Retriever protocol wrapper around a CausalDensityEmbedding checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        batch_size: int = 64,
        max_seq_length: int = 256,
    ) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"CDE checkpoint not found: {path}")

        from model import CausalDensityEmbedding

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        backbone = cfg.get("backbone", "google-bert/bert-base-uncased")
        proj_dim = cfg.get("proj_dim", 128)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = CausalDensityEmbedding(backbone=backbone, proj_dim=proj_dim).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = f"cde_{Path(checkpoint).parent.name}"

    def _tokenize(self, texts: list[str]):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        """Encode as transported mean: mu_A + delta_A (L2-normalized)."""
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            ids, mask = self._tokenize(texts[i : i + self.batch_size])
            mu, _, delta = self.model.encode(ids, mask)
            transported = F.normalize(mu + delta, p=2, dim=-1)
            out.append(transported.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.proj_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        """Encode as plain mean: mu_B (L2-normalized)."""
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            ids, mask = self._tokenize(texts[i : i + self.batch_size])
            mu, _, _ = self.model.encode(ids, mask)
            normed = F.normalize(mu, p=2, dim=-1)
            out.append(normed.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.proj_dim), dtype=np.float32)
