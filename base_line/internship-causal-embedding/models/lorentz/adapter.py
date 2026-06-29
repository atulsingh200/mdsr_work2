"""LorentzRetriever: wraps the LorentzBiEncoder checkpoint as a Retriever.

The Lorentz model produces (space, time) per text. For retrieval we combine
them into a single vector: [space || time] and L2-normalize. This preserves
both the semantic (space) and causal (time) signals in one embedding for
dot-product ranking.

For the asymmetric Lorentz scoring (space + f(time_diff)), use the original
lorentz_enc_workflow/evaluate.py in prev/.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lorentz_enc_workflow'))

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class LorentzRetriever:
    """Retriever protocol wrapper around a LorentzBiEncoder checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
    ) -> None:
        from model import LorentzBiEncoder, get_tokenizer

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Lorentz checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        backbone = cfg.get("backbone_name", "BAAI/bge-m3")
        space_dim = cfg.get("space_dim", 1004)
        time_dim = cfg.get("time_dim", 20)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = LorentzBiEncoder(
            backbone_name=backbone,
            space_dim=space_dim,
            time_dim=time_dim,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = get_tokenizer(backbone)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = f"lorentz_{Path(checkpoint).parent.name}"

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        from model import tokenize
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            ids, mask = tokenize(
                self.tokenizer,
                texts[i : i + self.batch_size],
                self.max_seq_length,
                self.device,
            )
            space, time = self.model(ids, mask)
            combined = torch.cat([space, time], dim=-1)
            combined = F.normalize(combined, p=2, dim=-1)
            out.append(combined.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros(
            (0, self.model.space_dim + self.model.time_dim), dtype=np.float32
        )

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)
