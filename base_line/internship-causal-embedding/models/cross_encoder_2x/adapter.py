"""CrossEncoder2xRetriever: wraps CrossEncoder-2x as a Retriever.

Cross-encoders score (anchor, candidate) pairs jointly — they cannot produce
independent anchor/candidate embeddings. To plug into the pool-retrieval
pipeline we use a scoring-matrix approach: for each anchor, score it against
every candidate in the pool explicitly.

Because this is O(N * pool_size) forward passes instead of O(N + pool_size),
evaluation is slower than bi-encoders. The runner handles this transparently.
The adapter exposes a special `score_matrix` method; the runner detects and
uses it when present.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'my_work', 'kl', 'cross_encoder_2x'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'my_work', 'kl', 'cross_encoder'))

from pathlib import Path

import numpy as np
import torch


class CrossEncoder2xRetriever:
    """Pool-retrieval wrapper for the CrossEncoder-2x model.

    Implements encode_anchors / encode_candidates as dummies (returns zeros)
    because cross-encoders cannot embed independently. The runner calls
    score_matrix() when it detects this adapter type.
    """

    IS_CROSS_ENCODER = True  # signal to runner to use score_matrix path

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        from model_ce2x import CrossEncoder2x, get_tokenizer

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"CrossEncoder2x checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        backbone = cfg.get("backbone", "google-bert/bert-base-uncased")
        n_layers = cfg.get("n_layers", 24)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = CrossEncoder2x(backbone=backbone, n_layers=n_layers).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = get_tokenizer(backbone)
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = f"cross_encoder_2x_{Path(checkpoint).parent.name}"

    @torch.no_grad()
    def score_pairs(self, anchors: list[str], candidates: list[str]) -> np.ndarray:
        """Score len(anchors) == len(candidates) pairs. Returns (N,) float array."""
        out: list[float] = []
        for start in range(0, len(anchors), self.batch_size):
            end = min(start + self.batch_size, len(anchors))
            enc = self.tokenizer(
                anchors[start:end], candidates[start:end],
                padding=True, truncation="longest_first",
                max_length=self.max_length, return_tensors="pt",
                return_token_type_ids=True,
            )
            ids = enc["input_ids"].to(self.device, non_blocking=True)
            mask = enc["attention_mask"].to(self.device, non_blocking=True)
            tti = enc.get("token_type_ids")
            if tti is not None:
                tti = tti.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = self.model(ids, mask, tti)
            out.extend(logits.float().cpu().tolist())
        return np.asarray(out, dtype=np.float32)

    @torch.no_grad()
    def score_matrix(self, anchors: list[str], candidates: list[str]) -> np.ndarray:
        """Score all (anchor_i, candidate_j) pairs. Returns (N_a, N_c) matrix.

        Used by the runner when IS_CROSS_ENCODER is True.
        Scores each anchor against every candidate via flat pair scoring.
        """
        n_a, n_c = len(anchors), len(candidates)
        flat_a = [a for a in anchors for _ in candidates]
        flat_c = candidates * n_a
        scores = self.score_pairs(flat_a, flat_c)
        return scores.reshape(n_a, n_c)

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        """Not applicable for cross-encoders — runner uses score_matrix instead."""
        return np.zeros((len(texts), 1), dtype=np.float32)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        """Not applicable for cross-encoders — runner uses score_matrix instead."""
        return np.zeros((len(texts), 1), dtype=np.float32)
