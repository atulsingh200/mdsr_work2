"""Built-in baseline Retrievers.

- PretrainedRetriever: off-the-shelf HuggingFace sentence encoder. Uses the
  same encoder for anchors and candidates (symmetric — no fine-tuning).
- TfidfRetriever: classical TF-IDF cosine-similarity baseline.

Model-specific adapters (e.g. for the trained bi-encoder) live in each model
module; see `biencoder/evaluation_adapter.py`.
"""

from __future__ import annotations

import numpy as np

from .interfaces import Retriever


# ---------------------------------------------------------------------------
# Pretrained HF encoder (symmetric baseline)
# ---------------------------------------------------------------------------
class PretrainedRetriever:
    name = "pretrained"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        pooling: str = "mean",
        max_seq_length: int = 512,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        # Lazy imports so this module is importable even when torch isn't installed.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.pooling_name = pooling
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name

    def _pool(self, model_output, attention_mask):
        torch = self._torch
        if self.pooling_name == "cls":
            return model_output.last_hidden_state[:, 0]
        # mean pooling
        token_embeds = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeds * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_out = self.model(**enc)
                embeds = self._pool(model_out, enc["attention_mask"])
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            out.append(embeds.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


# ---------------------------------------------------------------------------
# TF-IDF (classical baseline, no neural component)
# ---------------------------------------------------------------------------
class TfidfRetriever:
    name = "tfidf"

    def __init__(self, max_features: int = 10000, stop_words: str | None = "english") -> None:
        # Lazy import — sklearn isn't pulled in unconditionally.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self._fitted = False
        self._all_seen: list[str] = []

    def _ensure_fit(self, texts: list[str]) -> None:
        # We need a corpus to fit on. Strategy: accumulate all texts seen via
        # encode_anchors / encode_candidates calls, refit on first use of either.
        self._all_seen.extend(texts)
        if not self._fitted:
            self._vectorizer.fit(self._all_seen)
            self._fitted = True

    def _transform(self, texts: list[str]) -> np.ndarray:
        # Already L2-normalized by sklearn (norm='l2' default).
        mat = self._vectorizer.transform(texts)
        # toarray() for dot-product downstream; small docs only.
        return mat.toarray().astype(np.float32)

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        self._ensure_fit(texts)
        return self._transform(texts)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        self._ensure_fit(texts)
        return self._transform(texts)
