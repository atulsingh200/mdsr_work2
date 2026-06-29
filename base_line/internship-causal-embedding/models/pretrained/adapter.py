"""PretrainedRetriever and TfidfRetriever: off-the-shelf baselines.

These require no trained checkpoint — they are symmetric (same encoder
for anchors and candidates) and serve as zero-shot baselines.
"""

from __future__ import annotations

import numpy as np


class PretrainedRetriever:
    """Off-the-shelf HuggingFace sentence encoder (symmetric, no fine-tuning)."""

    def __init__(
        self,
        hf_id: str = "BAAI/bge-small-en-v1.5",
        pooling: str = "mean",
        max_seq_length: int = 512,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
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

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).to(self.device)
        self.model.eval()
        self.name = f"pretrained_{hf_id.replace('/', '_')}"

    def _pool(self, model_output, attention_mask):
        torch = self._torch
        if self.pooling_name == "cls":
            return model_output.last_hidden_state[:, 0]
        token_embeds = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_out = self.model(**enc)
                embeds = self._pool(model_out, enc["attention_mask"])
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            out.append(embeds.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros(
            (0, self.model.config.hidden_size), dtype=np.float32
        )

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class TfidfRetriever:
    """Classical TF-IDF cosine similarity (no neural component)."""

    name = "tfidf"

    def __init__(
        self,
        max_features: int = 10000,
        stop_words: str | None = "english",
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            norm="l2",
        )
        self._fitted = False
        self._corpus: list[str] = []

    def _fit_if_needed(self, texts: list[str]) -> None:
        self._corpus.extend(texts)
        if not self._fitted:
            self._vectorizer.fit(self._corpus)
            self._fitted = True

    def _transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.transform(texts).toarray().astype(np.float32)

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        self._fit_if_needed(texts)
        return self._transform(texts)

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        self._fit_if_needed(texts)
        return self._transform(texts)
