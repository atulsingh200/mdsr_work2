"""Semantic hard negative mining for the cause→effect GNN.

Mirrors the approach in finetune_eval/mine_hard_negatives.py:
  For each pair i (anchor_i, positive_i), find K other positives
  that are most similar to positive_i in cosine space — these become
  hard negatives because they look like the real effect but are not.

Extra filter (compared to finetune_eval):
  Skip candidate pair j if sim(anchor_i, anchor_j) > anchor_sim_threshold.
  Rationale: if two anchors are nearly identical, positive_j may actually
  be a valid effect of anchor_i — using it as a hard negative would be wrong
  and would give the model a contradictory training signal.

Memory safety:
  Chunk-based GPU computation avoids N×N matrix materialisation.
  Falls back to CPU if CUDA unavailable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _check_memory(label: str = "") -> None:
    """Print CPU memory usage; abort if available RAM < 4 GB."""
    import psutil
    vm = psutil.virtual_memory()
    avail_gb = vm.available / 1024 ** 3
    total_gb = vm.total / 1024 ** 3
    used_gb = vm.used / 1024 ** 3
    print(f"  [mem{(' ' + label) if label else ''}] "
          f"used={used_gb:.1f}GB  avail={avail_gb:.1f}GB / {total_gb:.0f}GB total")
    if avail_gb < 4.0:
        raise MemoryError(
            f"CPU RAM critically low: {avail_gb:.1f}GB available — aborting to prevent OOM."
        )


def encode_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    max_seq_length: int = 256,
) -> np.ndarray:
    """Encode texts into L2-normalised embeddings with a frozen sentence encoder.

    Returns: float32 ndarray of shape (N, D).
    """
    from evaluation.retrievers import PretrainedRetriever  # noqa: E402
    _check_memory("before encoding")
    retriever = PretrainedRetriever(
        model_name=model_name,
        pooling="mean",
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    t0 = time.time()
    emb = retriever.encode_candidates(texts)  # (N, D), L2-normalised float32
    print(f"  encoded {len(texts)} texts in {time.time() - t0:.1f}s  dim={emb.shape[1]}")
    _check_memory("after encoding")
    return emb


def mine_hard_negatives(
    anchors: list[str],
    positives: list[str],
    k: int = 4,
    anchor_sim_threshold: float = 0.8,
    chunk_size: int = 1024,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    max_seq_length: int = 256,
) -> np.ndarray:
    """Mine semantic hard negatives with an anchor-similarity filter.

    Algorithm (mirrors finetune_eval/mine_hard_negatives.py):
      1. Encode all positives → pos_emb  [N, D]
      2. Encode all anchors  → anc_emb  [N, D]
      3. For each query i, compute cosine sim to every other positive.
      4. Mask self (i == j).
      5. Mask pairs j where sim(anchor_i, anchor_j) > threshold — those
         anchors are too similar; positive_j may be a true effect of i.
      6. Take top-k remaining as hard negatives.

    Args:
        anchors:               N anchor texts.
        positives:             N positive (effect) texts, aligned with anchors.
        k:                     Hard negatives per anchor.
        anchor_sim_threshold:  Cosine threshold; pairs above this are excluded.
        chunk_size:            GPU chunk size for N×N computation.
        model_name:            Sentence encoder for mining (frozen).
        batch_size:            Batch size for sentence encoding.
        max_seq_length:        Max tokens for the encoder.

    Returns:
        hard_idx: int32 ndarray [N, k] — indices into the positives list.
    """
    N = len(anchors)
    assert len(positives) == N, "anchors and positives must have equal length"

    print(f"\n[mine_hard_negatives] N={N}  k={k}  anchor_threshold={anchor_sim_threshold}")
    _check_memory("start")

    print("  encoding positives …")
    pos_emb = encode_texts(positives, model_name, batch_size, max_seq_length)
    print("  encoding anchors …")
    anc_emb = encode_texts(anchors, model_name, batch_size, max_seq_length)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  computing {k}-NN on {device}  (chunks of {chunk_size}) …")

    pos_t = torch.from_numpy(pos_emb).to(device)  # [N, D]
    anc_t = torch.from_numpy(anc_emb).to(device)  # [N, D]

    hard_idx = np.empty((N, k), dtype=np.int32)
    n_filtered_total = 0
    t1 = time.time()

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        b = end - start

        # Cosine similarity: query positives vs ALL positives
        pos_sims = pos_t[start:end] @ pos_t.T               # [b, N]

        # Cosine similarity: query anchors vs ALL anchors
        anc_sims = anc_t[start:end] @ anc_t.T               # [b, N]

        # --- Mask self ---
        rows = torch.arange(b, device=device)
        cols = torch.arange(start, end, device=device)
        pos_sims[rows, cols] = float("-inf")

        # --- Mask anchor-too-similar candidates ---
        n_filtered = int((anc_sims > anchor_sim_threshold).sum().item()) - b  # exclude self
        n_filtered_total += max(n_filtered, 0)
        pos_sims[anc_sims > anchor_sim_threshold] = float("-inf")

        # top-k (may include -inf if too few candidates, but with N≈5K that's rare)
        vals, top_idx = pos_sims.topk(k, dim=1, largest=True, sorted=True)  # [b, k]

        # Sanity check: if any hard negative has -inf score, replace with random valid idx
        bad = (vals == float("-inf"))
        if bad.any():
            for bi in range(b):
                for ki in range(k):
                    if bad[bi, ki]:
                        # fallback: pick a random non-self, non-anchor-similar index
                        for attempt in range(100):
                            r = int(torch.randint(0, N, (1,)).item())
                            if r != start + bi:
                                top_idx[bi, ki] = r
                                break

        hard_idx[start:end] = top_idx.cpu().numpy()

        _check_memory(f"chunk {start//chunk_size + 1}")

    knn_time = time.time() - t1
    print(f"  kNN done in {knn_time:.1f}s  |  anchor-filtered slots: {n_filtered_total}")
    return hard_idx


def load_and_mine(
    pairs: list[tuple[str, str]],
    k: int = 4,
    anchor_sim_threshold: float = 0.8,
    out_dir: Path | None = None,
    dataset_name: str = "dataset",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1024,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Convenience wrapper: mine hard negatives and optionally cache to disk.

    Returns (hard_idx, pairs) where hard_idx is [N, k] int32.
    """
    import json

    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]

    if out_dir is not None:
        out_dir = Path(out_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / "gnn_hard_negatives.npy"
        pairs_path = out_dir / "gnn_train_pairs.jsonl"
        info_path = out_dir / "gnn_mining_info.json"

        if npy_path.exists() and pairs_path.exists():
            print(f"[load_and_mine] loading cached negatives from {npy_path}")
            hard_idx = np.load(npy_path)
            return hard_idx, pairs

    hard_idx = mine_hard_negatives(
        anchors, positives,
        k=k,
        anchor_sim_threshold=anchor_sim_threshold,
        chunk_size=chunk_size,
        model_name=model_name,
    )

    if out_dir is not None:
        np.save(npy_path, hard_idx)
        with open(pairs_path, "w") as fh:
            for a, p in pairs:
                fh.write(json.dumps({"anchor": a, "positive": p}) + "\n")
        info = {
            "n_pairs": len(pairs),
            "k": k,
            "anchor_sim_threshold": anchor_sim_threshold,
            "model": model_name,
        }
        info_path.write_text(json.dumps(info, indent=2))
        print(f"  saved → {npy_path}")

    return hard_idx, pairs
