"""Hard negative miner for Lorentz encoder.

Two-phase approach (same as finetune_eval/mine_hard_negatives.py):
1. Encode all positives with a frozen semantic encoder (all-MiniLM-L6-v2).
2. For each anchor i, find top-K nearest OTHER positives by cosine similarity.
   These are semantically similar passages that are NOT causal continuations → hard negatives.

New addition: anchor-similarity filter.
If anchor_j is too similar to anchor_i (cos > anchor_sim_threshold), positive_j
might actually be a valid follow-up for anchor_i. We skip those candidates to
avoid poisoning the training signal with false negatives.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


class SemanticHardNegMiner:
    """
    Offline semantic hard-negative miner.

    Replicates the finetune_eval/mine_hard_negatives.py logic:
      - Encode all positives once with a frozen sentence encoder.
      - k-NN search in positive space for each anchor.
      - Filter by: (a) not the anchor's own positive; (b) anchor similarity filter.

    Args:
        encoder_model:  HF model id for the frozen encoder (sentence-transformers).
        k:              How many hard negatives per anchor to keep.
        anchor_sim_threshold: If cos(anchor_i, anchor_j) > this, skip positive_j
                              (the anchors are so similar that j might be a true
                              positive for i, making it a false negative).
        chunk_size:     Chunk size for batched GPU kNN.
        batch_size:     Encoding batch size.
        max_seq_length: Max sequence length for the frozen encoder.
        pooling:        Pooling strategy for the frozen encoder.
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 7,
        anchor_sim_threshold: float = 0.85,
        chunk_size: int = 1024,
        batch_size: int = 256,
        max_seq_length: int = 256,
        pooling: str = "mean",
    ):
        self.encoder_model = encoder_model
        self.k = k
        self.anchor_sim_threshold = anchor_sim_threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.pooling = pooling
        self._miner = None  # lazy-loaded

    def _get_miner(self):
        if self._miner is None:
            import sys
            from pathlib import Path
            ROOT = Path(__file__).resolve().parent.parent.parent.parent
            sys.path.insert(0, str(ROOT / "src"))
            from evaluation.retrievers import PretrainedRetriever
            self._miner = PretrainedRetriever(
                model_name=self.encoder_model,
                pooling=self.pooling,
                max_seq_length=self.max_seq_length,
                batch_size=self.batch_size,
            )
        return self._miner

    def mine(
        self,
        pairs: List[Tuple[str, str]],
        out_dir: Path,
        split: str = "train",
    ) -> Tuple[np.ndarray, List]:
        """
        Mine hard negatives for a list of (anchor, positive) pairs.

        Returns:
            hard_idx: (N, k) int32 array — indices into positives list.
            pairs:    the pairs list (same as input, possibly capped).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        N = len(pairs)
        anchors = [a for a, _ in pairs]
        positives = [p for _, p in pairs]

        miner = self._get_miner()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  [HardNegMiner] encoding {N} positives…")
        t0 = time.time()
        pos_emb = miner.encode_candidates(positives)  # (N, D), L2-normalized
        print(f"  encoded in {time.time()-t0:.1f}s  dim={pos_emb.shape[1]}")

        print(f"  [HardNegMiner] encoding {N} anchors…")
        t1 = time.time()
        anc_emb = miner.encode_candidates(anchors)   # (N, D), L2-normalized
        print(f"  encoded in {time.time()-t1:.1f}s")

        pos_t = torch.from_numpy(pos_emb).to(device)
        anc_t = torch.from_numpy(anc_emb).to(device)

        # Compute k-NN on positive embeddings, with anchor-similarity filter
        hard_idx = np.full((N, self.k), -1, dtype=np.int32)

        print(f"  [HardNegMiner] kNN with anchor-sim filter (threshold={self.anchor_sim_threshold}) …")
        t2 = time.time()
        top_k_extra = min(self.k * 10 + 50, N - 1)  # over-retrieve to survive filtering

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk_size = end - start

            # Positive-space similarity
            pos_sims = pos_t[start:end] @ pos_t.T           # (b, N)
            # Mask self
            rows = torch.arange(start, end, device=device)
            pos_sims[torch.arange(chunk_size, device=device), rows] = float("-inf")

            # Anchor similarity: cos(anchor_i, anchor_j) for all j
            anc_sims = anc_t[start:end] @ anc_t.T           # (b, N)

            # For each query in chunk, apply filter and pick top-k
            _, top_candidates = pos_sims.topk(top_k_extra, dim=1, largest=True, sorted=True)

            for bi in range(chunk_size):
                gi = start + bi
                kept = []
                for cand_j in top_candidates[bi].cpu().numpy():
                    if cand_j == gi:
                        continue
                    # Anchor similarity filter: skip if anchors too similar
                    asim = float(anc_sims[bi, cand_j].item())
                    if asim > self.anchor_sim_threshold:
                        continue
                    kept.append(cand_j)
                    if len(kept) == self.k:
                        break
                # Fill what we found (pad with -1 if fewer than k survived)
                for ki, idx in enumerate(kept):
                    hard_idx[gi, ki] = idx

        print(f"  kNN+filter done in {time.time()-t2:.1f}s")
        coverage = (hard_idx != -1).all(axis=1).mean()
        print(f"  full-k coverage: {coverage:.1%}  (anchor_sim_threshold={self.anchor_sim_threshold})")

        # Save artifacts
        suffix = "" if split == "train" else f"{split}_"
        npy_path = out_dir / f"{suffix}hard_negatives.npy"
        pairs_path = out_dir / f"{suffix}pairs.jsonl"
        np.save(npy_path, hard_idx)
        with open(pairs_path, "w") as fh:
            for a, p in pairs:
                fh.write(json.dumps({"anchor": a, "positive": p}) + "\n")
        print(f"  saved {npy_path}")

        return hard_idx, pairs


def load_hard_negatives(out_dir: Path, split: str = "train"):
    """Load precomputed hard negatives and pairs."""
    out_dir = Path(out_dir)
    suffix = "" if split == "train" else f"{split}_"
    npy_path = out_dir / f"{suffix}hard_negatives.npy"
    pairs_path = out_dir / f"{suffix}pairs.jsonl"
    hard_idx = np.load(npy_path)
    pairs = []
    with open(pairs_path) as fh:
        for line in fh:
            row = json.loads(line.strip())
            pairs.append((row["anchor"], row["positive"]))
    return hard_idx, pairs


class HardNegTripleDataset(torch.utils.data.Dataset):
    """Dataset that yields (anchor, positive, [hard_negatives]).

    hard_idx[i, k] is an index into the positives list.
    -1 means no valid hard negative at that slot → replaced with positive (masked out in loss).
    """

    def __init__(self, pairs, hard_idx):
        self.anchors = [a for a, _ in pairs]
        self.positives = [p for _, p in pairs]
        self.hard_idx = hard_idx  # (N, K)

    def __len__(self):
        return len(self.anchors)

    @property
    def k(self):
        return self.hard_idx.shape[1]

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        hardnegs = []
        neg_valid = []
        for j in range(self.k):
            hi = int(self.hard_idx[idx, j])
            if hi >= 0:
                hardnegs.append(self.positives[hi])
                neg_valid.append(True)
            else:
                hardnegs.append(positive)   # placeholder
                neg_valid.append(False)
        return anchor, positive, hardnegs, neg_valid


def make_lorentz_collate(tokenizer, max_length: int = 256):
    """Collate (anchor, positive, hardnegs, neg_valid) into batch tensors."""
    from biencoder.model import tokenize_texts

    def collate(batch):
        anchors = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        all_hardnegs = [b[2] for b in batch]
        all_valid = [b[3] for b in batch]
        K = len(all_hardnegs[0])
        B = len(batch)

        a_ids, a_mask = tokenize_texts(tokenizer, anchors, max_length)
        p_ids, p_mask = tokenize_texts(tokenizer, positives, max_length)

        flat_negs = [t for row in all_hardnegs for t in row]
        n_ids, n_mask = tokenize_texts(tokenizer, flat_negs, max_length)

        # neg validity mask (B, K)
        neg_valid_t = torch.tensor(all_valid, dtype=torch.bool)  # (B, K)

        return a_ids, a_mask, p_ids, p_mask, n_ids, n_mask, neg_valid_t, K

    return collate
