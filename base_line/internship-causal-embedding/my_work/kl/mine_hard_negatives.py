"""Mine semantic hard negatives for the aep_causal dataset.

For each anchor-positive pair (a_i, p_i), finds the K positives p_j
(j != i) most similar to p_i under a frozen sentence encoder.  The
resulting index matrix (N, K) is the hard negatives used by train_cde.py.

Mines both the train and test splits so evaluation can also use hard negs.

Usage:
  python mine_hard_negatives.py            # train + test, K=4
  python mine_hard_negatives.py --k 4 --split train
  python mine_hard_negatives.py --k 4 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

KL_DIR = Path(__file__).resolve().parent
BIENCODER_ROOT = KL_DIR.parent.parent / "internship-causal-embedding"
sys.path.insert(0, str(BIENCODER_ROOT))
sys.path.insert(0, str(BIENCODER_ROOT / "src"))

from evaluation.retrievers import PretrainedRetriever   # noqa: E402

from dataset import load_pairs  # noqa: E402

OUT_DIR = KL_DIR / "results" / "aep_causal"


def mine_split(
    split: str,
    miner: "PretrainedRetriever",
    k: int,
    chunk_size: int,
    seed: int,
    no_cap: bool,
) -> dict:
    if split == "train":
        pairs_path = OUT_DIR / "train_pairs.jsonl"
        npy_out = OUT_DIR / "hard_negatives.npy"
        info_out = OUT_DIR / "mining_info.json"
    else:
        pairs_path = OUT_DIR / f"{split}_pairs.jsonl"
        npy_out = OUT_DIR / f"{split}_hard_negatives.npy"
        info_out = OUT_DIR / f"{split}_mining_info.json"

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"[{split}] pairs file not found: {pairs_path}\n"
            "Run prepare_data.py first."
        )

    print(f"\n[{split}] loading {pairs_path}")
    pairs = load_pairs(pairs_path, cap=None, seed=seed)
    N = len(pairs)
    print(f"  {N} pairs")

    positives = [p for _, p in pairs]
    t0 = time.time()
    print(f"  encoding {N} positives …")
    pos_emb = miner.encode_candidates(positives)   # (N, D), L2-normalised
    enc_time = time.time() - t0
    print(f"  encoded in {enc_time:.1f}s  dim={pos_emb.shape[1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_t = torch.from_numpy(pos_emb).to(device)

    hard_idx = np.empty((N, k), dtype=np.int32)
    t1 = time.time()
    print(f"  k-NN on {device} (chunk={chunk_size}) …")
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sims = pos_t[start:end] @ pos_t.T            # (b, N)
        rows = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), rows] = float("-inf")
        _, top_idx = sims.topk(k, dim=1, largest=True, sorted=True)
        hard_idx[start:end] = top_idx.cpu().numpy()
    knn_time = time.time() - t1
    print(f"  k-NN done in {knn_time:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy_out, hard_idx)
    info = {
        "split": split,
        "n_pairs": N,
        "k": k,
        "miner": miner.model_name,
        "enc_seconds": enc_time,
        "knn_seconds": knn_time,
        "hard_negatives_path": str(npy_out),
        "pairs_path": str(pairs_path),
    }
    info_out.write_text(json.dumps(info, indent=2))
    print(f"  saved {npy_out}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "val", "both"],
                    help="Which split(s) to mine (default: both train+test).")
    ap.add_argument("--k", type=int, default=4, help="Hard negatives per anchor.")
    ap.add_argument("--miner-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--miner-pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--miner-max-seq-length", type=int, default=256)
    ap.add_argument("--miner-batch-size", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[miner] {args.miner_model}")
    miner = PretrainedRetriever(
        model_name=args.miner_model,
        pooling=args.miner_pooling,
        max_seq_length=args.miner_max_seq_length,
        batch_size=args.miner_batch_size,
    )

    targets = ["train", "test"] if args.split == "both" else [args.split]
    for split in targets:
        mine_split(
            split=split,
            miner=miner,
            k=args.k,
            chunk_size=args.chunk_size,
            seed=args.seed,
            no_cap=False,
        )

    print("\n[done]")


if __name__ == "__main__":
    main()
