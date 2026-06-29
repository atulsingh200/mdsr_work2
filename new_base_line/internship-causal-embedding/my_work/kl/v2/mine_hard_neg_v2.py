"""Mine K=8 hard negatives for CDEv2 training.

Saves hard_negatives_k8.npy (train) and test_hard_negatives_k8.npy (test)
to results/aep_causal/.  Uses the same frozen all-MiniLM-L6-v2 encoder as
the v1 miner so the negatives are comparable.

Usage:
  python v2/mine_hard_neg_v2.py              # mine K=8 for train + test
  python v2/mine_hard_neg_v2.py --k 8 --split train
  python v2/mine_hard_neg_v2.py --k 16       # mine K=16 if you want more
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
# Project root holds src/{biencoder,evaluation,followup_data}.
PROJECT_ROOT = KL_DIR.parent.parent
# Fallback to sibling internship-causal-embedding if local src is absent.
_SIBLING = PROJECT_ROOT.parent / "internship-causal-embedding"
BIENCODER_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "src" / "evaluation").exists() else _SIBLING
sys.path.insert(0, str(KL_DIR))
sys.path.insert(0, str(BIENCODER_ROOT))
sys.path.insert(0, str(BIENCODER_ROOT / "src"))

from dataset import load_pairs   # noqa: E402

OUT_DIR = KL_DIR / "results" / "aep_causal"


def mine_split(
    split: str,
    miner,
    k: int,
    chunk_size: int,
    seed: int,
) -> None:
    if split == "train":
        pairs_path = OUT_DIR / "train_pairs.jsonl"
        npy_out    = OUT_DIR / f"hard_negatives_k{k}.npy"
        info_out   = OUT_DIR / f"mining_info_k{k}.json"
    else:
        pairs_path = OUT_DIR / f"{split}_pairs.jsonl"
        npy_out    = OUT_DIR / f"{split}_hard_negatives_k{k}.npy"
        info_out   = OUT_DIR / f"{split}_mining_info_k{k}.json"

    if not pairs_path.exists():
        print(f"[{split}] pairs file not found: {pairs_path} — skipping")
        return

    print(f"\n[{split}] loading {pairs_path}")
    pairs = load_pairs(pairs_path, cap=None, seed=seed)
    N = len(pairs)
    print(f"  {N} pairs")

    positives = [p for _, p in pairs]
    t0 = time.time()
    print(f"  encoding {N} positives …")
    pos_emb = miner.encode_candidates(positives)  # (N, D), L2-normalised
    enc_time = time.time() - t0
    print(f"  encoded in {enc_time:.1f}s  dim={pos_emb.shape[1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_t = torch.from_numpy(pos_emb).to(device)

    hard_idx = np.empty((N, k), dtype=np.int32)
    t1 = time.time()
    print(f"  k-NN ({k} per anchor) on {device}  chunk={chunk_size} …")
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sims = pos_t[start:end] @ pos_t.T
        rows = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), rows] = float("-inf")
        _, top_idx = sims.topk(k, dim=1, largest=True, sorted=True)
        hard_idx[start:end] = top_idx.cpu().numpy()
    knn_time = time.time() - t1
    print(f"  k-NN done in {knn_time:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy_out, hard_idx)
    info_out.write_text(json.dumps({
        "split": split,
        "n_pairs": N,
        "k": k,
        "miner": miner.model_name,
        "enc_seconds": enc_time,
        "knn_seconds": knn_time,
        "hard_negatives_path": str(npy_out),
    }, indent=2))
    print(f"  saved {npy_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal",
                    help="Dataset name; sets OUT_DIR=results/<dataset>/.")
    ap.add_argument("--split",   default="both", choices=["train", "test", "val", "both"])
    ap.add_argument("--k",       type=int, default=8)
    ap.add_argument("--miner-model",    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--miner-pooling",  default="mean", choices=["mean", "cls"])
    ap.add_argument("--miner-max-seq",  type=int, default=256)
    ap.add_argument("--miner-batch",    type=int, default=256)
    ap.add_argument("--chunk-size",     type=int, default=1024)
    ap.add_argument("--seed",           type=int, default=0)
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = KL_DIR / "results" / args.dataset

    from evaluation.retrievers import PretrainedRetriever   # noqa: E402
    print(f"[miner] {args.miner_model}  K={args.k}")
    miner = PretrainedRetriever(
        model_name=args.miner_model,
        pooling=args.miner_pooling,
        max_seq_length=args.miner_max_seq,
        batch_size=args.miner_batch,
    )

    targets = ["train", "test"] if args.split == "both" else [args.split]
    for split in targets:
        mine_split(split=split, miner=miner, k=args.k,
                   chunk_size=args.chunk_size, seed=args.seed)

    print("\n[done]")


if __name__ == "__main__":
    main()
