"""Mine semantic hard negatives via a frozen sentence encoder.

For each (anchor_i, positive_i) training pair, find the K most semantically
similar OTHER positives (y* ≠ y_i) and use their indices as hard negatives.
The miner encodes once per dataset and writes a single
`hard_negatives.npy` of shape `(N, K)` (int32 indices into the positives
list of the training file).

Distance metric: cosine similarity (encoder embeds are L2-normalized).
Default model: sentence-transformers/all-MiniLM-L6-v2.

Usage:
  .venv/bin/python finetune_eval/mine_hard_negatives.py \\
      --dataset aep_causal --k 4
  .venv/bin/python finetune_eval/mine_hard_negatives.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.retrievers import PretrainedRetriever            # noqa: E402

from finetune_eval.data import load_pairs                         # noqa: E402
from finetune_eval.datasets import SPLITS, TRAIN_CAPS, list_datasets, split_path  # noqa: E402


def mine_for_dataset(
    dataset: str,
    miner: PretrainedRetriever,
    k: int,
    out_dir: Path,
    seed: int,
    chunk_size: int,
    split: str = "train",
    no_cap: bool = False,
) -> dict:
    src_path = split_path(dataset, split)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"no {split} file for {dataset}: {src_path}")

    # Cap only applies on the train split; test/val are used in full.
    # --no-cap forces the full split even when TRAIN_CAPS has a value.
    if no_cap or split != "train":
        cap = None
    else:
        cap = TRAIN_CAPS.get(dataset)
    print(f"\n[{dataset} · {split}] loading {src_path}  (cap={cap})")
    pairs = load_pairs(src_path, cap=cap, seed=seed)
    N = len(pairs)
    print(f"  {N} pairs")

    positives = [p for _, p in pairs]
    t0 = time.time()
    print(f"  encoding {N} positives with miner …")
    pos_emb = miner.encode_candidates(positives)  # (N, D), L2-normalized
    enc_time = time.time() - t0
    print(f"  encoded in {enc_time:.1f}s  (dim={pos_emb.shape[1]})")

    # k+1 NN — we'll drop self.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_t = torch.from_numpy(pos_emb).to(device)

    hard_idx = np.empty((N, k), dtype=np.int32)

    t1 = time.time()
    print(f"  computing {k}-NN over {N}×{N} corpus on {device} (chunks of {chunk_size}) …")
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # similarity of this query chunk to ALL positives
        sims = pos_t[start:end] @ pos_t.T                # (b, N)
        # Mask self.
        rows = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), rows] = float("-inf")
        # top-K
        _, top_idx = sims.topk(k, dim=1, largest=True, sorted=True)  # (b, k)
        hard_idx[start:end] = top_idx.cpu().numpy()
    knn_time = time.time() - t1
    print(f"  kNN computed in {knn_time:.1f}s")

    # Output file naming: train split keeps the original name for backwards-
    # compat; other splits get a "<split>_" prefix so train + test artifacts
    # coexist in the same dataset folder.
    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    if split == "train":
        npy_name = "hard_negatives.npy"
        pairs_name = "train_pairs.jsonl"
        info_name = "mining_info.json"
    else:
        npy_name = f"{split}_hard_negatives.npy"
        pairs_name = f"{split}_pairs.jsonl"
        info_name = f"{split}_mining_info.json"
    npy_path = ds_dir / npy_name
    np.save(npy_path, hard_idx)
    pairs_path = ds_dir / pairs_name
    with open(pairs_path, "w") as fh:
        for a, p in pairs:
            fh.write(json.dumps({"anchor": a, "positive": p}) + "\n")

    info = {
        "dataset": dataset,
        "split": split,
        "n_pairs": N,
        "k_hard_negatives": k,
        "encoding_seconds": enc_time,
        "knn_seconds": knn_time,
        "miner": miner.model_name,
        "hard_negatives_path": str(npy_path),
        "pairs_path": str(pairs_path),
    }
    (ds_dir / info_name).write_text(json.dumps(info, indent=2))
    print(f"  saved {npy_path} and {pairs_path}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    help="Dataset name, or 'all' for every entry in datasets.py.")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="Which split to mine on (train caps still apply; test/val use full file).")
    ap.add_argument("--k", type=int, default=4, help="Hard negatives per anchor.")
    ap.add_argument("--no-cap", action="store_true",
                    help="Ignore TRAIN_CAPS — mine on the entire train split.")
    ap.add_argument("--miner-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HF model id for semantic similarity (frozen).")
    ap.add_argument("--miner-pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--miner-max-seq-length", type=int, default=256)
    ap.add_argument("--miner-batch-size", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="Query rows per GPU chunk during kNN.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "finetune_eval/results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    targets = list_datasets() if args.dataset == "all" else [args.dataset]
    for t in targets:
        if t not in SPLITS:
            raise SystemExit(f"unknown dataset: {t!r}")

    print(f"[load miner] {args.miner_model}")
    miner = PretrainedRetriever(
        model_name=args.miner_model,
        pooling=args.miner_pooling,
        max_seq_length=args.miner_max_seq_length,
        batch_size=args.miner_batch_size,
    )

    summaries = []
    for ds in targets:
        try:
            summaries.append(mine_for_dataset(
                ds, miner, args.k, out_dir, args.seed, args.chunk_size,
                split=args.split, no_cap=args.no_cap,
            ))
        except Exception as e:
            print(f"  ERROR ({type(e).__name__}): {e}")
            summaries.append({"dataset": ds, "error": f"{type(e).__name__}: {e}"})

    summary_name = "mining_summary.json" if args.split == "train" else f"{args.split}_mining_summary.json"
    out_path = out_dir / summary_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
