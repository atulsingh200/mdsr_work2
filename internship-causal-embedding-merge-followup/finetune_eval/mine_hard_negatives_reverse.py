"""Prepare reverse-direction training data (no kNN mining needed).

For every (anchor=x, positive=y) pair the sole hard negative is x itself —
the reverse direction: enc_positive(x) should score LOW against enc_anchor(x).

This script writes:
  results_reverse/<dataset>/train_pairs.jsonl       — {"anchor": x, "positive": y, "hard_neg": x}
  results_reverse/<dataset>/test_pairs.jsonl        — same for test split
  results_reverse/<dataset>/mining_info.json

No GPU or sentence-transformer needed; just a pass over the JSONL files.

Usage:
  uv run python finetune_eval/mine_hard_negatives_reverse.py --dataset all
  uv run python finetune_eval/mine_hard_negatives_reverse.py --dataset aep_causal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune_eval.data import load_pairs                                    # noqa: E402
from finetune_eval.datasets import SPLITS, TRAIN_CAPS, list_datasets, split_path  # noqa: E402


def mine_for_dataset(dataset: str, out_dir: Path, seed: int, split: str = "train") -> dict:
    src_path = split_path(dataset, split)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"no {split} file for {dataset}: {src_path}")

    cap = None if split != "train" else TRAIN_CAPS.get(dataset)
    print(f"\n[{dataset} · {split}] loading {src_path}  (cap={cap})")
    t0 = time.time()
    pairs = load_pairs(src_path, cap=cap, seed=seed)
    N = len(pairs)
    print(f"  {N} pairs  →  hard_neg = anchor (reverse direction)")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    pairs_name = "train_pairs.jsonl" if split == "train" else f"{split}_pairs.jsonl"
    pairs_path = ds_dir / pairs_name
    with open(pairs_path, "w") as fh:
        for anchor, positive in pairs:
            fh.write(json.dumps({"anchor": anchor, "positive": positive, "hard_neg": anchor}) + "\n")

    elapsed = time.time() - t0
    info = {
        "dataset": dataset,
        "split": split,
        "n_pairs": N,
        "hard_neg_type": "reverse_direction (hard_neg = anchor)",
        "elapsed_seconds": elapsed,
        "pairs_path": str(pairs_path),
    }
    info_name = "mining_info.json" if split == "train" else f"{split}_mining_info.json"
    (ds_dir / info_name).write_text(json.dumps(info, indent=2))
    print(f"  saved {pairs_path}  ({elapsed:.1f}s)")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    help="Dataset name, or 'all' for every entry in datasets.py.")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "val", "test"],
                    help="Which splits to process.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "finetune_eval/results_reverse"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    targets = list_datasets() if args.dataset == "all" else [args.dataset]
    for t in targets:
        if t not in SPLITS:
            raise SystemExit(f"unknown dataset: {t!r}")

    summaries: list[dict] = []
    for ds in targets:
        for split in args.splits:
            try:
                summaries.append(mine_for_dataset(ds, out_dir, args.seed, split=split))
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                summaries.append({"dataset": ds, "split": split, "error": str(e)})

    out_path = out_dir / "mining_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
