"""Quick stats on a dataset: pair count, length distributions, sample triples.

Usage:
    uv run python scripts/inspect_dataset.py qrecc
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from followup_data import default_root, load
from followup_data.negatives import with_random_negatives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--root", default=str(default_root()))
    p.add_argument("--limit", type=int, default=5000)
    args = p.parse_args()

    try:
        ds = load(args.dataset, root=Path(args.root), split=args.split)
    except ValueError:
        ds = load(args.dataset, root=Path(args.root))

    if not ds.is_downloaded():
        print(f"{args.dataset}: not downloaded; running download()...")
        ds.download()

    pairs = []
    for ex in ds:
        pairs.append(ex)
        if len(pairs) >= args.limit:
            break

    if not pairs:
        print(f"{args.dataset}: no pairs produced")
        return

    a_lens = [len(ex.anchor) for ex in pairs]
    p_lens = [len(ex.positive) for ex in pairs]
    print(f"=== {args.dataset} [{args.split}] ===")
    print(f"pairs sampled : {len(pairs)} (limit={args.limit})")
    print(f"anchor   chars: mean={statistics.mean(a_lens):.0f} "
          f"median={statistics.median(a_lens):.0f} max={max(a_lens)}")
    print(f"positive chars: mean={statistics.mean(p_lens):.0f} "
          f"median={statistics.median(p_lens):.0f} max={max(p_lens)}")

    print("\n--- 3 sample triples ---")
    for t in list(with_random_negatives(pairs, k=2, seed=0))[:3]:
        print(f"\nA: {t.anchor[:200]}")
        print(f"P: {t.positive[:200]}")
        for i, n in enumerate(t.negatives):
            print(f"N{i}: {n[:200]}")


if __name__ == "__main__":
    main()
