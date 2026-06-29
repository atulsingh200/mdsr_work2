"""Build a 50/50 mixed dataset from semantic and indomain hard-negative splits.

For train and val: takes floor(N/2) rows from each source (both have equal N),
shuffles them together, writes to --out-dir. Test is copied from --semantic-dir
unchanged.

Usage:
    python3 -m src.classifier.crossencoder.build_mixed_dataset \
        --semantic-dir data/aep_causal_classification_hard_neg_semantic \
        --indomain-dir data/aep_causal_classification_hard_neg_indomain \
        --out-dir data/aep_causal_classification_hard_neg_mixed
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def mix_split(sem_path: Path, ind_path: Path, out_path: Path, seed: int) -> int:
    sem_lines = sem_path.read_text().splitlines()
    ind_lines = ind_path.read_text().splitlines()

    rng = random.Random(seed)
    rng.shuffle(sem_lines)
    rng.shuffle(ind_lines)

    half_sem = len(sem_lines) // 2
    half_ind = len(ind_lines) // 2
    mixed = sem_lines[:half_sem] + ind_lines[:half_ind]
    rng.shuffle(mixed)

    out_path.write_text("\n".join(mixed) + "\n")
    return len(mixed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-dir", required=True)
    ap.add_argument("--indomain-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files.")
    args = ap.parse_args()

    sem_dir = Path(args.semantic_dir)
    ind_dir = Path(args.indomain_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        out_path = out_dir / f"directional_{split}.jsonl"
        if out_path.exists() and not args.force:
            print(f"  skipping {split} (already exists; use --force to overwrite)")
            continue
        n = mix_split(
            sem_dir / f"directional_{split}.jsonl",
            ind_dir / f"directional_{split}.jsonl",
            out_path,
            seed=args.seed,
        )
        print(f"  {split}: wrote {n} rows -> {out_path}")

    test_dst = out_dir / "directional_test.jsonl"
    if test_dst.exists() and not args.force:
        print(f"  skipping test (already exists; use --force to overwrite)")
    else:
        shutil.copy2(sem_dir / "directional_test.jsonl", test_dst)
        print(f"  test: copied from {sem_dir / 'directional_test.jsonl'}")


if __name__ == "__main__":
    main()
