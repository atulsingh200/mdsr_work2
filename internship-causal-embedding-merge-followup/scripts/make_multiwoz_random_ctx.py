"""Create data_6/multiwoz_v24_ctx/ — MultiWOZ with randomised context depth.

Each sample in multiwoz_v24 has the full conversation history concatenated
as newline-separated turns in the `anchor` field.  This script produces a
new dataset where each sample is randomly assigned one of five context
depths (uniform, seeded for reproducibility):

  full       — entire history (unchanged)
  3_back     — last 7 lines  (3 exchange-pairs + current user turn)
  2_back     — last 5 lines  (2 exchange-pairs + current user turn)
  1_back     — last 3 lines  (1 exchange-pair  + current user turn)
  last_only  — last 1 line   (current user turn only)

If an anchor has fewer lines than the requested depth the full anchor is
kept (can't truncate past what exists).

The `positive` field is left untouched — it is always the next response.

Output: data_6/multiwoz_v24_ctx/{train,val,test}.jsonl
        Each record has an extra field `ctx_level` recording which depth
        was applied, so downstream analysis can stratify by context depth.

Usage:
    .venv/bin/python scripts/make_multiwoz_random_ctx.py
    .venv/bin/python scripts/make_multiwoz_random_ctx.py --seed 42
    .venv/bin/python scripts/make_multiwoz_random_ctx.py --split test
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data_6" / "multiwoz_v24"
DST_DIR = ROOT / "data_6" / "multiwoz_v24_ctx"

# (level_name, max_lines_to_keep)
# max_lines=None means keep all
LEVELS: list[tuple[str, int | None]] = [
    ("full",      None),
    ("3_back",    7),
    ("2_back",    5),
    ("1_back",    3),
    ("last_only", 1),
]
LEVEL_NAMES = [name for name, _ in LEVELS]


def truncate_anchor(anchor: str, max_lines: int | None) -> str:
    if max_lines is None:
        return anchor
    lines = anchor.split("\n")
    if len(lines) <= max_lines:
        return anchor
    return "\n".join(lines[-max_lines:])


def process_split(split: str, rng: random.Random) -> dict:
    src_path = SRC_DIR / f"{split}.jsonl"
    dst_path = DST_DIR / f"{split}.jsonl"

    if not src_path.exists():
        print(f"  [skip] {src_path} not found")
        return {}

    level_counts: dict[str, int] = {name: 0 for name in LEVEL_NAMES}
    records_written = 0

    with open(src_path) as fin, open(dst_path, "w") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)

            anchor = (record.get("anchor") or "").strip()
            if not anchor:
                continue

            # randomly pick a context level
            level_name, max_lines = rng.choice(LEVELS)
            record["anchor"]    = truncate_anchor(anchor, max_lines)
            record["ctx_level"] = level_name

            # track original turn count for info
            orig_turns = len(anchor.split("\n"))
            new_turns  = len(record["anchor"].split("\n"))
            record.setdefault("metadata", {})
            record["metadata"]["orig_anchor_turns"] = orig_turns
            record["metadata"]["ctx_anchor_turns"]  = new_turns

            fout.write(json.dumps(record) + "\n")
            level_counts[level_name] += 1
            records_written += 1

    return {"records": records_written, "level_counts": level_counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed",  type=int, default=0,
                    help="RNG seed for reproducibility (default 0).")
    ap.add_argument("--split", nargs="+",
                    default=["train", "val", "test"],
                    choices=["train", "val", "test"],
                    help="Which splits to process (default: all three).")
    args = ap.parse_args()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Source : {SRC_DIR}")
    print(f"Output : {DST_DIR}")
    print(f"Seed   : {args.seed}")
    print(f"Levels : {LEVEL_NAMES}\n")

    for split in args.split:
        print(f"Processing {split}...")
        stats = process_split(split, rng)
        if not stats:
            continue
        print(f"  written : {stats['records']:,} records")
        for name, count in stats["level_counts"].items():
            pct = count / stats["records"] * 100
            print(f"  {name:<12}: {count:6,}  ({pct:.1f}%)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
