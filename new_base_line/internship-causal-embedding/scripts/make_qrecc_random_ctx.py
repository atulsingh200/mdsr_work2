"""Create data_6/qrecc_ctx/ — QReCC with randomised context depth.

QReCC anchors are already resolved Q+A pairs (one per record):
    "[Rewritten question]? [Answer to that question]"

Unlike MultiWOZ, history is NOT embedded in a single anchor — every record
is self-contained.  To add more history this script groups records by
conversation_no, sorts by turn_no, and concatenates previous Q+A pairs.

Five context levels (uniform random, seeded):

  full       — all Q+A pairs from turn 2 .. current turn  (newline-joined)
  3_back     — last 3 Q+A pairs  (current + 2 preceding)
  2_back     — last 2 Q+A pairs  (current + 1 preceding)
  1_back     — current Q+A only  (= original anchor, unchanged)
  last_only  — question part of current anchor only (text up to first "?")

If fewer turns exist than requested, all available turns are used.

Output: data_6/qrecc_ctx/{train,test}.jsonl
        Each record carries ctx_level and orig/ctx turn counts in metadata.

Usage:
    .venv/bin/python scripts/make_qrecc_random_ctx.py
    .venv/bin/python scripts/make_qrecc_random_ctx.py --seed 42
    .venv/bin/python scripts/make_qrecc_random_ctx.py --split test
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data_6" / "qrecc"
DST_DIR = ROOT / "data_6" / "qrecc_ctx"

# (level_name, max_qa_pairs_to_keep)   None = keep all
LEVELS: list[tuple[str, int | None]] = [
    ("full",      None),
    ("3_back",    3),
    ("2_back",    2),
    ("1_back",    1),
    ("last_only", 0),   # 0 = question-only, special case
]
LEVEL_NAMES = [n for n, _ in LEVELS]


def extract_question(anchor: str) -> str:
    """Return only the question part of an anchor (text up to first '?')."""
    idx = anchor.find("?")
    if idx == -1:
        return anchor          # no "?" found — return as-is
    return anchor[: idx + 1].strip()


def build_anchor(history: list[str], max_pairs: int | None, question_only: bool) -> str:
    """Build the anchor from a list of previous Q+A anchors + current anchor.

    history[-1] is the current-turn anchor; history[:-1] are earlier turns.
    """
    current = history[-1]

    if question_only:
        return extract_question(current)

    if max_pairs is None:
        selected = history              # full history
    else:
        selected = history[-max_pairs:] # last N pairs (includes current)

    return "\n".join(selected)


def load_split(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def process_split(split: str, rng: random.Random) -> dict:
    src_path = SRC_DIR / f"{split}.jsonl"
    dst_path = DST_DIR / f"{split}.jsonl"

    if not src_path.exists():
        print(f"  [skip] {src_path} not found")
        return {}

    records = load_split(src_path)

    # group by conversation, sort by turn
    conversations: dict[int, list[dict]] = {}
    for r in records:
        cno = r["conversation_no"]
        conversations.setdefault(cno, []).append(r)
    for cno in conversations:
        conversations[cno].sort(key=lambda x: x["turn_no"])

    level_counts: dict[str, int] = {n: 0 for n in LEVEL_NAMES}
    out_records: list[dict] = []

    for cno, turns in conversations.items():
        # build cumulative history of anchors: history[k] = anchors[0..k]
        anchor_history: list[str] = []
        for record in turns:
            anchor_history.append(record["anchor"])
            orig_turns = len(anchor_history)

            level_name, max_pairs = rng.choice(LEVELS)
            question_only = (max_pairs == 0)

            new_anchor = build_anchor(
                anchor_history,
                max_pairs=max_pairs,
                question_only=question_only,
            )
            ctx_turns = len(new_anchor.split("\n"))

            out = dict(record)
            out["anchor"]    = new_anchor
            out["ctx_level"] = level_name
            out.setdefault("metadata", {})
            out["metadata"] = dict(out.get("metadata") or {})
            out["metadata"]["orig_anchor_turns"] = orig_turns
            out["metadata"]["ctx_anchor_turns"]  = ctx_turns

            out_records.append(out)
            level_counts[level_name] += 1

    # write in original order (sort by conversation_no, turn_no)
    out_records.sort(key=lambda r: (r["conversation_no"], r["turn_no"]))
    with open(dst_path, "w") as fout:
        for r in out_records:
            fout.write(json.dumps(r) + "\n")

    return {"records": len(out_records), "level_counts": level_counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--split", nargs="+", default=["train", "test"],
                    choices=["train", "test"])
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
