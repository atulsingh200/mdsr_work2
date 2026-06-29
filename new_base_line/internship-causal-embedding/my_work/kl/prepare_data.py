"""Prepare aep_causal JSONL splits for CDE training.

Reads raw JSONL files from kl/aep_causal/{train,val,test}.jsonl
which have 'text_1_causal' and 'text_2_target' fields, and writes
anchor/positive pairs to results/aep_causal/.

Usage:
  python prepare_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

KL_DIR = Path(__file__).resolve().parent
RAW_DIR = KL_DIR / "aep_causal"
OUT_DIR = KL_DIR / "results" / "aep_causal"

SPLITS = {
    "train": ("train.jsonl",       "train_pairs.jsonl"),
    "val":   ("val.jsonl",         "val_pairs.jsonl"),
    "test":  ("test.jsonl",        "test_pairs.jsonl"),
}


def convert_split(src: Path, dst: Path) -> int:
    rows_written = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            anchor = (row.get("text_1_causal") or row.get("anchor") or "").strip()
            positive = (row.get("text_2_target") or row.get("positive") or "").strip()
            if not anchor or not positive:
                continue
            out = {"anchor": anchor, "positive": positive}
            # Preserve useful metadata for analysis.
            for key in ("pair_id", "causal_type", "confidence", "anchor_text",
                        "source_doc_title", "target_doc_title"):
                if key in row:
                    out[key] = row[key]
            fout.write(json.dumps(out) + "\n")
            rows_written += 1
    return rows_written


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, (src_name, dst_name) in SPLITS.items():
        src = RAW_DIR / src_name
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue
        dst = OUT_DIR / dst_name
        n = convert_split(src, dst)
        print(f"  [{split}] {n} pairs  ->  {dst}")


if __name__ == "__main__":
    main()
