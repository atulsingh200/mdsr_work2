#!/usr/bin/env python3
"""
Convert aep_causal_classification_34 to exponential soft labels.

Forward-direction pairs (original label=1):
    gap  = abs(tier_1 - tier_2)
    base = 0.7 ** (1/13)        # so gap=14 -> exactly 0.7
    soft_label = base ** (gap - 1)

    gap=1  -> 1.0
    gap=7  -> ~0.834
    gap=14 -> 0.7

Reverse-direction pairs (original label=0): soft_label = 0.0

All other fields are copied unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC_DIR = Path("data/aep_causal_classification_34")
DST_DIR = Path("data/aep_causal_classification_soft_exp")

BASE = 0.7 ** (1 / 13)  # ≈ 0.9726


def soft_label(record: dict) -> float:
    if record["label"] == 0:
        return 0.0
    gap = abs(record["tier_1"] - record["tier_2"])
    return BASE ** (gap - 1)


def convert_split(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            r = json.loads(line)
            r["label"] = round(soft_label(r), 6)
            fout.write(json.dumps(r) + "\n")
            n += 1
    print(f"  {src.name}: {n:,} records -> {dst}")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Soft labels only for training; val/test keep original binary labels
    convert_split(SRC_DIR / "directional_train.jsonl", DST_DIR / "directional_train.jsonl")

    for split in ("val", "test"):
        src = SRC_DIR / f"directional_{split}.jsonl"
        dst = DST_DIR / f"directional_{split}.jsonl"
        shutil.copy(src, dst)
        print(f"  copied {src.name} (binary labels unchanged) -> {dst}")

    manifest_src = SRC_DIR / "directional_manifest.json"
    if manifest_src.exists():
        shutil.copy(manifest_src, DST_DIR / "directional_manifest.json")
        print(f"  copied directional_manifest.json")

    # Quick sanity check
    print("\nSanity check (first 3 unique gaps from train):")
    seen = {}
    with open(DST_DIR / "directional_train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["label"] == 0.0, abs(r["tier_1"] - r["tier_2"]))
            if key not in seen:
                seen[key] = r["label"]
                neg, gap = key
                print(f"    {'neg' if neg else 'pos'} gap={gap} -> label={r['label']:.6f}")
            if len(seen) >= 20:
                break


if __name__ == "__main__":
    main()
