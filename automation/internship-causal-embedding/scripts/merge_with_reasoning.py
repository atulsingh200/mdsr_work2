#!/usr/bin/env python3
"""Build two merged datasets by concatenating all three source reasoning files.

Instead of joining against v2, we directly concatenate the three reasoning
datasets (procedural + aep_notier11 + ajo) into unified train/val/test splits.

Output datasets:
  aep_ajo_procedural_res/        — original reasoning (labels flipped in-place)
  aep_ajo_procedural_fixed_reas/ — fixed reasoning + fixed labels

Sources:
  procedural -> directional_procedural_with_reasoning/
  aep        -> aep_gapdecay_notier11_with_reasoning/
  ajo        -> ajo_gapdecay_15tier_with_reasoning/
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/data")

SOURCES = {
    "procedural": BASE / "new_aep_workflow_scrap/directional_procedural_with_reasoning",
    "aep":        BASE / "aep_gapdecay_notier11_with_reasoning",
    "ajo":        BASE / "ajo_gapdecay_15tier_with_reasoning",
}

OUT_RES   = BASE / "aep_ajo_procedural_res"
OUT_FIXED = BASE / "aep_ajo_procedural_fixed_reas"
SPLITS    = ["train", "val", "test"]


def src_path(source_dir: Path, split: str, fixed: bool) -> Path:
    suffix = "_fixed" if fixed else ""
    p = source_dir / f"directional_{split}_with_reasoning{suffix}.jsonl"
    # procedural has no _fixed copy (was fixed in-place) — fall back to original
    if not p.exists() and fixed:
        p = source_dir / f"directional_{split}_with_reasoning.jsonl"
    return p


def build_split(split: str, fixed: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    total = 0
    with out_path.open("w") as out_f:
        for source, source_dir in SOURCES.items():
            p = src_path(source_dir, split, fixed)
            if not p.exists():
                print(f"  [WARN] not found: {p}", flush=True)
                continue
            n = 0
            with p.open() as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    r["source"] = source
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
            print(f"  {source:12} {split}: {n} rows", flush=True)
            total += n

    print(f"  → {out_path.name}: {total} total rows", flush=True)


def main() -> None:
    for fixed, out_dir in [(False, OUT_RES), (True, OUT_FIXED)]:
        label = "FIXED" if fixed else "ORIGINAL"
        print(f"\n{'='*60}")
        print(f"Building {out_dir.name}  [{label} reasoning]")
        for split in SPLITS:
            print(f"\n[{split}]", flush=True)
            build_split(split, fixed=fixed, out_dir=out_dir)

    print("\n=== SUMMARY ===")
    for out_dir in [OUT_RES, OUT_FIXED]:
        print(f"\n{out_dir.name}/")
        for split in SPLITS:
            p = out_dir / f"{split}.jsonl"
            if p.exists():
                n = sum(1 for _ in p.open())
                print(f"  {split}.jsonl : {n} rows")


if __name__ == "__main__":
    main()

