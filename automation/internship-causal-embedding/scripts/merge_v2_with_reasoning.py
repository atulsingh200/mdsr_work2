#!/usr/bin/env python3
"""Build reasoning-augmented versions of aep_ajo_procedural_workflow_v2.

Joins reasoning onto v2 rows by (text_1, text_2) per source, preserving
the original v2 row order and all original fields.

Outputs:
  aep_ajo_procedural_workflow_v2_res/        — original reasoning
  aep_ajo_procedural_workflow_v2_fixed_reas/ — fixed reasoning

Source mapping (by 'source' field in v2):
  procedural -> new_aep_workflow_scrap/directional_procedural_with_reasoning
  aep        -> aep_gapdecay_with_reasoning
  ajo        -> ajo_gapdecay_15tier_with_reasoning
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/data")

REASONING_DIRS = {
    "procedural": BASE / "new_aep_workflow_scrap/directional_procedural_with_reasoning",
    "aep":        BASE / "aep_gapdecay_with_reasoning",
    "ajo":        BASE / "ajo_gapdecay_15tier_with_reasoning",
}

V2_DIR    = BASE / "aep_ajo_procedural_workflow_v2"
OUT_RES   = BASE / "aep_ajo_procedural_workflow_v2_res"
OUT_FIXED = BASE / "aep_ajo_procedural_workflow_v2_fixed_reas"
SPLITS    = ["train", "val", "test"]


def reas_file(source_dir: Path, split: str, fixed: bool) -> Path:
    # Both datasets now use the same clean source files
    # (_with_reasoning.jsonl has been fixed in-place for all sources)
    return source_dir / f"directional_{split}_with_reasoning.jsonl"


def build_lookup(source: str, split: str, fixed: bool) -> dict[tuple, dict]:
    """(text_1, text_2) -> row, first occurrence wins."""
    p = reas_file(REASONING_DIRS[source], split, fixed)
    if not p.exists():
        print(f"  [WARN] not found: {p}", flush=True)
        return {}
    lookup: dict[tuple, dict] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["text_1"], r["text_2"])
            if key not in lookup:
                lookup[key] = r
    return lookup


def build_split(split: str, fixed: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    lookups = {src: build_lookup(src, split, fixed) for src in REASONING_DIRS}
    print(f"  lookups: { {s: len(lookups[s]) for s in lookups} }", flush=True)

    matched = missing = 0
    with out_path.open("w") as out_f:
        with (V2_DIR / f"{split}.jsonl").open() as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["text_1"], row["text_2"])
                source = row.get("source", "")
                reas_row = lookups.get(source, {}).get(key)
                if reas_row:
                    row["reasoning"] = reas_row.get("reasoning")
                    row["label"]     = reas_row["label"]   # use aligned label
                    matched += 1
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    missing += 1  # removed stubborn row — skip entirely

    print(f"  {split}: {matched} matched, {missing} missing → {out_path.name}", flush=True)


def main() -> None:
    for fixed, out_dir in [(False, OUT_RES), (True, OUT_FIXED)]:
        label = "FIXED" if fixed else "ORIGINAL"
        print(f"\n{'='*60}", flush=True)
        print(f"Building {out_dir.name}  [{label} reasoning]", flush=True)
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
