"""Sample up to <cap> rows from a JSONL file, deterministically.

Uses reservoir sampling so only <cap> rows are ever held in RAM —
safe even for multi-GB source files like workflow (1.6 GB / 2.1M rows).

Usage:
    python scripts/_sample_jsonl.py <src> <dst> <cap> <seed>
"""
import json
import sys
import numpy as np
from pathlib import Path

src  = Path(sys.argv[1])
dst  = Path(sys.argv[2])
cap  = int(sys.argv[3])
seed = int(sys.argv[4])

dst.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed)

# Reservoir sampling: keep at most `cap` valid lines in memory at all times.
reservoir: list[str] = []
total_valid = 0

with src.open(errors="replace") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if not (r.get("anchor", "").strip() and r.get("positive", "").strip()):
                continue
        except Exception:
            continue

        total_valid += 1
        if len(reservoir) < cap:
            reservoir.append(line)
        else:
            # Replace a random element with decreasing probability
            j = int(rng.integers(0, total_valid))
            if j < cap:
                reservoir[j] = line

# Shuffle reservoir for deterministic but unordered output
rng.shuffle(reservoir)

with dst.open("w") as f:
    for line in reservoir:
        f.write(line + "\n")

if total_valid <= cap:
    print(f"  kept all {total_valid} rows (< cap {cap})", flush=True)
else:
    print(f"  sampled {cap} / {total_valid} rows (reservoir)", flush=True)
