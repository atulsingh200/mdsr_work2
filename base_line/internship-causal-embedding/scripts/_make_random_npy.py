"""Generate a random negative index file with the same shape as a hard-neg .npy.

Usage:
    python scripts/_make_random_npy.py <pairs_jsonl> <out_npy> <k> <seed>
"""
import json
import sys
import numpy as np
from pathlib import Path

pairs_file = Path(sys.argv[1])
hn_out     = Path(sys.argv[2])
k          = int(sys.argv[3])
seed       = int(sys.argv[4])

pairs = []
for line in pairs_file.read_text(errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
        if r.get("anchor", "").strip() and r.get("positive", "").strip():
            pairs.append(r)
    except Exception:
        pass

N = len(pairs)
rng = np.random.default_rng(seed)
# Vectorized: sample k+1 from [0,N), then remove self by shifting indices >= i
rand_idx = np.empty((N, k), dtype=np.int32)
raw = rng.integers(0, N - 1, size=(N, k), dtype=np.int32)
rows = np.arange(N, dtype=np.int32)[:, None]
rand_idx = np.where(raw < rows, raw, raw + 1).astype(np.int32)

hn_out.parent.mkdir(parents=True, exist_ok=True)
np.save(hn_out, rand_idx)
print(f"  saved {hn_out}  shape={rand_idx.shape}", flush=True)
