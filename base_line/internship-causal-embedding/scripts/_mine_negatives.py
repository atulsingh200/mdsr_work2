"""Mine semantic hard negatives using all-MiniLM-L6-v2.

Usage:
    python scripts/_mine_negatives.py <pairs_jsonl> <out_npy> <k>
"""
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "src")
from evaluation.retrievers import PretrainedRetriever

pairs_file = Path(sys.argv[1])
hn_out     = Path(sys.argv[2])
k          = int(sys.argv[3])

MINER = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK = 1024

pairs = []
for line in pairs_file.read_text(errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
        a, p = r.get("anchor", "").strip(), r.get("positive", "").strip()
        if a and p:
            pairs.append((a, p))
    except Exception:
        pass

N = len(pairs)
print(f"  {N} pairs — {pairs_file.name}", flush=True)

miner = PretrainedRetriever(
    model_name=MINER, pooling="mean", max_seq_length=256, batch_size=256
)
positives = [p for _, p in pairs]

t0 = time.time()
pos_emb = miner.encode_candidates(positives)
print(f"  encoded in {time.time()-t0:.1f}s  dim={pos_emb.shape[1]}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pos_t = torch.from_numpy(pos_emb).to(device)
hard_idx = np.empty((N, k), dtype=np.int32)

t1 = time.time()
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    sims = pos_t[start:end] @ pos_t.T
    local_rows = torch.arange(end - start, device=device)
    global_rows = torch.arange(start, end, device=device)
    sims[local_rows, global_rows] = float("-inf")
    _, top_idx = sims.topk(k, dim=1, largest=True, sorted=True)
    hard_idx[start:end] = top_idx.cpu().numpy()
print(f"  kNN in {time.time()-t1:.1f}s", flush=True)

hn_out.parent.mkdir(parents=True, exist_ok=True)
np.save(hn_out, hard_idx)
print(f"  saved {hn_out}  shape={hard_idx.shape}", flush=True)
