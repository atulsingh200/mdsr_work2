"""
Build a hard-negative test set from the existing directional test set.

For every positive pair  x -> y  (label=1, tier(x) < tier(y)):
  - Find all chunks in the test pool whose tier < tier(x)  →  candidate y*
  - Embed y and every y* with a sentence encoder
  - Pick the y* most similar to y  (highest cosine sim)
  - Emit: (x, y,  label=1)   — original positive
           (x, y*, label=0)  — hard negative  [y* looks like y but is in the wrong tier]

Pairs where no candidate y* exists (tier(x)==1, only 107/2935) are dropped.

Output: data/aep_causal_classification_hard_neg_test/test.jsonl

Run (from repo root, with the venv active; --output is relative to cwd):
  cd /mnt/localssd/automation/internship-causal-embedding
  source .venv/bin/activate
  python3 src/classifier/build_hard_neg_test.py

  # or as a one-liner:
  cd /mnt/localssd/automation/internship-causal-embedding && \
    source .venv/bin/activate && \
    python3 src/classifier/build_hard_neg_test.py

Optional overrides:
  python3 src/classifier/build_hard_neg_test.py \
    --input  data/aep_dataset/directional_test.jsonl \
    --output data/aep_dataset_hard_neg_test/directional_test.jsonl \
    --model  all-MiniLM-L6-v2 --batch 256 --seed 42

  # keep the original label==0 rows too (in addition to mined hard negs):
  python3 src/classifier/build_hard_neg_test.py --keep-orig-neg
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="/mnt/localssd/automation/internship-causal-embedding/data/aep_dataset/directional_test.jsonl")
parser.add_argument("--output", default="data/aep_dataset_hard_neg_test/directional_test.jsonl")
parser.add_argument("--model",  default="all-MiniLM-L6-v2", help="SentenceTransformer model for similarity")
parser.add_argument("--batch",  type=int, default=256)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--keep-orig-neg", action="store_true",
                    help="Also keep the original label==0 rows from --input in the "
                         "output (default: drop them, emit only positives + mined "
                         "hard negatives).")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# Load test set
# ---------------------------------------------------------------------------
with open(args.input) as f:
    rows = [json.loads(l) for l in f]

positives = [r for r in rows if r["label"] == 1]
print(f"Total rows: {len(rows)}  |  positives: {len(positives)}")

# ---------------------------------------------------------------------------
# Build a deduplicated pool of all chunks, indexed by tier
# ---------------------------------------------------------------------------
# Each entry: {"text", "tier", "sub", "url"}
chunk_pool_by_tier: dict[int, list[dict]] = defaultdict(list)
seen_texts: set[str] = set()

for r in rows:
    for side in [("text_1", "tier_1", "sub_1", "url_1"),
                 ("text_2", "tier_2", "sub_2", "url_2")]:
        text_key, tier_key, sub_key, url_key = side
        text = r[text_key]
        if text not in seen_texts:
            seen_texts.add(text)
            chunk_pool_by_tier[r[tier_key]].append({
                "text": text,
                "tier": r[tier_key],
                "sub":  r[sub_key],
                "url":  r[url_key],
            })

print("Unique chunks per tier:")
for t in sorted(chunk_pool_by_tier):
    print(f"  T{t}: {len(chunk_pool_by_tier[t])}")

# ---------------------------------------------------------------------------
# Collect all unique texts that need to be embedded
# ---------------------------------------------------------------------------
# We need embeddings for:
#   - every y  (text_2 of each positive)
#   - every candidate y*  (all chunks with tier < tier(x) for some positive)

all_texts_to_embed: list[str] = list(seen_texts)
text_to_idx = {t: i for i, t in enumerate(all_texts_to_embed)}

print(f"\nEmbedding {len(all_texts_to_embed)} unique texts with '{args.model}' ...")
model = SentenceTransformer(args.model)
embeddings = model.encode(
    all_texts_to_embed,
    batch_size=args.batch,
    show_progress_bar=True,
    normalize_embeddings=True,   # unit vectors → cosine sim = dot product
    convert_to_numpy=True,
)
print("Embeddings done:", embeddings.shape)

# ---------------------------------------------------------------------------
# For each positive pair, find the best y*
# ---------------------------------------------------------------------------
output_rows = []
skipped = 0

for r in positives:
    tier_x = r["tier_1"]
    text_x = r["text_1"]
    text_y = r["text_2"]

    # All chunks with tier strictly less than tier_x, excluding x and y themselves
    candidates = []
    for t in range(1, tier_x):
        for c in chunk_pool_by_tier[t]:
            if c["text"] != text_x and c["text"] != text_y:
                candidates.append(c)

    if not candidates:
        skipped += 1
        continue

    # Cosine similarity between y and each candidate y*
    emb_y    = embeddings[text_to_idx[text_y]].reshape(1, -1)   # (1, d)
    cand_idx = [text_to_idx[c["text"]] for c in candidates]
    emb_cands = embeddings[cand_idx]                             # (n_cands, d)

    sims = (emb_y @ emb_cands.T).squeeze(0)                     # dot product = cosine (normalised)
    best = int(np.argmax(sims))
    best_chunk = candidates[best]
    best_sim   = float(sims[best])

    # Positive: x -> y
    output_rows.append({
        "text_1":   r["text_1"],
        "text_2":   r["text_2"],
        "label":    1,
        "tier_1":   r["tier_1"],
        "tier_2":   r["tier_2"],
        "sub_1":    r["sub_1"],
        "sub_2":    r["sub_2"],
        "url_1":    r["url_1"],
        "url_2":    r["url_2"],
        "hard_neg": False,
    })

    # Hard negative: x -> y*   (label=0 because tier(y*) < tier(x), wrong direction)
    output_rows.append({
        "text_1":        r["text_1"],
        "text_2":        best_chunk["text"],
        "label":         0,
        "tier_1":        r["tier_1"],
        "tier_2":        best_chunk["tier"],
        "sub_1":         r["sub_1"],
        "sub_2":         best_chunk["sub"],
        "url_1":         r["url_1"],
        "url_2":         best_chunk["url"],
        "hard_neg":      True,
        "sim_y_ystar":   round(best_sim, 6),
    })

# ---------------------------------------------------------------------------
# Optionally keep the ORIGINAL label==0 rows from the input (not dropped).
# Tagged orig_neg=True (and hard_neg=False) so the three row types remain
# distinguishable: positives, mined hard negatives, original negatives.
# ---------------------------------------------------------------------------
kept_orig_neg = 0
if args.keep_orig_neg:
    for r in rows:
        if r["label"] == 0:
            row = dict(r)            # preserve all original fields as-is
            row["hard_neg"] = False
            row["orig_neg"] = True
            output_rows.append(row)
            kept_orig_neg += 1
    print(f"Kept original label==0 rows: {kept_orig_neg}")

print(f"\nSkipped (no y* candidate): {skipped}")
n_pos = sum(1 for r in output_rows if r["label"] == 1)
n_hardneg = sum(1 for r in output_rows if r.get("hard_neg"))
print(f"Output rows: {len(output_rows)}  "
      f"({n_pos} positives + {n_hardneg} hard negatives"
      + (f" + {kept_orig_neg} original negatives" if args.keep_orig_neg else "")
      + ")")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    for row in output_rows:
        f.write(json.dumps(row) + "\n")

print(f"Saved → {args.output}")

# ---------------------------------------------------------------------------
# Quick stats
# ---------------------------------------------------------------------------
from collections import Counter
label_counts = Counter(r["label"] for r in output_rows)
hard_neg_counts = Counter(r.get("hard_neg", False) for r in output_rows)
print(f"Label dist: {dict(label_counts)}")
print(f"Hard neg dist: {dict(hard_neg_counts)}")

orig_neg_counts = Counter(r.get("orig_neg", False) for r in output_rows)
print(f"Orig neg dist: {dict(orig_neg_counts)}")

sims_all = [r["sim_y_ystar"] for r in output_rows if r.get("hard_neg")]
if sims_all:
    print(f"Sim(y, y*) stats: min={min(sims_all):.4f}  "
          f"mean={np.mean(sims_all):.4f}  max={max(sims_all):.4f}")
