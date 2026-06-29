"""
Build a *training/val* dataset augmented with SEMANTIC hard negatives.

Unlike build_hard_neg_test.py (which builds a balanced test probe and DROPS the
label-0 rows), this miner is for training: it KEEPS every original row exactly as
it is (both label 0 and label 1), and ADDITIONALLY mines hard negatives only for
label-1 positives.

For every positive pair  x -> y  (label=1, tier(x) < tier(y)):
  - Find all chunks in the split pool whose tier < tier(x)  →  candidate y*
  - Embed y and every y* with a sentence encoder (all-MiniLM by default)
  - Pick the top-k y* most similar to y (highest cosine sim)
  - Emit:  (x, y*, label=0, hard_neg=True)   — y* looks like y but is wrong tier

Positives with no candidate y* (tier(x) == 1) are still kept as-is; we just can't
add a hard neg for them.

Output (per --split):
  data/aep_causal_classification_hard_neg_semantic/directional_<split>.jsonl

The test split is intentionally NOT mined; copy it verbatim from the source dir
(handled by run_all.sh / done separately).
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir",
                   default="data/aep_causal_classification_34")
    p.add_argument("--out-dir",
                   default="data/aep_causal_classification_hard_neg_semantic")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--model", default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model for similarity")
    p.add_argument("--topk", type=int, default=3,
                   help="Hard negatives mined per label-1 positive")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.default_rng(args.seed)

    in_path = os.path.join(args.source_dir, f"directional_{args.split}.jsonl")
    out_path = os.path.join(args.out_dir, f"directional_{args.split}.jsonl")

    with open(in_path) as f:
        rows = [json.loads(l) for l in f]
    positives = [r for r in rows if r["label"] == 1]
    print(f"[{args.split}] total rows: {len(rows)}  |  positives: {len(positives)}")

    # ---- Build a deduplicated pool of all chunks, indexed by tier ----
    chunk_pool_by_tier: dict[int, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()
    for r in rows:
        for text_key, tier_key, sub_key, url_key in [
            ("text_1", "tier_1", "sub_1", "url_1"),
            ("text_2", "tier_2", "sub_2", "url_2"),
        ]:
            text = r[text_key]
            if text not in seen_texts:
                seen_texts.add(text)
                chunk_pool_by_tier[r[tier_key]].append({
                    "text": text, "tier": r[tier_key],
                    "sub": r[sub_key], "url": r[url_key],
                })
    print("Unique chunks per tier:")
    for t in sorted(chunk_pool_by_tier):
        print(f"  T{t}: {len(chunk_pool_by_tier[t])}")

    # ---- Embed all unique texts once ----
    all_texts = list(seen_texts)
    text_to_idx = {t: i for i, t in enumerate(all_texts)}
    print(f"\nEmbedding {len(all_texts)} unique texts with '{args.model}' ...")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        all_texts, batch_size=args.batch, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    print("Embeddings done:", embeddings.shape)

    # ---- Pass through ALL original rows unchanged ----
    output_rows: list[dict] = []
    for r in rows:
        out = dict(r)
        out.setdefault("hard_neg", False)
        output_rows.append(out)

    # ---- Mine top-k hard negs for each label-1 positive ----
    n_mined = 0
    n_skipped = 0
    sims_all: list[float] = []
    for r in positives:
        tier_x = r["tier_1"]
        text_x, text_y = r["text_1"], r["text_2"]

        candidates = [
            c for t in range(1, tier_x) for c in chunk_pool_by_tier[t]
            if c["text"] != text_x and c["text"] != text_y
        ]
        if not candidates:
            n_skipped += 1
            continue

        emb_y = embeddings[text_to_idx[text_y]].reshape(1, -1)
        cand_idx = [text_to_idx[c["text"]] for c in candidates]
        sims = (emb_y @ embeddings[cand_idx].T).squeeze(0)
        k = min(args.topk, len(candidates))
        top = np.argsort(-sims)[:k]
        for j in top:
            c = candidates[int(j)]
            sims_all.append(float(sims[int(j)]))
            output_rows.append({
                "text_1": text_x, "text_2": c["text"], "label": 0,
                "tier_1": tier_x, "tier_2": c["tier"],
                "sub_1": r["sub_1"], "sub_2": c["sub"],
                "url_1": r["url_1"], "url_2": c["url"],
                "hard_neg": True,
                "sim_y_ystar": round(float(sims[int(j)]), 6),
            })
            n_mined += 1

    print(f"\nMined hard negs: {n_mined}  "
          f"(positives skipped, no candidate: {n_skipped})")
    print(f"Total output rows: {len(output_rows)} "
          f"(orig {len(rows)} + mined {n_mined})")

    # ---- Write ----
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved → {out_path}")

    # ---- Stats ----
    label_counts = Counter(r["label"] for r in output_rows)
    hn_counts = Counter(r.get("hard_neg", False) for r in output_rows)
    print(f"Label dist: {dict(label_counts)}")
    print(f"Hard neg dist: {dict(hn_counts)}")
    if sims_all:
        print(f"Sim(y, y*) stats: min={min(sims_all):.4f}  "
              f"mean={np.mean(sims_all):.4f}  max={max(sims_all):.4f}")


if __name__ == "__main__":
    main()
