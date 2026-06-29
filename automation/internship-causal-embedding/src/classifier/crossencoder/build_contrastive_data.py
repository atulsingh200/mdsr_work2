"""Build contrastive training data for the cross-encoder.

Pipeline (per split):
  1. Load directional_{split}.jsonl from --source-dir
  2. Flip label=0 rows to label=1 (swap text_1/text_2, tier, sub, url fields)
  3. Deduplicate on (text_1, text_2)
  4. For each all-positive row (x, y):
       - up to 2 hard negatives : top-2 cosine-similar y* to y, tier(y*) < tier(x)
       - up to 2 easy negatives : random chunks, tier(chunk) < tier(x)
       - 1 reverse pair         : (y, x, label=0)
     The tier constraint is STRICT: hard/easy negatives are only ever drawn from
     tiers below tier(x). When the lower-tier pool is too small we emit fewer
     negatives rather than relax the constraint. Tier-1 anchors have no lower
     tier, so they get only the reverse pair.
  5. Each anchor x forms one group of 2-6 rows (1 positive + 1-5 negatives).
     Group sizes are VARIABLE; the InfoNCE loss handles this via per-group split.

Output JSONL schema (one row per pair):
  group_id, text_1, text_2, label, tier_1, tier_2,
  sub_1, sub_2, url_1, url_2, neg_type

Usage (from repo root):
  python3 -m src.classifier.crossencoder.build_contrastive_data \\
      --source-dir data/aep_causal_classification_34 \\
      --out-dir    data/aep_causal_allpos_34 \\
      --split      train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--source-dir", default="data/aep_causal_classification_34")
    p.add_argument("--out-dir", default="data/aep_causal_allpos_34")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                   help="SentenceTransformer model for hard neg mining")
    p.add_argument("--topk-hard", type=int, default=2,
                   help="Hard negatives mined per positive")
    p.add_argument("--n-easy", type=int, default=2,
                   help="Easy negatives sampled per positive")
    p.add_argument("--batch", type=int, default=256,
                   help="Encoding batch size")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_id(anchor_text: str, idx: int) -> str:
    h = hashlib.sha256(anchor_text.encode()).hexdigest()[:16]
    return f"{h}_{idx}"


def build_chunk_pool(rows: list[dict]) -> dict[int, list[dict]]:
    """Return {tier -> [chunk_meta]} deduplicated across all texts in rows."""
    pool: dict[int, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for r in rows:
        for text_k, tier_k, sub_k, url_k in [
            ("text_1", "tier_1", "sub_1", "url_1"),
            ("text_2", "tier_2", "sub_2", "url_2"),
        ]:
            text = r[text_k]
            if text not in seen:
                seen.add(text)
                pool[r[tier_k]].append({
                    "text": text,
                    "tier": r[tier_k],
                    "sub": r[sub_k],
                    "url": r[url_k],
                })
    return pool


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode and L2-normalise all texts. Returns (N, d) float32 array."""
    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def mine_hard_negatives(
    pos_row: dict,
    chunk_pool_by_tier: dict[int, list[dict]],
    all_texts: list[str],
    embeddings: np.ndarray,
    text_to_idx: dict[str, int],
    exclude_texts: set[str],
    topk: int = 2,
) -> list[dict]:
    """Return up to topk hard neg rows for one positive.

    Candidates: tier(candidate) < tier(x), text not in exclude_texts.
    Ranked by cosine similarity to y (already normalised → dot product).
    """
    tier_x = pos_row["tier_1"]
    text_y = pos_row["text_2"]

    candidates = [
        c for t in range(1, tier_x)
        for c in chunk_pool_by_tier.get(t, [])
        if c["text"] not in exclude_texts
    ]
    if not candidates:
        return []

    emb_y = embeddings[text_to_idx[text_y]].reshape(1, -1)
    cand_idx = [text_to_idx[c["text"]] for c in candidates]
    sims = (emb_y @ embeddings[cand_idx].T).squeeze(0)
    k = min(topk, len(candidates))
    top = np.argsort(-sims)[:k]

    result = []
    for j in top:
        c = candidates[int(j)]
        result.append({
            "text_1": pos_row["text_1"], "text_2": c["text"], "label": 0,
            "tier_1": tier_x, "tier_2": c["tier"],
            "sub_1": pos_row["sub_1"], "sub_2": c["sub"],
            "url_1": pos_row["url_1"], "url_2": c["url"],
            "neg_type": "hard",
            "sim_y_ystar": round(float(sims[int(j)]), 6),
        })
    return result


def sample_easy_negatives(
    pos_row: dict,
    chunk_pool_by_tier: dict[int, list[dict]],
    exclude_texts: set[str],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Return up to n easy neg rows sampled randomly from tier < tier(x).

    Text must not be in exclude_texts. May return fewer than n (or 0) when the
    lower-tier pool is exhausted — we never relax the tier constraint.
    """
    tier_x = pos_row["tier_1"]
    candidates = [
        c for t in range(1, tier_x)
        for c in chunk_pool_by_tier.get(t, [])
        if c["text"] not in exclude_texts
    ]

    if not candidates:
        return []

    k = min(n, len(candidates))
    chosen = rng.choice(len(candidates), size=k, replace=False)
    result = []
    for idx in chosen:
        c = candidates[int(idx)]
        result.append({
            "text_1": pos_row["text_1"], "text_2": c["text"], "label": 0,
            "tier_1": tier_x, "tier_2": c["tier"],
            "sub_1": pos_row["sub_1"], "sub_2": c["sub"],
            "url_1": pos_row["url_1"], "url_2": c["url"],
            "neg_type": "easy",
        })
    return result


# ---------------------------------------------------------------------------
# Main pipeline per split
# ---------------------------------------------------------------------------

def build_contrastive_split(
    source_path: str,
    out_path: str,
    model_name: str,
    topk_hard: int,
    n_easy: int,
    batch_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"Loading {source_path}")
    with open(source_path) as f:
        raw_rows = [json.loads(ln) for ln in f]
    print(f"  Loaded {len(raw_rows)} rows  "
          f"(label=1: {sum(r['label']==1 for r in raw_rows)}, "
          f"label=0: {sum(r['label']==0 for r in raw_rows)})")

    # ---- Phase A: flip label=0 rows, deduplicate ----
    allpos: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for r in raw_rows:
        if r["label"] == 1:
            row = dict(r)
        else:
            # Flip reversed pair to correct causal direction
            row = {
                "text_1": r["text_2"], "text_2": r["text_1"], "label": 1,
                "tier_1": r["tier_2"], "tier_2": r["tier_1"],
                "sub_1": r["sub_2"], "sub_2": r["sub_1"],
                "url_1": r["url_2"], "url_2": r["url_1"],
            }
        key = (row["text_1"], row["text_2"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            allpos.append(row)

    print(f"  All-positive rows after flip+dedup: {len(allpos)}")

    # ---- Build chunk pool and embeddings ----
    print(f"  Building chunk pool from {len(allpos)} all-positive rows ...")
    chunk_pool_by_tier = build_chunk_pool(allpos)
    for t in sorted(chunk_pool_by_tier):
        print(f"    T{t}: {len(chunk_pool_by_tier[t])} unique chunks")

    all_texts = list({
        c["text"]
        for tier_chunks in chunk_pool_by_tier.values()
        for c in tier_chunks
    })
    text_to_idx = {t: i for i, t in enumerate(all_texts)}

    print(f"  Embedding {len(all_texts)} unique texts with '{model_name}' ...")
    embeddings = embed_texts(all_texts, model_name, batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ---- Build anchor_to_ys lookup (to exclude true effects from negatives) ----
    anchor_to_ys: dict[str, set[str]] = defaultdict(set)
    for r in allpos:
        anchor_to_ys[r["text_1"]].add(r["text_2"])

    # ---- Phase B: mine negatives, build groups ----
    output_rows: list[dict] = []
    n_hard_total = n_easy_total = n_reverse = 0
    n_tier1_anchors = 0

    for idx, pos_row in enumerate(allpos):
        anchor_text = pos_row["text_1"]
        tier_x = pos_row["tier_1"]

        # Texts that must be excluded from any negative candidate
        exclude = {anchor_text, pos_row["text_2"]} | anchor_to_ys[anchor_text]

        group_id = _group_id(anchor_text, idx)
        group: list[dict] = []

        # Positive (always first in group)
        group.append({**pos_row, "group_id": group_id, "neg_type": "positive"})

        # Hard + easy negatives are STRICTLY drawn from tier < tier_x.
        # When the lower-tier pool is too small we simply emit fewer negatives;
        # we never fall back to same/higher-tier chunks. Tier-1 anchors have no
        # lower tier, so they get only the reverse pair as a negative.
        if tier_x > 1:
            hard = mine_hard_negatives(
                pos_row, chunk_pool_by_tier, all_texts, embeddings,
                text_to_idx, exclude, topk=topk_hard,
            )
            n_hard_total += len(hard)

            # Exclude hard neg texts from the easy pool too.
            hard_texts = {r["text_2"] for r in hard}
            easy = sample_easy_negatives(
                pos_row, chunk_pool_by_tier, exclude | hard_texts,
                n=n_easy, rng=rng,
            )
            n_easy_total += len(easy)

            for r in hard:
                group.append({**r, "group_id": group_id})
            for r in easy:
                group.append({**r, "group_id": group_id})
        else:
            n_tier1_anchors += 1

        # Reverse pair (always a valid negative)
        reverse = {
            "text_1": pos_row["text_2"], "text_2": pos_row["text_1"], "label": 0,
            "tier_1": pos_row["tier_2"], "tier_2": pos_row["tier_1"],
            "sub_1": pos_row["sub_2"], "sub_2": pos_row["sub_1"],
            "url_1": pos_row["url_2"], "url_2": pos_row["url_1"],
            "neg_type": "reverse", "group_id": group_id,
        }
        group.append(reverse)
        n_reverse += 1

        output_rows.extend(group)

    # ---- Stats ----
    n_groups = len(allpos)
    per_group_count: Counter = Counter()
    for r in output_rows:
        per_group_count[r["group_id"]] += 1
    size_dist = Counter(per_group_count.values())
    print(f"\n  Groups: {n_groups}")
    print(f"  Total output rows: {len(output_rows)}")
    print(f"  Group size distribution (size -> #groups): {dict(sorted(size_dist.items()))}")
    print(f"  Hard negs mined: {n_hard_total}")
    print(f"  Easy negs sampled: {n_easy_total}")
    print(f"  Reverse pairs: {n_reverse}")
    print(f"  Tier-1 anchors (only reverse neg): {n_tier1_anchors}")
    neg_type_dist = Counter(r["neg_type"] for r in output_rows)
    print(f"  neg_type dist: {dict(neg_type_dist)}")

    # ---- Write ----
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved → {out_path}")


def main():
    args = parse_args()

    import subprocess
    print("===== GPU / CPU memory =====")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv"],
        check=False,
    )
    subprocess.run(["free", "-g"], check=False)

    in_path = os.path.join(args.source_dir, f"directional_{args.split}.jsonl")
    out_path = os.path.join(args.out_dir, f"directional_{args.split}.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    build_contrastive_split(
        source_path=in_path,
        out_path=out_path,
        model_name=args.model,
        topk_hard=args.topk_hard,
        n_easy=args.n_easy,
        batch_size=args.batch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
