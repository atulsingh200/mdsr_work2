import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_test.jsonl"
OUTPUT_PATH = "/mnt/localssd/automation/internship-causal-embedding/top10_cross_tier_pairs.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

print("Loading dataset...")
records = []
with open(DATASET_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Total records: {len(records)}")

# Build a mapping from text -> (tier, sub, url) using the first occurrence
text_meta = {}
for r in records:
    if r["text_1"] not in text_meta:
        text_meta[r["text_1"]] = {"tier": r["tier_1"], "sub": r["sub_1"], "url": r["url_1"]}
    if r["text_2"] not in text_meta:
        text_meta[r["text_2"]] = {"tier": r["tier_2"], "sub": r["sub_2"], "url": r["url_2"]}

unique_texts = list(text_meta.keys())
print(f"Unique texts to encode: {len(unique_texts)}")

print(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding all unique texts...")
embeddings = model.encode(
    unique_texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)
# embeddings shape: (N, D), already L2-normalized

# Full N x N cosine similarity matrix via dot product
print("Computing full similarity matrix (all combinations)...")
sim_matrix = np.dot(embeddings, embeddings.T)  # (N, N)

tiers = np.array([text_meta[t]["tier"] for t in unique_texts])  # (N,)
N = len(unique_texts)

# Build mask: True where tier_i != tier_j and i != j
tier_i = tiers[:, None].repeat(N, axis=1)  # (N, N)
tier_j = tiers[None, :].repeat(N, axis=0)  # (N, N)
cross_tier_mask = (tier_i != tier_j)        # excludes same-tier AND same-text (since same text => same tier)

# Zero out non-cross-tier entries so they don't show up in top-k
sim_matrix[~cross_tier_mask] = -np.inf

# Flatten and find top 10
flat_sim = sim_matrix.flatten()
top10_flat = np.argsort(flat_sim)[::-1][:10]

print(f"\nTop 10 cross-tier pairs (all {N}x{N} combinations searched):")
results = []
for rank, flat_idx in enumerate(top10_flat, start=1):
    i, j = divmod(int(flat_idx), N)
    meta_i = text_meta[unique_texts[i]]
    meta_j = text_meta[unique_texts[j]]
    result = {
        "rank": rank,
        "similarity": round(float(sim_matrix[i, j]), 4),
        "tier_1": meta_i["tier"],
        "tier_2": meta_j["tier"],
        "sub_1": meta_i["sub"],
        "sub_2": meta_j["sub"],
        "url_1": meta_i["url"],
        "url_2": meta_j["url"],
        "text_1": unique_texts[i],
        "text_2": unique_texts[j],
    }
    results.append(result)
    print(f"  Rank {rank}: similarity={result['similarity']:.4f}  tier_1={result['tier_1']}  tier_2={result['tier_2']}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {OUTPUT_PATH}")
