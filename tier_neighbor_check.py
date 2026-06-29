"""
Check whether semantically similar training texts share the same tier.

Dedups all text_1/text_2 from directional_train.jsonl into unique texts, assigns
each its (majority) tier, embeds with a generic sentence-transformer, then for
each text finds its top-K nearest neighbors and measures same-tier agreement.

Diagnoses tier overlap / label noise: if nearest neighbors frequently land in a
DIFFERENT tier, the tier boundaries are not semantically separable and the model
will struggle to predict them.
"""
import json
from collections import defaultdict, Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_train.jsonl"
MODEL = "all-MiniLM-L6-v2"
TOPK = 5
DEVICE = "cuda:1"  # idle GPU

# ── 1. Dedup texts -> majority tier ─────────────────────────────────────────
text2tiers = defaultdict(Counter)
with open(PATH) as f:
    for line in f:
        o = json.loads(line)
        text2tiers[o["text_1"]][o["tier_1"]] += 1
        text2tiers[o["text_2"]][o["tier_2"]] += 1

texts = list(text2tiers.keys())
tiers = np.array([text2tiers[t].most_common(1)[0][0] for t in texts])
print(f"unique texts: {len(texts)}")

# ── 2. Embed ─────────────────────────────────────────────────────────────────
model = SentenceTransformer(MODEL, device=DEVICE)
emb = model.encode(texts, batch_size=128, convert_to_tensor=True,
                   normalize_embeddings=True, show_progress_bar=False)

# ── 3. Cosine similarity (normalized -> dot product) ────────────────────────
sim = (emb @ emb.T).cpu().numpy()
np.fill_diagonal(sim, -1.0)  # exclude self

# ── 4. Top-K neighbor tier agreement ────────────────────────────────────────
nn_idx = np.argsort(-sim, axis=1)[:, :TOPK]

# top-1 same-tier rate
top1 = nn_idx[:, 0]
top1_same = (tiers[top1] == tiers).mean()

# fraction of top-K neighbors that share the tier (per text), then averaged
same_in_k = (tiers[nn_idx] == tiers[:, None]).mean(axis=1)
mean_topk_same = same_in_k.mean()

print(f"\n=== Semantic-neighbor tier agreement (K={TOPK}) ===")
print(f"top-1 nearest neighbor in SAME tier:  {top1_same*100:.1f}%")
print(f"avg fraction of top-{TOPK} in same tier: {mean_topk_same*100:.1f}%")
random_baseline = (Counter(tiers).most_common(1)[0][1] / len(tiers))
print(f"(random/majority baseline same-tier:  ~{100/len(set(tiers)):.1f}% uniform, "
      f"{random_baseline*100:.1f}% majority)")

# ── 5. Per-tier breakdown: how 'pure' is each tier's neighborhood ───────────
print(f"\n=== Per-tier top-1 same-tier rate ===")
print(f"{'tier':>5} {'#texts':>7} {'top1-same%':>11}  {'most-confused-with (count)'}")
for t in sorted(set(tiers)):
    mask = tiers == t
    nbr_tiers = tiers[top1[mask]]
    same = (nbr_tiers == t).mean()
    confusion = Counter(nbr_tiers[nbr_tiers != t])
    conf_str = ", ".join(f"t{c}({n})" for c, n in confusion.most_common(3))
    print(f"{t:>5} {mask.sum():>7} {same*100:>10.1f}%  {conf_str}")

# ── 6. Show worst offenders: very similar texts in DIFFERENT tiers ──────────
print(f"\n=== Most similar cross-tier pairs (high similarity, different tier) ===")
cross = sim.copy()
same_tier_mask = tiers[:, None] == tiers[None, :]
cross[same_tier_mask] = -1.0
flat = np.argsort(-cross, axis=None)[:8]
seen = set()
shown = 0
for fi in flat:
    i, j = divmod(fi, len(texts))
    key = tuple(sorted((i, j)))
    if key in seen:
        continue
    seen.add(key)
    print(f"\nsim={sim[i,j]:.3f}  tier {tiers[i]} vs tier {tiers[j]}")
    print(f"  [t{tiers[i]}] {texts[i][:140]}")
    print(f"  [t{tiers[j]}] {texts[j][:140]}")
    shown += 1
    if shown >= 5:
        break
