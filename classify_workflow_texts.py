"""
Classify the 3 workflow texts (t1, t2, t3) by nearest-neighbor tier voting
against the training set. For each query text, find its most similar training
texts and report which tier they belong to -> the predicted category.
"""
import json
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer

PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_train.jsonl"
MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda:1"
TOPK = 10

queries = {
    "t1": "Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.",
    "t2": "Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.",
    "t3": "Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.",
}

# ── Build training corpus: text -> majority tier (+ keep sub for context) ───
text2tier = defaultdict(Counter)
text2sub = defaultdict(Counter)
with open(PATH) as f:
    for line in f:
        o = json.loads(line)
        text2tier[o["text_1"]][o["tier_1"]] += 1
        text2tier[o["text_2"]][o["tier_2"]] += 1
        text2sub[o["text_1"]][o["sub_1"]] += 1
        text2sub[o["text_2"]][o["sub_2"]] += 1

texts = list(text2tier.keys())
tiers = np.array([text2tier[t].most_common(1)[0][0] for t in texts])
subs = [text2sub[t].most_common(1)[0][0] for t in texts]

model = SentenceTransformer(MODEL, device=DEVICE)
corpus_emb = model.encode(texts, batch_size=128, convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=False)

q_keys = list(queries.keys())
q_emb = model.encode([queries[k] for k in q_keys], convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False)

sim = (q_emb @ corpus_emb.T).cpu().numpy()

for qi, k in enumerate(q_keys):
    order = np.argsort(-sim[qi])
    topk = order[:TOPK]

    # tier vote (count) and similarity-weighted vote
    vote = Counter()
    wvote = defaultdict(float)
    for idx in topk:
        vote[int(tiers[idx])] += 1
        wvote[int(tiers[idx])] += float(sim[qi, idx])

    pred_count = vote.most_common(1)[0]
    pred_weighted = max(wvote.items(), key=lambda x: x[1])

    print("=" * 90)
    print(f"QUERY {k}: {queries[k][:110]}...")
    print(f"\n  >>> PREDICTED TIER (top-{TOPK} majority vote): tier {pred_count[0]}  "
          f"({pred_count[1]}/{TOPK} neighbors)")
    print(f"  >>> PREDICTED TIER (similarity-weighted):    tier {pred_weighted[0]}  "
          f"(weight {pred_weighted[1]:.2f})")
    print(f"  tier vote spread: {dict(vote.most_common())}")
    print(f"\n  Top {TOPK} nearest training texts:")
    print(f"  {'rank':>4} {'sim':>6} {'tier':>5} {'sub':>5}  text")
    for r, idx in enumerate(topk, 1):
        print(f"  {r:>4} {sim[qi,idx]:>6.3f} {tiers[idx]:>5} {subs[idx]:>5}  {texts[idx][:90]}")
    print()
