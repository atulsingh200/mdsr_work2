"""
Generate a markdown report: for each workflow text (t1,t2,t3), list the top-5
most similar training texts with their FULL text and tier, plus the predicted
tier via nearest-neighbor voting.
"""
import json
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer

PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_train.jsonl"
MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda:1"
TOPK = 5
OUT = "/mnt/localssd/workflow_tier_report.md"

queries = {
    "t1": "Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.",
    "t2": "Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.",
    "t3": "Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.",
}

# corpus
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

lines = []
lines.append("# Workflow Text → Tier Report (Nearest-Neighbor Mapping)\n")
lines.append(f"**Corpus:** `directional_train.jsonl` ({len(texts)} unique texts, 15 tiers)  ")
lines.append(f"**Embedder:** `{MODEL}` (cosine similarity)  ")
lines.append(f"**Method:** for each query, the {TOPK} most similar training texts; predicted tier = majority vote.\n")
lines.append("---\n")

for qi, k in enumerate(q_keys):
    order = np.argsort(-sim[qi])[:TOPK]
    vote = Counter(int(tiers[i]) for i in order)
    pred = vote.most_common(1)[0]

    lines.append(f"## Query `{k}`\n")
    lines.append(f"> {queries[k]}\n")
    lines.append(f"**Predicted tier: {pred[0]}** (won {pred[1]}/{TOPK} of top neighbors)  ")
    lines.append(f"**Tier vote spread:** {dict(vote.most_common())}\n")
    lines.append(f"### Top {TOPK} most similar training texts\n")
    for r, idx in enumerate(order, 1):
        lines.append(f"**{r}. Tier {tiers[idx]}**  (sub `{subs[idx]}`, similarity = `{sim[qi,idx]:.3f}`)")
        lines.append("")
        lines.append(f"> {texts[idx]}")
        lines.append("")
    lines.append("---\n")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"Report written to {OUT}")
print(f"({len(lines)} lines)")
