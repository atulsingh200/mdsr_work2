"""
BM25 version: for each workflow text (t1,t2,t3), find the top-5 most similar
training texts via BM25 lexical search, with FULL text + tier, plus predicted
tier by majority vote. Writes a markdown report.
"""
import json
import re
from collections import defaultdict, Counter

import numpy as np
from rank_bm25 import BM25Okapi

PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_train.jsonl"
TOPK = 5
OUT = "/mnt/localssd/workflow_tier_report_bm25.md"

queries = {
    "t1": "Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.",
    "t2": "Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.",
    "t3": "Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.",
}

# simple lowercase word tokenizer
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

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
tiers = [text2tier[t].most_common(1)[0][0] for t in texts]
subs = [text2sub[t].most_common(1)[0][0] for t in texts]

tokenized = [tok(t) for t in texts]
bm25 = BM25Okapi(tokenized)

lines = []
lines.append("# Workflow Text → Tier Report (BM25 Lexical Search)\n")
lines.append(f"**Corpus:** `directional_train.jsonl` ({len(texts)} unique texts, 15 tiers)  ")
lines.append("**Method:** BM25Okapi keyword scoring over lowercased word tokens.  ")
lines.append(f"For each query, the {TOPK} highest-BM25 training texts; predicted tier = majority vote.\n")
lines.append("---\n")

for k, q in queries.items():
    scores = bm25.get_scores(tok(q))
    order = np.argsort(-scores)[:TOPK]
    vote = Counter(int(tiers[i]) for i in order)
    pred = vote.most_common(1)[0]

    lines.append(f"## Query `{k}`\n")
    lines.append(f"> {q}\n")
    lines.append(f"**Predicted tier: {pred[0]}** (won {pred[1]}/{TOPK} of top neighbors)  ")
    lines.append(f"**Tier vote spread:** {dict(vote.most_common())}\n")
    lines.append(f"### Top {TOPK} BM25 matches\n")
    for r, idx in enumerate(order, 1):
        lines.append(f"**{r}. Tier {tiers[idx]}**  (sub `{subs[idx]}`, BM25 score = `{scores[idx]:.2f}`)")
        lines.append("")
        lines.append(f"> {texts[idx]}")
        lines.append("")
    lines.append("---\n")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"Report written to {OUT}")

# also print a compact comparison summary to stdout
print("\n=== BM25 predicted tiers ===")
for k, q in queries.items():
    scores = bm25.get_scores(tok(q))
    order = np.argsort(-scores)[:TOPK]
    vote = Counter(int(tiers[i]) for i in order)
    print(f"  {k}: tier {vote.most_common(1)[0][0]}   spread {dict(vote.most_common())}")
