"""
Final grounded report: for each workflow text (t1,t2,t3), determine which tier
it belongs to, using BOTH the tier definitions (from build_classification_data.py)
AND the nearest-neighbor evidence (semantic + BM25). Explains WHY via the
sub-chapter the neighbors come from.
"""
import json
import re
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_train.jsonl"
MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda:1"
TOPK = 5
OUT = "/mnt/localssd/workflow_tier_assignment_report.md"

# Tier names + the sub-chapters that compose them (from build_classification_data.py)
TIER_NAME = {
    1: "Product orientation", 2: "Admin floor", 3: "Data foundations",
    4: "Profiles, audiences & subscriptions", 5: "Content authoring foundations",
    6: "Channels + campaigns (first sends)", 7: "Journeys (core)",
    8: "Personalization & decisioning", 9: "Advanced journey patterns & experimentation",
    10: "Multi-journey orchestration", 11: "Admin configuration",
    12: "Governance & privacy", 13: "Observability & reporting",
    14: "AI agents & assistants", 15: "Use cases, labs, capstones",
}
# What a few key sub-chapters mean (for the WHY explanation)
SUB_MEANING = {
    "20a": "abandoned-cart / customer-onboarding use-case (Luma retail scenario)",
    "4c":  "orchestrated campaigns (build audience, enrich, personalize, send)",
    "9f":  "in-app channel", "9d": "email channel",
    "11b": "personalization editor: helper functions, contextual events, dynamic content",
    "13a": "decisioning (offers / decision management)",
    "13b": "decision management",
    "5c":  "advanced journey use-cases (transactional, audience qualification)",
    "12b": "content experiments for emails",
    "2a":  "Journey Optimizer overview / personas",
}

queries = {
    "t1": "Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.",
    "t2": "Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.",
    "t3": "Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.",
}

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

# corpus
text2tier, text2sub = defaultdict(Counter), defaultdict(Counter)
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

# semantic
model = SentenceTransformer(MODEL, device=DEVICE)
ce = model.encode(texts, batch_size=128, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
qkeys = list(queries)
qe = model.encode([queries[k] for k in qkeys], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
sem = (qe @ ce.T).cpu().numpy()
# bm25
bm25 = BM25Okapi([tok(t) for t in texts])

def topk(scores):
    return np.argsort(-scores)[:TOPK]

L = []
L.append("# Which Tier Does Each Workflow Text Belong To?\n")
L.append("Tier definitions are from `build_classification_data.py` (the 15-tier curriculum used to build the dataset). "
         "Each text is assigned by combining (a) what the tier *means* and (b) the tier of its most-similar training texts, "
         "under both **semantic** (MiniLM) and **lexical** (BM25) retrieval.\n")
L.append("---\n")

for qi, k in enumerate(qkeys):
    sem_idx = topk(sem[qi])
    bm_scores = bm25.get_scores(tok(queries[k]))
    bm_idx = topk(bm_scores)
    sem_vote = Counter(tiers[i] for i in sem_idx)
    bm_vote = Counter(tiers[i] for i in bm_idx)
    sem_pred = sem_vote.most_common(1)[0]
    bm_pred = bm_vote.most_common(1)[0]

    L.append(f"## `{k}` — {queries[k]}\n")
    L.append(f"| Method | Predicted tier | Tier name | Vote |")
    L.append(f"|---|---|---|---|")
    L.append(f"| Semantic | **{sem_pred[0]}** | {TIER_NAME[sem_pred[0]]} | {sem_pred[1]}/{TOPK} |")
    L.append(f"| BM25 | **{bm_pred[0]}** | {TIER_NAME[bm_pred[0]]} | {bm_pred[1]}/{TOPK} |\n")

    L.append("### Similar texts and why they point to a tier\n")
    L.append("**Semantic top-5:**\n")
    for r, idx in enumerate(sem_idx, 1):
        why = SUB_MEANING.get(subs[idx], "")
        L.append(f"{r}. **Tier {tiers[idx]}** ({TIER_NAME[tiers[idx]]}) — sub `{subs[idx]}`"
                 f"{' = ' + why if why else ''}, sim `{sem[qi,idx]:.3f}`")
        L.append(f"   > {texts[idx][:240]}")
    L.append("\n**BM25 top-5:**\n")
    for r, idx in enumerate(bm_idx, 1):
        why = SUB_MEANING.get(subs[idx], "")
        L.append(f"{r}. **Tier {tiers[idx]}** ({TIER_NAME[tiers[idx]]}) — sub `{subs[idx]}`"
                 f"{' = ' + why if why else ''}, BM25 `{bm_scores[idx]:.1f}`")
        L.append(f"   > {texts[idx][:240]}")
    L.append("\n---\n")

with open(OUT, "w") as f:
    f.write("\n".join(L))
print("written", OUT)
# stdout compact
for qi, k in enumerate(qkeys):
    sv = Counter(tiers[i] for i in topk(sem[qi]))
    bv = Counter(tiers[i] for i in topk(bm25.get_scores(tok(queries[k]))))
    print(f"{k}: semantic=T{sv.most_common(1)[0][0]} {dict(sv)}  |  bm25=T{bv.most_common(1)[0][0]} {dict(bv)}")
