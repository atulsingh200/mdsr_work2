import json
from itertools import permutations

texts = [
    "I need to set up the audience in AEP by defining lapsed customers (no purchase or interaction in 90+ days) and high-value criteria (lifetime value >= $500), then apply email consent filters, preview the size estimate, name it appropriately, and choose between batch or streaming evaluation.",
    "Now I'm moving to AJO to build a personalized offer for \"20% Off - July 4th\" valid from June 28 through July 7, 2026, with channel-specific representations for email and display media. I'll set the eligibility rule to target the lapsed high-value audience, configure priority and capping limits, then create a fallback offer and decision collection.",
    "I'm creating a scheduled campaign in AJO that links to the audience from Phase 1 and sets the launch date for July 4th, 2026.",
    "I'm adding an email action to the campaign with a July 4th-themed template that includes personalized elements like the customer's name and their unique 20% off code. I'll configure the subject line, sender details, unsubscribe link, and tracking for opens and clicks, then preview and test before sending.",
    "I'm activating the lapsed high-value audience to paid media destinations like Google Ads Customer Match and Meta Ads by mapping identity attributes and scheduling the sync to occur before July 4th.",
    "Before going live, I need to QA the email and creative assets, verify the audience size is statistically significant, set up tracking for open rates, click-through rates, conversions, and revenue lift, then get final approval to launch.",
]

labels = [f"t{i+1}" for i in range(len(texts))]

pairs = [
    {"label1": labels[i], "label2": labels[j], "text1": texts[i], "text2": texts[j]}
    for i, j in permutations(range(len(texts)), 2)
]

out = "/mnt/localssd/test_samples_milan.json"
with open(out, "w") as f:
    json.dump(pairs, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(pairs)} ordered pairs from {len(texts)} texts to {out}")
