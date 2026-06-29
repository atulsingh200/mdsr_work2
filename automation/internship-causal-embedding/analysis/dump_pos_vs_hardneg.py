"""Pair each positive x->y (label=1) with its mined hard negative x->y* (label=0)
and dump them sorted by sim(y, y*) so y can be compared to y* by hand.

Pairing rule (from build_hard_neg_test.py): the positive and its hard negative are
emitted CONSECUTIVELY and share the anchor x (text_1/url_1). We also attach the model's
predicted prob for each side (row-aligned 1:1 with test_predictions_eval.jsonl).

Outputs:
  analysis/pos_vs_hardneg_compare.csv   -- all 2828 triples, sorted by sim desc
  analysis/pos_vs_hardneg_top50.txt     -- readable top-50 with full y and y* text
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/aep_causal_classification_hard_neg/directional_test.jsonl"
PREDS = ROOT / "runs/classifier/best_mlp_bge_small/test_predictions_eval.jsonl"
OUT = ROOT / "analysis"
THR = 0.5

recs = [json.loads(l) for l in open(DATA)]
preds = [json.loads(l) for l in open(PREDS)]
assert len(recs) == len(preds)
for r, p in zip(recs, preds):
    r["prob"] = p["prob"]  # row-aligned

# pair consecutive (positive, hard_neg) sharing anchor x
pairs = []
i = 0
while i < len(recs) - 1:
    a, b = recs[i], recs[i + 1]
    if a["label"] == 1 and b.get("hard_neg") and a["text_1"] == b["text_1"]:
        pairs.append((a, b))
        i += 2
    else:
        i += 1
assert len(pairs) == sum(r.get("hard_neg", False) for r in recs), "pairing incomplete"

pairs.sort(key=lambda ab: -ab[1]["sim_y_ystar"])

# ---- CSV: one row per (x, y, y*) triple ----
csv_path = OUT / "pos_vs_hardneg_compare.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "sim_y_ystar",
        "prob_pos_xy", "prob_hardneg_xystar",
        "pred_pos", "pred_hardneg", "hardneg_fooled",
        "tier_x", "tier_y", "tier_ystar",
        "sub_x", "sub_y", "sub_ystar",
        "url_x", "url_y", "url_ystar",
        "text_x", "text_y_TRUE", "text_ystar_HARDNEG",
    ])
    for pos, neg in pairs:
        pp = 1 if pos["prob"] >= THR else 0
        pn = 1 if neg["prob"] >= THR else 0
        w.writerow([
            f"{neg['sim_y_ystar']:.6f}",
            f"{pos['prob']:.6f}", f"{neg['prob']:.6f}",
            pp, pn, int(pn == 1),
            pos["tier_1"], pos["tier_2"], neg["tier_2"],
            pos["sub_1"], pos["sub_2"], neg["sub_2"],
            pos["url_1"], pos["url_2"], neg["url_2"],
            pos["text_1"], pos["text_2"], neg["text_2"],
        ])

# ---- readable top-50 ----
txt_path = OUT / "pos_vs_hardneg_top50.txt"
with open(txt_path, "w") as f:
    f.write("TOP 50 hard-negative triples by sim(y, y*)  -- compare y (TRUE effect) vs y* (hard neg)\n")
    f.write("=" * 100 + "\n\n")
    for rank, (pos, neg) in enumerate(pairs[:50], 1):
        pn = 1 if neg["prob"] >= THR else 0
        f.write(f"[{rank:2d}] sim(y,y*) = {neg['sim_y_ystar']:.4f}    "
                f"hard-neg model prob={neg['prob']:.3f} -> {'FOOLED (pred=1)' if pn else 'rejected (pred=0)'}    "
                f"pos prob={pos['prob']:.3f}\n")
        f.write(f"     direction: x=tier{pos['tier_1']}  y=tier{pos['tier_2']}  y*=tier{neg['tier_2']}   "
                f"(y* tier < x tier => reversed)\n")
        f.write(f"     X  (anchor, tier{pos['tier_1']}, sub {pos['sub_1']}):\n        {pos['text_1']}\n")
        f.write(f"     Y  (TRUE effect,  tier{pos['tier_2']}, sub {pos['sub_2']}):\n        {pos['text_2']}\n")
        f.write(f"     Y* (HARD NEG,     tier{neg['tier_2']}, sub {neg['sub_2']}):\n        {neg['text_2']}\n")
        f.write("-" * 100 + "\n")

print(f"paired triples: {len(pairs)}")
print(f"sim range: {pairs[-1][1]['sim_y_ystar']:.4f} .. {pairs[0][1]['sim_y_ystar']:.4f}")
print("wrote:")
print(" ", csv_path)
print(" ", txt_path)
