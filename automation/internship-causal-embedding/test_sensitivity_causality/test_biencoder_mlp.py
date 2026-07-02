"""
General Causality Sensitivity Test — Bi-encoder MLP (BGE-small)
=================================================================
Model  : DirectionalClassifier  (Siamese bi-encoder + MLP head)
Ckpt   : runs/classifier/best_mlp_bge_small/best.pt
Backbone: BAAI/bge-small-en-v1.5  (two untied encoders: src + tgt)
Head   : MLP  [cat(A, B, A-B, A*B) → 512 → 128 → 1]

text1 and text2 are encoded SEPARATELY through each tower.
The interaction happens only inside the MLP head via feature concat.

Same 10 general causal samples as test_general_causality.py,
plus the everyday "food → walk" sample, for direct comparison
between the two architectures.

Results are saved to biencoder_mlp_results.json in this folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_REPO_DIR = Path("/mnt/localssd/automation/internship-causal-embedding")
_CKPT     = _REPO_DIR / "runs/classifier/best_mlp_bge_small/best.pt"

sys.path.insert(0, str(_REPO_DIR))
from src.classifier.model import DirectionalClassifier  # noqa: E402
from transformers import AutoTokenizer

# ── load model ────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_bf16 = device.type == "cuda"
print(f"Device : {device}")

print("Loading DirectionalClassifier (bge-small bi-encoder + MLP) …", flush=True)
ckpt = torch.load(_CKPT, map_location=device, weights_only=False)
cfg  = ckpt["cfg"]

model = DirectionalClassifier(
    base_model_name=cfg["base_model"],
    head_cfg=cfg["head_cfg"],
    n_tiers=cfg.get("n_tiers", 0),
)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
MAX_LEN   = cfg["max_seq_len"]

val_info = ckpt.get("val", {})
print(
    f"Loaded  epoch={ckpt['epoch']}  "
    f"val_acc={val_info.get('acc', 'n/a')}  "
    f"val_auc={val_info.get('auc', 'n/a')}\n"
)

_SAFE_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})


# ── scoring helper ─────────────────────────────────────────────────────────────
@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> tuple[float, float]:
    """Return (prob, logit) for P(text_a causally precedes text_b).

    text_a is encoded through model.src tower,
    text_b is encoded through model.tgt tower — independently.
    """
    enc1 = tokenizer(text_a, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")
    enc2 = tokenizer(text_b, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")
    enc1 = {k: v.to(device) for k, v in enc1.items() if k in _SAFE_KEYS}
    enc2 = {k: v.to(device) for k, v in enc2.items() if k in _SAFE_KEYS}

    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
          if use_bf16 else __import__("contextlib").nullcontext
    with ctx():
        logit, _, _ = model(enc1, enc2)

    logit_val = logit.float().cpu().item()
    prob      = torch.sigmoid(torch.tensor(logit_val)).item()
    return prob, logit_val


# ── samples ───────────────────────────────────────────────────────────────────
# Same 10 general causal pairs from test_general_causality.py
# + 1 everyday "food → walk" edge case
SAMPLES: list[tuple[str, str, str, str]] = [
    (
        "S01",
        "Physical / weather",
        "It rained heavily for several hours",
        "The streets flooded and traffic came to a standstill",
    ),
    (
        "S02",
        "Medical",
        "The patient was prescribed and took the medication daily",
        "The patient's symptoms gradually improved over two weeks",
    ),
    (
        "S03",
        "Business / marketing",
        "The company launched a large-scale marketing campaign",
        "Sales increased significantly in the following quarter",
    ),
    (
        "S04",
        "Education",
        "The student studied hard and revised all topics before the exam",
        "The student passed the exam with a high score",
    ),
    (
        "S05",
        "Software / IT reliability",
        "The production server experienced a critical unhandled error",
        "The website went offline and became inaccessible to users",
    ),
    (
        "S06",
        "Sports / performance",
        "The athlete followed a rigorous daily training programme for six months",
        "The athlete won the national championship",
    ),
    (
        "S07",
        "Software development",
        "The developer identified and diagnosed the root cause of the bug",
        "The developer fixed the bug and deployed the patch to production",
    ),
    (
        "S08",
        "Safety / emergency",
        "The fire alarm was triggered in the building",
        "The employees evacuated the building immediately",
    ),
    (
        "S09",
        "Data science / ML",
        "The raw dataset was cleaned, deduplicated, and preprocessed",
        "The machine learning model was trained on the prepared data",
    ),
    (
        "S10",
        "Finance / risk",
        "The bank detected suspicious activity on the customer account",
        "The bank froze the account and notified the customer",
    ),
    (
        "S11",
        "Everyday life (edge case)",
        "I am having food",
        "I am going for a walk",
    ),
]

# ── run tests ─────────────────────────────────────────────────────────────────
W = 94
print("=" * W)
print("GENERAL CAUSALITY SENSITIVITY TEST  —  Bi-encoder MLP (bge-small)")
print(f"Model   : {cfg['base_model']}  (two untied towers + MLP head)")
print(f"Ckpt    : {_CKPT.parent.name}")
print(f"Task    : fwd = P(text1→text2) vs rev = P(text2→text1)  →  expect fwd > rev")
print("=" * W)

records: list[dict] = []
for sid, domain, text1, text2 in SAMPLES:
    fwd_prob, fwd_logit = score_pair(text1, text2)
    rev_prob, rev_logit = score_pair(text2, text1)
    delta   = fwd_prob - rev_prob
    correct = fwd_prob > rev_prob
    margin  = "strong" if abs(delta) > 0.5 else ("moderate" if abs(delta) > 0.2 else "weak")

    rec = {
        "id": sid,
        "domain": domain,
        "text1": text1,
        "text2": text2,
        "fwd_prob": round(fwd_prob, 6),
        "rev_prob": round(rev_prob, 6),
        "fwd_logit": round(fwd_logit, 4),
        "rev_logit": round(rev_logit, 4),
        "delta": round(delta, 6),
        "correct": correct,
        "margin": margin,
    }
    records.append(rec)

    tag = "PASS" if correct else "FAIL"
    print(f"\n[{tag}] {sid} – {domain}  (margin: {margin})")
    print(f"  text1 : {text1}")
    print(f"  text2 : {text2}")
    print(f"  fwd  P(t1→t2) = {fwd_prob:.4f}  logit = {fwd_logit:+.3f}")
    print(f"  rev  P(t2→t1) = {rev_prob:.4f}  logit = {rev_logit:+.3f}")
    print(f"  Δ fwd − rev   = {delta:+.4f}  ({'correct direction' if correct else 'WRONG direction'})")

n_pass   = sum(r["correct"] for r in records)
n_fail   = len(records) - n_pass
n_strong   = sum(1 for r in records if r["margin"] == "strong")
n_moderate = sum(1 for r in records if r["margin"] == "moderate")
n_weak     = sum(1 for r in records if r["margin"] == "weak")

print(f"\n{'=' * W}")
print("SUMMARY")
print(f"  Samples : {len(records)}")
print(f"  PASS    : {n_pass}  |  FAIL : {n_fail}  "
      f"(accuracy = {n_pass / len(records) * 100:.0f}%)")
print(f"  Margin breakdown — strong (|Δ|>0.5): {n_strong}  "
      f"| moderate (|Δ|>0.2): {n_moderate}  | weak (|Δ|≤0.2): {n_weak}")
print("=" * W)

# ── results table ─────────────────────────────────────────────────────────────
print(f"\n{'─' * W}")
print("RESULTS TABLE")
print(f"{'ID':<4}  {'Domain':<30}  {'fwd':>6}  {'rev':>6}  {'Δ':>7}  {'Margin':<8}  Result")
print(f"{'─'*4}  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*6}")
for r in records:
    tag = "PASS" if r["correct"] else "FAIL"
    print(
        f"{r['id']:<4}  {r['domain']:<30}  "
        f"{r['fwd_prob']:>6.4f}  {r['rev_prob']:>6.4f}  "
        f"{r['delta']:>+7.4f}  {r['margin']:<8}  {tag}"
    )
print()

# ── failure analysis ───────────────────────────────────────────────────────────
if n_fail:
    print("FAILURE ANALYSIS")
    for r in records:
        if not r["correct"]:
            print(f"  [{r['id']}] {r['domain']}")
            print(f"    text1 : {r['text1']}")
            print(f"    text2 : {r['text2']}")
            print(f"    fwd={r['fwd_prob']:.4f}  rev={r['rev_prob']:.4f}  "
                  f"Δ={r['delta']:+.4f}")
    print()

# ── save results ───────────────────────────────────────────────────────────────
out_path = _HERE / "biencoder_mlp_results.json"
output = {
    "model": cfg["base_model"],
    "architecture": "bi-encoder (untied) + MLP head",
    "head_cfg": cfg["head_cfg"],
    "run": _CKPT.parent.name,
    "checkpoint_epoch": ckpt["epoch"],
    "val_acc": val_info.get("acc"),
    "val_auc": val_info.get("auc"),
    "n_samples": len(records),
    "n_pass": n_pass,
    "n_fail": n_fail,
    "accuracy": round(n_pass / len(records), 4),
    "samples": records,
}
out_path.write_text(json.dumps(output, indent=2))
print(f"Results saved → {out_path}")
