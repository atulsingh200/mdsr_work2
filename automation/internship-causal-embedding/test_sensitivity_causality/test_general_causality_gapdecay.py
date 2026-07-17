"""
General Causality Sensitivity Test
====================================
Model  : CrossEncoder2x  (microsoft/deberta-v3-large backbone)
Run    : crossencoder2x_deberta / procedural_workflow
Task   : P(text1 causally precedes / leads to text2)

The model was trained exclusively on AEP/AJO procedural workflow steps.
This test probes it on 10 *general* causal pairs drawn from diverse domains
(physical world, medicine, business, software, safety, ML/data, etc.)
to reveal where the model generalises and where it fails.

Each sample is scored in both directions:
  fwd  = P(text1 → text2)   ← should be HIGH  (correct causal order)
  rev  = P(text2 → text1)   ← should be LOW   (reversed / wrong order)

A sample PASSES if fwd > rev.
Results are printed to stdout and saved as results.json in this folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_MODEL_SRC = Path(
    "/mnt/localssd/causal-embedding-research/new_base_line/internship-causal-embedding"
    "/my_work/kl/cross_encoder_2x"
)
_RUN_DIR = Path(
    "/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding"
    "/runs/crossencoder2x_deberta/merged_gapdecay_procedural"
)

sys.path.insert(0, str(_MODEL_SRC))
from model_ce2x import CrossEncoder2x, get_tokenizer  # noqa: E402

# ── load model ────────────────────────────────────────────────────────────────
cfg = json.loads((_RUN_DIR / "config.json").read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

print("Loading CrossEncoder2x (deberta-v3-large) …", flush=True)
model = CrossEncoder2x(
    backbone=cfg["backbone"],
    n_layers=cfg["n_layers"],
    dropout=cfg["dropout"],
    native_backbone=cfg["native_backbone"],
).to(device)

ckpt = torch.load(_RUN_DIR / "best.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print(
    f"Loaded  epoch={ckpt['epoch']}  "
    f"val_acc={ckpt['val']['acc']:.4f}  val_auc={ckpt['val']['auc']:.4f}\n"
)

tokenizer = get_tokenizer(cfg["backbone"])
MAX_LEN = cfg["max_seq_len"]


# ── scoring helper ─────────────────────────────────────────────────────────────
@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> tuple[float, float]:
    """Return (prob, logit) for P(text_a causally precedes text_b)."""
    enc = tokenizer(
        text_a, text_b,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logit = model(
        enc["input_ids"],
        enc.get("attention_mask"),
        enc.get("token_type_ids"),
    ).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    return prob, logit


# ── 10 general causal pairs ────────────────────────────────────────────────────
# Format: (id, domain, text1 [cause], text2 [effect])
# text1 causally precedes / enables text2 in every case.
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
]


# ── run tests ─────────────────────────────────────────────────────────────────
W = 94
print("=" * W)
print("GENERAL CAUSALITY SENSITIVITY TEST")
print(f"Model : {cfg['backbone']}  |  run: {_RUN_DIR.name}")
print(f"Task  : fwd = P(text1→text2) vs rev = P(text2→text1)  →  expect fwd > rev")
print("=" * W)

records: list[dict] = []
for sid, domain, text1, text2 in SAMPLES:
    fwd_prob, fwd_logit = score_pair(text1, text2)
    rev_prob, rev_logit = score_pair(text2, text1)
    delta = fwd_prob - rev_prob
    correct = fwd_prob > rev_prob
    margin = "strong" if abs(delta) > 0.5 else ("moderate" if abs(delta) > 0.2 else "weak")

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

n_pass = sum(r["correct"] for r in records)
n_fail = len(records) - n_pass
n_strong = sum(1 for r in records if r["margin"] == "strong")
n_moderate = sum(1 for r in records if r["margin"] == "moderate")
n_weak = sum(1 for r in records if r["margin"] == "weak")

print(f"\n{'=' * W}")
print(f"SUMMARY")
print(f"  Samples : {len(records)}")
print(f"  PASS    : {n_pass}  |  FAIL : {n_fail}  "
      f"(accuracy = {n_pass / len(records) * 100:.0f}%)")
print(f"  Margin breakdown — strong (|Δ|>0.5): {n_strong}  "
      f"| moderate (|Δ|>0.2): {n_moderate}  | weak: {n_weak}")
print("=" * W)

# ── compact results table ──────────────────────────────────────────────────────
print(f"\n{'─' * W}")
print("RESULTS TABLE")
print(f"{'ID':<4}  {'Domain':<30}  {'fwd':>6}  {'rev':>6}  {'Δ':>7}  {'Margin':<8}  {'Result'}")
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
    print("  The following samples were scored incorrectly (rev > fwd):")
    for r in records:
        if not r["correct"]:
            print(f"  [{r['id']}] {r['domain']}")
            print(f"    text1 : {r['text1']}")
            print(f"    text2 : {r['text2']}")
            print(f"    fwd={r['fwd_prob']:.4f}  rev={r['rev_prob']:.4f}  Δ={r['delta']:+.4f}")
    print()
    print("  Likely reason: the model was trained on AEP/AJO procedural workflow")
    print("  steps (platform-specific action verbs). Pairs whose surface form does")
    print("  not resemble workflow instructions may fall outside the training")
    print("  distribution and be scored by spurious lexical cues instead of")
    print("  genuine causal reasoning.")
else:
    print("No failures — the model correctly ordered all general causal pairs.")
    print("Note: this may reflect broad distributional overlap between the")
    print("training domain (procedural steps) and general cause→effect language,")
    print("or it could indicate over-confident predictions on easy pairs.")

# ── save results ───────────────────────────────────────────────────────────────
out_path = _HERE / "results_gapdecay.json"
output = {
    "model": cfg["backbone"],
    "run": _RUN_DIR.name,
    "checkpoint_epoch": ckpt["epoch"],
    "val_acc": ckpt["val"]["acc"],
    "val_auc": ckpt["val"]["auc"],
    "n_samples": len(records),
    "n_pass": n_pass,
    "n_fail": n_fail,
    "accuracy": round(n_pass / len(records), 4),
    "samples": records,
}
out_path.write_text(json.dumps(output, indent=2))
print(f"\nResults saved → {out_path}")
