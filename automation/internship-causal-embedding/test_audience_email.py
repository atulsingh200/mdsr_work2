"""
Robustness test: "build an audience" → "send an email to the customers"

The model is a directional cross-encoder trained on procedural AEP/AJO
workflow steps.  It scores P(text1 precedes text2) via sigmoid(logit).

Expected behaviour: score(text1→text2) > score(text2→text1) for all variants
because building an audience is a causal prerequisite for sending an email.

5 variants are tested; each is a paraphrase / surface-form modification
of the same underlying causal pair, testing model robustness.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
_MODEL_SRC = Path(
    "/mnt/localssd/new_base_line/internship-causal-embedding"
    "/my_work/kl/cross_encoder_2x"
)
_RUN_DIR = Path(
    "/mnt/localssd/automation/internship-causal-embedding"
    "/runs/crossencoder2x_deberta/procedural_workflow"
)

sys.path.insert(0, str(_MODEL_SRC))
from model_ce2x import CrossEncoder2x, get_tokenizer  # noqa: E402

# ── load model ────────────────────────────────────────────────────────────────
cfg = json.loads((_RUN_DIR / "config.json").read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
print(f"Checkpoint loaded  (epoch {ckpt['epoch']}, "
      f"val_acc={ckpt['val']['acc']:.4f}, val_auc={ckpt['val']['auc']:.4f})")

tokenizer = get_tokenizer(cfg["backbone"])
MAX_LEN = cfg["max_seq_len"]


# ── scoring helper ─────────────────────────────────────────────────────────────
@torch.no_grad()
def score(text_a: str, text_b: str) -> tuple[float, float]:
    """Return (prob, logit) for P(text_a precedes text_b)."""
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


# ── test variants ─────────────────────────────────────────────────────────────
# Each tuple: (label, text1, text2)
# text1 = "build audience" concept  (should come FIRST)
# text2 = "send email"     concept  (should come SECOND)
VARIANTS: list[tuple[str, str, str]] = [
    (
        "V1 – Minimal / clean",
        "Build an audience",
        "Send an email to the customers",
    ),
    (
        "V2 – Paraphrase (formal)",
        "Create and define your target audience segment",
        "Send a promotional email campaign to the customers",
    ),
    (
        "V3 – Platform action style (AEP/AJO verbs)",
        "Navigate to Audiences and click Create audience to define customer criteria",
        "Configure the email channel and send the message to the segmented customers",
    ),
    (
        "V4 – Marketing language",
        "Identify and group target users into a reusable audience based on their attributes",
        "Deliver a personalized email to all customers in the selected audience",
    ),
    (
        "V5 – Stress test (longer, swapped surface emphasis)",
        "Set up and publish your audience segment with the desired attribute filters",
        "Write, schedule, and send a targeted marketing email to the customer list",
    ),
]

# ── run and report ─────────────────────────────────────────────────────────────
SEP = "─" * 90
print(f"\n{SEP}")
print("ROBUSTNESS TEST  –  audience → email  (5 variants)")
print(f"Model  : {cfg['backbone']}")
print(f"Run    : {_RUN_DIR.name}")
print(f"Metric : P(text1 precedes text2) = sigmoid(logit)")
print(f"Goal   : fwd > rev  for every variant  (text1 causally precedes text2)")
print(SEP)

results = []
for label, text1, text2 in VARIANTS:
    fwd_prob, fwd_logit = score(text1, text2)
    rev_prob, rev_logit = score(text2, text1)
    correct = fwd_prob > rev_prob
    delta = fwd_prob - rev_prob
    results.append({
        "variant": label,
        "text1": text1,
        "text2": text2,
        "fwd_prob": fwd_prob,
        "rev_prob": rev_prob,
        "fwd_logit": fwd_logit,
        "rev_logit": rev_logit,
        "delta": delta,
        "correct": correct,
    })

    status = "PASS" if correct else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"  text1 : {text1}")
    print(f"  text2 : {text2}")
    print(f"  fwd  P(t1→t2) = {fwd_prob:.4f}  logit={fwd_logit:+.3f}")
    print(f"  rev  P(t2→t1) = {rev_prob:.4f}  logit={rev_logit:+.3f}")
    print(f"  Δ (fwd−rev)   = {delta:+.4f}")

n_pass = sum(r["correct"] for r in results)
n_fail = len(results) - n_pass

print(f"\n{SEP}")
print(f"SUMMARY  –  {n_pass}/{len(results)} variants PASS  ({n_fail} FAIL)")
print(SEP)

if n_fail:
    print("\nFailed variants:")
    for r in results:
        if not r["correct"]:
            print(f"  {r['variant']}")
            print(f"    fwd={r['fwd_prob']:.4f}  rev={r['rev_prob']:.4f}  "
                  f"Δ={r['delta']:+.4f}")

# ── directional consistency ────────────────────────────────────────────────────
print(f"\n{'─'*90}")
print("DIRECTIONAL CONSISTENCY TABLE")
print(f"{'Variant':<45}  {'fwd':>6}  {'rev':>6}  {'Δ':>7}  {'Result':>6}")
print(f"{'─'*45}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")
for r in results:
    tag = "PASS" if r["correct"] else "FAIL"
    print(f"{r['variant']:<45}  {r['fwd_prob']:>6.4f}  {r['rev_prob']:>6.4f}"
          f"  {r['delta']:>+7.4f}  {tag:>6}")

print()
