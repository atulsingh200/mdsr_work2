"""
Audience → Email Robustness Test — CrossEncoder2x (deberta-v3-large)
=====================================================================
Same 5 paraphrase variants as test_biencoder_mlp_audience_email.py, but run
through OUR cross-encoder trained on aep_ajo_procedural_workflow so the two
architectures can be compared on identical inputs.

Model  : CrossEncoder2x  (microsoft/deberta-v3-large backbone)
Run    : crossencoder2x_deberta / aep_ajo_procedural_workflow
Task   : P(text1 causally precedes text2) = sigmoid(logit)
Goal   : fwd > rev for every variant (audience creation precedes email send),
         and the margin should stay stable across paraphrases (robustness).
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
    "/runs/crossencoder2x_deberta/aep_ajo_procedural_workflow"
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


# ── scoring helper (joint cross-encoder tokenization) ────────────────────────
@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> tuple[float, float]:
    """Return (prob, logit) for P(text_a causally precedes text_b)."""
    enc = tokenizer(text_a, text_b, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    logit = model(
        enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"),
    ).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    return prob, logit


# ── 5 variants (identical to the bi-encoder test) ────────────────────────────
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
print(f"Model  : {cfg['backbone']}  (cross-encoder 2x)")
print(f"Run    : {_RUN_DIR.name}")
print(f"Metric : P(text1 precedes text2) = sigmoid(logit)")
print(f"Goal   : fwd > rev  for every variant")
print(SEP)

results = []
for label, text1, text2 in VARIANTS:
    fwd_prob, fwd_logit = score_pair(text1, text2)
    rev_prob, rev_logit = score_pair(text2, text1)
    correct = fwd_prob > rev_prob
    delta = fwd_prob - rev_prob
    margin = "strong" if abs(delta) > 0.5 else ("moderate" if abs(delta) > 0.2 else "weak")

    results.append({
        "variant": label, "text1": text1, "text2": text2,
        "fwd_prob": round(fwd_prob, 6), "rev_prob": round(rev_prob, 6),
        "fwd_logit": round(fwd_logit, 4), "rev_logit": round(rev_logit, 4),
        "delta": round(delta, 6), "correct": correct, "margin": margin,
    })

    tag = "PASS" if correct else "FAIL"
    print(f"\n[{tag}] {label}  (margin: {margin})")
    print(f"  text1 : {text1}")
    print(f"  text2 : {text2}")
    print(f"  fwd  P(t1→t2) = {fwd_prob:.4f}  logit = {fwd_logit:+.3f}")
    print(f"  rev  P(t2→t1) = {rev_prob:.4f}  logit = {rev_logit:+.3f}")
    print(f"  Δ (fwd−rev)   = {delta:+.4f}")

n_pass = sum(r["correct"] for r in results)
n_fail = len(results) - n_pass

print(f"\n{SEP}")
print(f"SUMMARY  –  {n_pass}/{len(results)} variants PASS  ({n_fail} FAIL)")
print(SEP)

print(f"\n{'─'*90}")
print("DIRECTIONAL CONSISTENCY TABLE")
print(f"{'Variant':<48}  {'fwd':>6}  {'rev':>6}  {'Δ':>7}  {'Margin':<8}  Result")
print(f"{'─'*48}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*6}")
for r in results:
    tag = "PASS" if r["correct"] else "FAIL"
    print(
        f"{r['variant']:<48}  {r['fwd_prob']:>6.4f}  {r['rev_prob']:>6.4f}"
        f"  {r['delta']:>+7.4f}  {r['margin']:<8}  {tag}"
    )
print()

if n_fail:
    print("Failed variants:")
    for r in results:
        if not r["correct"]:
            print(f"  {r['variant']}")
            print(f"    fwd={r['fwd_prob']:.4f}  rev={r['rev_prob']:.4f}  Δ={r['delta']:+.4f}")
    print()

# robustness: spread of the forward margin across paraphrases
deltas = [r["delta"] for r in results]
print(f"Robustness: Δ range [{min(deltas):+.4f}, {max(deltas):+.4f}]  "
      f"spread={max(deltas) - min(deltas):.4f}  (smaller = more paraphrase-stable)")

# ── save ───────────────────────────────────────────────────────────────────────
out_path = _HERE / "audience_email_results_aepajo.json"
out_path.write_text(json.dumps({
    "model": cfg["backbone"],
    "architecture": "cross-encoder 2x (joint [CLS] t1 [SEP] t2 [SEP])",
    "run": _RUN_DIR.name,
    "checkpoint_epoch": ckpt["epoch"],
    "val_acc": ckpt["val"]["acc"],
    "val_auc": ckpt["val"]["auc"],
    "n_pass": n_pass,
    "n_fail": n_fail,
    "samples": results,
}, indent=2))
print(f"Results saved → {out_path}")
