"""
Audience → Email Robustness Test — Bi-encoder MLP (BGE-small)
==============================================================
Same 5 variants as test_audience_email.py (cross-encoder baseline),
now run through the Siamese bi-encoder MLP model so both architectures
can be compared side-by-side on identical inputs.

Model  : DirectionalClassifier  (two untied bge-small towers + MLP head)
Ckpt   : runs/classifier/best_mlp_bge_small/best.pt
Task   : P(text1 causally precedes text2) = sigmoid(logit)
Goal   : fwd > rev for every variant
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer

_REPO_DIR = Path("/mnt/localssd/automation/internship-causal-embedding")
_CKPT     = _REPO_DIR / "runs/classifier/best_mlp_bge_small/best.pt"

sys.path.insert(0, str(_REPO_DIR))
from src.classifier.model import DirectionalClassifier  # noqa: E402

# ── load model ────────────────────────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    f"val_acc={val_info.get('acc', 'n/a'):.4f}  "
    f"val_auc={val_info.get('auc', 'n/a'):.4f}\n"
)

_SAFE_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})


# ── scoring helper ─────────────────────────────────────────────────────────────
@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> tuple[float, float]:
    enc1 = tokenizer(text_a, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")
    enc2 = tokenizer(text_b, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")
    enc1 = {k: v.to(device) for k, v in enc1.items() if k in _SAFE_KEYS}
    enc2 = {k: v.to(device) for k, v in enc2.items() if k in _SAFE_KEYS}

    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
          if use_bf16 else nullcontext
    with ctx():
        logit, _, _ = model(enc1, enc2)

    logit_val = logit.float().cpu().item()
    prob      = torch.sigmoid(torch.tensor(logit_val)).item()
    return prob, logit_val


# ── 5 variants (identical to test_audience_email.py) ─────────────────────────
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
print(f"Model  : {cfg['base_model']}  (bi-encoder + MLP)")
print(f"Run    : {_CKPT.parent.name}")
print(f"Metric : P(text1 precedes text2) = sigmoid(logit)")
print(f"Goal   : fwd > rev  for every variant")
print(SEP)

results = []
for label, text1, text2 in VARIANTS:
    fwd_prob, fwd_logit = score_pair(text1, text2)
    rev_prob, rev_logit = score_pair(text2, text1)
    correct = fwd_prob > rev_prob
    delta   = fwd_prob - rev_prob
    margin  = "strong" if abs(delta) > 0.5 else ("moderate" if abs(delta) > 0.2 else "weak")

    results.append({
        "variant": label,
        "text1": text1,
        "text2": text2,
        "fwd_prob": round(fwd_prob, 6),
        "rev_prob": round(rev_prob, 6),
        "fwd_logit": round(fwd_logit, 4),
        "rev_logit": round(rev_logit, 4),
        "delta": round(delta, 6),
        "correct": correct,
        "margin": margin,
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

# ── save ───────────────────────────────────────────────────────────────────────
out_path = Path(__file__).resolve().parent / "biencoder_mlp_audience_email_results.json"
out_path.write_text(json.dumps({
    "model": cfg["base_model"],
    "architecture": "bi-encoder (untied) + MLP head",
    "run": _CKPT.parent.name,
    "checkpoint_epoch": ckpt["epoch"],
    "val_acc": val_info.get("acc"),
    "val_auc": val_info.get("auc"),
    "n_pass": n_pass,
    "n_fail": n_fail,
    "samples": results,
}, indent=2))
print(f"Results saved → {out_path}")
