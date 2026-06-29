#!/usr/bin/env python3
"""Score text pairs with the contrastive cross-encoder (ce_contrastive).

Loads the single BERT-base cross-encoder trained with InfoNCE contrastive loss
and scores each pair as:

    P(text_1 causally precedes text_2) = sigmoid(logit)

Reads /mnt/localssd/test_samples_milan.json (list of {label1,label2,text1,text2})
and writes scores back out as JSON.

Usage:
  cd /mnt/localssd/automation/internship-causal-embedding
  source .venv/bin/activate
  python3 /mnt/localssd/score_ce_contrastive.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/localssd/automation/internship-causal-embedding")
CE2X_DIR = Path("/mnt/localssd/new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x")
sys.path.insert(0, str(CE2X_DIR))
from model_ce2x import CrossEncoder2x  # noqa: E402

CKPT = REPO / "runs/crossencoder2x/ce2x_org/best.pt"
CFG_PATH = REPO / "runs/crossencoder2x/ce2x_org/config.json"

SAMPLES = Path("runs/crossencoder2x/new_ajo_ce2x_org/best.pt")
OUT = Path("/mnt/localssd/test_samples_milan_simple_ce_claude_data.json")

BASE_MODEL = "google-bert/bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 16


def load_model(ckpt_path: Path, device) -> CrossEncoder2x:
    cfg = json.loads(CFG_PATH.read_text())
    base = cfg.get("backbone", BASE_MODEL)
    n_layers = cfg.get("n_layers", 24)
    dropout = cfg.get("dropout", 0.1)
    model = CrossEncoder2x(backbone=base, n_layers=n_layers, dropout=dropout).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def score_pairs(model, tokenizer, t1, t2, device, use_bf16) -> list[float]:
    """Return P(text_1 precedes text_2) for each pair."""
    probs: list[float] = []
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.no_grad())
    for s in range(0, len(t1), BATCH_SIZE):
        bt1, bt2 = t1[s:s + BATCH_SIZE], t2[s:s + BATCH_SIZE]
        enc = tokenizer(bt1, bt2, padding=True, truncation="longest_first",
                        max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with ctx:
            logits = model(**enc)
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"device={device}  bf16={use_bf16}")

    pairs = json.loads(SAMPLES.read_text())
    t1 = [p["text1"] for p in pairs]
    t2 = [p["text2"] for p in pairs]
    print(f"loaded {len(pairs)} pairs from {SAMPLES}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"loading contrastive CE  <- {CKPT}")
    model = load_model(CKPT, device)
    probs = score_pairs(model, tokenizer, t1, t2, device, use_bf16)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    for p, prob in zip(pairs, probs):
        results.append({
            **{k: p[k] for k in ("label1", "label2") if k in p},
            "text1": p["text1"],
            "text2": p["text2"],
            "score": round(prob, 6),   # P(text1 causally precedes text2)
            "pred": int(prob > 0.5),   # 1 = text1 causally precedes text2
        })

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(results)} scored pairs to {OUT}")
    print("(score = P(text1 causally precedes text2))\n")
    for r in results:
        lbl = f"{r.get('label1','?')}->{r.get('label2','?')}"
        print(f"  {lbl:10s} score={r['score']:.4f}  pred={r['pred']}")


if __name__ == "__main__":
    main()
