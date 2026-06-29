"""
Build a *training/val* dataset augmented with IN-DOMAIN hard negatives mined by
the trained dual-encoder directional classifier (best_mlp_bge_small).

Idea ("hardest-fooling y*"): for each positive  x -> y  (label=1), among all
WRONG-tier candidates y* (tier(y*) < tier(x)), pick the y* that the trained model
most confidently — but WRONGLY — scores as a valid causal pair, i.e. highest
P(x causally precedes y*) = sigmoid(dir_logit(x, y*)). These are the model's true
failure cases and make the strongest training signal for a cross-encoder.

Like build_hard_neg_semantic.py, this KEEPS every original row (label 0 and 1) and
only ADDS top-k hard negs for label-1 positives. Positives with no wrong-tier
candidate (tier(x)==1) are kept as-is.

Efficiency: the model encodes src(text_1) and tgt(text_2) independently with CLS
pooling + L2 norm, then runs an MLP head on [A;B;A-B;A*B]. So we precompute CLS
embeddings ONCE for every unique x (as src) and every unique candidate text (as
tgt), then the per-(x, y*) scoring is just the cheap head forward — no repeated
BERT passes.

Output (per --split):
  data/aep_causal_classification_hard_neg_indomain/directional_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# Support `python -m src.classifier.build_hard_neg_indomain` and direct file run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.classifier.model import DirectionalClassifier
else:
    from .model import DirectionalClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir",
                   default="data/aep_causal_classification_34")
    p.add_argument("--out-dir",
                   default="data/aep_causal_classification_hard_neg_indomain")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--ckpt",
                   default="runs/classifier/best_mlp_bge_small/best.pt")
    p.add_argument("--topk", type=int, default=3,
                   help="Hard negatives mined per label-1 positive")
    p.add_argument("--encode-batch", type=int, default=512,
                   help="Batch size for the BERT CLS encoding pass")
    p.add_argument("--head-batch", type=int, default=8192,
                   help="Batch size for the cheap head forward over (x,y*) pairs")
    return p.parse_args()


@torch.no_grad()
def encode_cls(encoder, tokenizer, texts, max_len, device, use_bf16, batch):
    """CLS-pool + L2-normalize a list of texts through one encoder. (N, d)."""
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.autocast(device_type="cpu", enabled=False))
    out = np.empty((len(texts), encoder.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        with ctx:
            o = encoder(**enc)
            cls = o.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
        out[i:i + len(chunk)] = cls.float().cpu().numpy()
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"

    in_path = os.path.join(args.source_dir, f"directional_{args.split}.jsonl")
    out_path = os.path.join(args.out_dir, f"directional_{args.split}.jsonl")

    with open(in_path) as f:
        rows = [json.loads(l) for l in f]
    positives = [r for r in rows if r["label"] == 1]
    print(f"[{args.split}] total rows: {len(rows)}  |  positives: {len(positives)}")

    # ---- Load trained dual-encoder (rebuild from cfg, like eval_only.py) ----
    print(f"Loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    base_model = cfg["base_model"]
    head_cfg = cfg["head_cfg"]
    n_tiers = cfg.get("n_tiers", 0) or (15 if cfg.get("tier_aux_weight", 0.0) > 0 else 0)
    max_len = cfg["max_seq_len"]
    print(f"  base_model={base_model}  head_cfg={head_cfg}  max_seq_len={max_len}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = DirectionalClassifier(base_model, head_cfg=head_cfg, n_tiers=n_tiers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- Pool of wrong-tier candidates, indexed by tier ----
    chunk_pool_by_tier: dict[int, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()
    for r in rows:
        for text_key, tier_key, sub_key, url_key in [
            ("text_1", "tier_1", "sub_1", "url_1"),
            ("text_2", "tier_2", "sub_2", "url_2"),
        ]:
            text = r[text_key]
            if text not in seen_texts:
                seen_texts.add(text)
                chunk_pool_by_tier[r[tier_key]].append({
                    "text": text, "tier": r[tier_key],
                    "sub": r[sub_key], "url": r[url_key],
                })
    print("Unique chunks per tier:")
    for t in sorted(chunk_pool_by_tier):
        print(f"  T{t}: {len(chunk_pool_by_tier[t])}")

    # ---- Precompute CLS embeddings once ----
    # src embeddings for every unique positive x (text_1)
    x_texts = sorted({r["text_1"] for r in positives})
    x_idx = {t: i for i, t in enumerate(x_texts)}
    print(f"\nEncoding {len(x_texts)} unique x texts (src) ...")
    emb_x = encode_cls(model.src, tokenizer, x_texts, max_len, device,
                       use_bf16, args.encode_batch)

    # tgt embeddings for every unique candidate text (all pool chunks)
    cand_texts = sorted(seen_texts)
    cand_idx = {t: i for i, t in enumerate(cand_texts)}
    print(f"Encoding {len(cand_texts)} unique candidate texts (tgt) ...")
    emb_c = encode_cls(model.tgt, tokenizer, cand_texts, max_len, device,
                       use_bf16, args.encode_batch)

    emb_x_t = torch.from_numpy(emb_x).to(device)
    emb_c_t = torch.from_numpy(emb_c).to(device)

    # ---- Build the full list of (positive_row, candidate) pairs to score ----
    # We score in big batches through the head only.
    pair_x_rows: list[int] = []   # which embedding row (in emb_x) is x
    pair_c_rows: list[int] = []   # which embedding row (in emb_c) is y*
    pair_pos_id: list[int] = []   # index into `positives`
    pair_cand_meta: list[dict] = []

    n_skipped = 0
    for pid, r in enumerate(positives):
        tier_x = r["tier_1"]
        text_x, text_y = r["text_1"], r["text_2"]
        cands = [c for t in range(1, tier_x) for c in chunk_pool_by_tier[t]
                 if c["text"] != text_x and c["text"] != text_y]
        if not cands:
            n_skipped += 1
            continue
        for c in cands:
            pair_x_rows.append(x_idx[text_x])
            pair_c_rows.append(cand_idx[c["text"]])
            pair_pos_id.append(pid)
            pair_cand_meta.append(c)

    print(f"\nScoring {len(pair_x_rows)} (x, y*) candidate pairs through head "
          f"(positives skipped: {n_skipped}) ...")

    # ---- Run the head on [A;B;A-B;A*B] in big batches ----
    @torch.no_grad()
    def head_forward(feat):
        if model.head_backbone is not None:
            hidden = model.head_backbone(feat)
            return model.head_final(hidden).squeeze(-1)
        return model.head(feat)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.autocast(device_type="cpu", enabled=False))
    probs = np.empty(len(pair_x_rows), dtype=np.float32)
    xr = torch.tensor(pair_x_rows, device=device)
    cr = torch.tensor(pair_c_rows, device=device)
    for i in range(0, len(pair_x_rows), args.head_batch):
        a = emb_x_t[xr[i:i + args.head_batch]]
        b = emb_c_t[cr[i:i + args.head_batch]]
        feat = torch.cat([a, b, a - b, a * b], dim=-1)
        with ctx:
            logit = head_forward(feat)
        probs[i:i + a.size(0)] = torch.sigmoid(logit).float().cpu().numpy()

    # ---- Group by positive, pick top-k highest-prob (hardest fooling) ----
    by_pos: dict[int, list[int]] = defaultdict(list)
    for j, pid in enumerate(pair_pos_id):
        by_pos[pid].append(j)

    output_rows: list[dict] = []
    for r in rows:
        out = dict(r)
        out.setdefault("hard_neg", False)
        output_rows.append(out)

    n_mined = 0
    dirprobs_all: list[float] = []
    for pid, js in by_pos.items():
        r = positives[pid]
        js_sorted = sorted(js, key=lambda j: -probs[j])
        k = min(args.topk, len(js_sorted))
        for j in js_sorted[:k]:
            c = pair_cand_meta[j]
            dirprobs_all.append(float(probs[j]))
            output_rows.append({
                "text_1": r["text_1"], "text_2": c["text"], "label": 0,
                "tier_1": r["tier_1"], "tier_2": c["tier"],
                "sub_1": r["sub_1"], "sub_2": c["sub"],
                "url_1": r["url_1"], "url_2": c["url"],
                "hard_neg": True,
                "dirprob": round(float(probs[j]), 6),
            })
            n_mined += 1

    print(f"Mined hard negs: {n_mined}")
    print(f"Total output rows: {len(output_rows)} "
          f"(orig {len(rows)} + mined {n_mined})")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved → {out_path}")

    label_counts = Counter(r["label"] for r in output_rows)
    hn_counts = Counter(r.get("hard_neg", False) for r in output_rows)
    print(f"Label dist: {dict(label_counts)}")
    print(f"Hard neg dist: {dict(hn_counts)}")
    if dirprobs_all:
        print(f"dirprob(x->y*) stats: min={min(dirprobs_all):.4f}  "
              f"mean={np.mean(dirprobs_all):.4f}  max={max(dirprobs_all):.4f}")


if __name__ == "__main__":
    main()
