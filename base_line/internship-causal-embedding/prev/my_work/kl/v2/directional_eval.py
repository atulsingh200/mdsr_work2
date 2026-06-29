"""Zero-shot directional classification on aep_causal_classification.

Task (matches the reference classifier at
internship-causal-embedding/runs/classifier/best_mlp_bge_small):
  given an ordered pair (text_1, text_2), predict label=1 if text_1 causally
  precedes text_2, else 0.  Metrics: accuracy (pred=prob>0.5), AUC(prob), F1.

This script is PARAMETER-FREE / zero-shot: it reuses the asymmetric scorers
already trained on the aep_causal *retrieval* task and predicts direction from
the score margin
        margin = score(t1 -> t2) - score(t2 -> t1)
        pred   = 1 if margin > 0 else 0
For AUC we map the margin to a pseudo-probability via sigmoid(margin / scale)
(scale chosen on the margin std); AUC is threshold-free so scale is irrelevant
to its value — only the ranking of margins matters.

Reference baseline (bge-small dual-encoder + MLP, trained on this data):
  Accuracy 0.8571  AUC 0.9031  F1 0.8558
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
PROJECT_ROOT = KL_DIR.parent.parent
sys.path.insert(0, str(KL_DIR))
sys.path.insert(0, str(KL_DIR / "cross_encoder"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import get_tokenizer, tokenize_batch              # noqa: E402
from v2.model_v2 import CDEv2                                   # noqa: E402
from model_ce import CrossEncoder, get_tokenizer as ce_tok_fn   # noqa: E402

DATA = PROJECT_ROOT / "data_6" / "aep_causal_classification"
OUT_DIR = KL_DIR / "results" / "aep_causal"


def load_dir_pairs(path, limit=None):
    t1, t2, y = [], [], []
    for i, line in enumerate(open(path)):
        if limit and i >= limit:
            break
        r = json.loads(line)
        t1.append(r["text_1"]); t2.append(r["text_2"]); y.append(int(r["label"]))
    return t1, t2, np.array(y)


def report(name, margin, y):
    pred = (margin > 0).astype(int)
    acc = float((pred == y).mean())
    # pseudo-prob for AUC/F1 (AUC is rank-based; scale doesn't change it)
    prob = 1.0 / (1.0 + np.exp(-margin / (np.std(margin) + 1e-9)))
    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(y, pred))
    print(f"  {name:<24} acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
    return {"acc": acc, "auc": auc, "f1": f1}


@torch.no_grad()
def cde_margin(suffix, t1, t2, bs, device, max_len):
    ckpt = torch.load(OUT_DIR / f"cde_v2{suffix}_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    tok = get_tokenizer(cfg["backbone"])
    model = CDEv2(
        cfg["backbone"], cfg["proj_dim"],
        shared_encoder=cfg.get("shared_encoder", False),
        kl_scale=cfg.get("kl_scale", "dim"),
        init_cos_weight=cfg.get("init_cos_weight", 1.0),
        pooling=cfg.get("pooling", "mean"),
        mu_identity=cfg.get("mu_identity", False),
        log_sigma_init=cfg.get("log_sigma_init", 0.0),
        score_type=cfg.get("score_type", "kl"),
    ).to(device).eval()
    # non-strict: KL checkpoints lack the MMD gate params (unused under score_type=kl)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ml = max_len or cfg.get("max_seq_length", 256)

    def enc_cause(texts):
        M, L = [], []
        for i in range(0, len(texts), bs):
            ids, mask = tokenize_batch(tok, texts[i:i+bs], ml)
            mu, ls, de = model.encode_cause(ids.to(device), mask.to(device))
            mu, ls = model.transport(mu, ls, de); M.append(mu); L.append(ls)
        return torch.cat(M), torch.cat(L)

    def enc_effect(texts):
        M, L = [], []
        for i in range(0, len(texts), bs):
            ids, mask = tokenize_batch(tok, texts[i:i+bs], ml)
            mu, ls = model.encode_effect(ids.to(device), mask.to(device)); M.append(mu); L.append(ls)
        return torch.cat(M), torch.cat(L)

    mu1c, ls1c = enc_cause(t1); mu2e, ls2e = enc_effect(t2)
    s_fwd = model.score_pointwise(mu1c, ls1c, mu2e, ls2e).float().cpu().numpy()
    mu2c, ls2c = enc_cause(t2); mu1e, ls1e = enc_effect(t1)
    s_rev = model.score_pointwise(mu2c, ls2c, mu1e, ls1e).float().cpu().numpy()
    return s_fwd - s_rev


@torch.no_grad()
def ce_margin(ce_dirs, t1, t2, bs, device, max_len):
    def score(ce, tok, a, b):
        out = []
        for s in range(0, len(a), bs):
            e = tok(a[s:s+bs], b[s:s+bs], padding=True, truncation="longest_first",
                    max_length=max_len, return_tensors="pt", return_token_type_ids=True)
            ids = e["input_ids"].to(device); mask = e["attention_mask"].to(device)
            tti = e.get("token_type_ids"); tti = tti.to(device) if tti is not None else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out.extend(ce(ids, mask, tti).float().cpu().tolist())
        return np.asarray(out)
    mar = []
    for d in ce_dirs:
        ck = torch.load(Path(d) / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        bb = ck.get("cfg", {}).get("backbone", "google-bert/bert-base-uncased")
        ce = CrossEncoder(bb).to(device).eval(); ce.load_state_dict(ck["model_state_dict"])
        tok = ce_tok_fn(bb)
        m = score(ce, tok, t1, t2) - score(ce, tok, t2, t1)
        mar.append((m - m.mean()) / (m.std() + 1e-9)); del ce; torch.cuda.empty_cache()
    return np.mean(mar, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cde-suffixes", nargs="*", default=["_run5c", "_mmd_pure"])
    ap.add_argument("--ce-dir", nargs="+", default=[
        str(KL_DIR / "cross_encoder/results/aep_causal_rerank"),
        str(KL_DIR / "cross_encoder/results/aep_causal_rerank_v2")])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1, t2, y = load_dir_pairs(DATA / f"directional_{args.split}.jsonl", args.limit)
    print(f"[zero-shot dir] {args.split}: {len(y)} pairs  label1={int(y.sum())} label0={int((1-y).sum())}")
    print(f"  {'majority-baseline':<24} acc={max(y.mean(),1-y.mean()):.4f}")

    res = {}
    margins = {}
    for sfx in args.cde_suffixes:
        m = cde_margin(sfx, t1, t2, args.batch_size, device, args.max_length)
        margins[f"cde{sfx}"] = m
        res[f"cde{sfx}"] = report(f"CDEv2 {sfx}", m, y)
    m_ce = ce_margin(args.ce_dir, t1, t2, args.batch_size, device, args.max_length)
    margins["ce_ens"] = m_ce
    res["ce_ensemble"] = report("CE-ensemble", m_ce, y)
    # combine best cde + ce
    mz = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    best_cde = max([k for k in margins if k.startswith("cde")], key=lambda k: res[k]["acc"])
    res["combo"] = report(f"{best_cde}+CE", mz(margins[best_cde]) + mz(m_ce), y)

    (OUT_DIR / f"directional_zeroshot_{args.split}.json").write_text(json.dumps(res, indent=2))
    print(f"[saved] directional_zeroshot_{args.split}.json")


if __name__ == "__main__":
    main()
