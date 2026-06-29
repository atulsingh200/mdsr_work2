"""Verify fine-tuned BiEncoder AUC vs 4 semantic hard negatives (per-query z-score).

Same evaluation protocol as verify_cdev2_hardneg_auc.py but uses the two
underlying fine-tuned BERT models (anchor_encoder / positive_encoder) directly.
Score = cosine similarity between anchor_encoder(cause) and positive_encoder(effect).

Run:
  cd /mnt/localssd/internship-causal-embedding-merge-followup
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python my_work/kl/v2/verify_biencoder_hardneg_auc.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

HERE   = Path(__file__).resolve().parent          # my_work/kl/v2
KL_DIR = HERE.parent                              # my_work/kl
sys.path.insert(0, str(KL_DIR))

from dataset import get_tokenizer, load_pairs, tokenize_batch  # noqa: E402

# ── data ──────────────────────────────────────────────────────────────────────
OUT_DIR   = KL_DIR / "results" / "aep_causal"
CKPT_PATH = KL_DIR.parent.parent / "finetune_eval" / "results" / "aep_causal" / "checkpoint_best.pt"

pairs     = load_pairs(OUT_DIR / "test_pairs.jsonl")
anchors   = [a for a, _ in pairs]
positives = [p for _, p in pairs]
N         = len(pairs)
hn        = np.load(OUT_DIR / "test_hard_negatives.npy")   # (N, 4)
K         = hn.shape[1]
print(f"[data] N={N}  hard-neg K={K}  miner=all-MiniLM-L6-v2")

# ── load checkpoint ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
cfg    = ckpt["cfg"]
ml     = cfg.get("max_seq_length", 256)
pooling = cfg.get("pooling", "cls")
print(f"[model] BiEncoder  backbone={cfg['backbone']}  pooling={pooling}  "
      f"val_mrr={ckpt['metrics']['mrr']:.4f}  device={device}")

tok = get_tokenizer(cfg["backbone"])

# rebuild two bare BERT models and load weights from the checkpoint
def _load_encoder(state_dict, prefix):
    enc = AutoModel.from_pretrained(cfg["backbone"])
    enc_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    enc.load_state_dict(enc_sd, strict=True)
    return enc.to(device).eval()

sd = ckpt["model_state_dict"]
anchor_enc   = _load_encoder(sd, "anchor_encoder.")
positive_enc = _load_encoder(sd, "positive_encoder.")

# ── pool helper ───────────────────────────────────────────────────────────────
def _pool(out, mask, method):
    if method == "cls":
        return out.last_hidden_state[:, 0]
    # mean pooling
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

# ── encode ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode(texts, encoder):
    vecs = []
    for i in range(0, len(texts), 64):
        ids, mask = tokenize_batch(tok, texts[i:i+64], ml)
        out = encoder(input_ids=ids.to(device), attention_mask=mask.to(device))
        v   = _pool(out, mask.to(device), pooling)
        vecs.append(F.normalize(v, dim=-1).cpu())
    return torch.cat(vecs)

print("[encode] anchors …")
a_vecs = encode(anchors, anchor_enc)
print("[encode] positives …")
p_vecs = encode(positives, positive_enc)

# ── per-query: score [positive, hn0..hn3], z-score, collect ──────────────────
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)

print("[score] 5 candidates per query …")
pos_z = []
neg_z = []
for i in range(N):
    cands = [i] + [int(j) for j in hn[i][:K]]
    sc    = np.array([float(torch.dot(a_vecs[i], p_vecs[j])) for j in cands])
    zz    = zscore(sc)
    pos_z.append(zz[0])
    neg_z.extend(zz[1:])

pos_z = np.array(pos_z)
neg_z = np.array(neg_z).reshape(N, K)

# ── metrics ───────────────────────────────────────────────────────────────────
y    = np.concatenate([np.ones(N), np.zeros(N * K)])
scrs = np.concatenate([pos_z, neg_z.ravel()])
auc  = roc_auc_score(y, scrs)
p1   = float(np.mean(pos_z > neg_z.max(axis=1)))

print()
print("=" * 58)
print(f"  BiEncoder (2×BERT)  |  AUC(hard-neg, per-query z) = {auc:.4f}")
print(f"                      |  P@1(hard-neg)              = {p1:.4f}")
print(f"  (N={N}, K={K} semantic hard negs, miner=all-MiniLM-L6-v2)")
print("=" * 58)
