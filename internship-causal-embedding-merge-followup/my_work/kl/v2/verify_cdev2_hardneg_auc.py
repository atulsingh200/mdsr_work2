"""Verify CDEv2 AUC vs 4 semantic hard negatives (per-query z-score method).

Reproduces the 0.8184 number using CDEv2 run5c ONLY — no BiEncoder, no CE.
For each query: score [positive, hard_neg_0..3] with CDEv2, z-score those 5,
then compute AUC + P@1.

Run from the repo root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python my_work/kl/v2/verify_cdev2_hardneg_auc.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# ── path setup ────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent          # my_work/kl/v2
KL_DIR   = HERE.parent                              # my_work/kl
REPO     = KL_DIR.parent.parent                     # repo root
sys.path.insert(0, str(KL_DIR))                     # for dataset.py
sys.path.insert(0, "/tmp/cdev2_verify")             # fallback if model_v2 not on disk

# model_v2.py lives on feat/cde-v2; if not in working tree, extract it first:
if not (HERE / "model_v2.py").exists():
    import subprocess, textwrap
    Path("/tmp/cdev2_verify/v2").mkdir(parents=True, exist_ok=True)
    for fname in ["model_v2.py", "losses_v2.py"]:
        subprocess.run(
            f"git show feat/cde-v2:my_work/kl/v2/{fname} > /tmp/cdev2_verify/v2/{fname}",
            shell=True, cwd=REPO, check=True,
        )
    subprocess.run(
        "git show feat/cde-v2:my_work/kl/dataset.py > /tmp/cdev2_verify/dataset.py",
        shell=True, cwd=REPO, check=True,
    )
    Path("/tmp/cdev2_verify/v2/__init__.py").touch()
    sys.path.insert(0, "/tmp/cdev2_verify")
    print("[setup] extracted model_v2.py from feat/cde-v2 into /tmp/cdev2_verify")

from v2.model_v2 import CDEv2                           # noqa: E402
from dataset import get_tokenizer, load_pairs, tokenize_batch  # noqa: E402

# ── data ──────────────────────────────────────────────────────────────────────
OUT_DIR = KL_DIR / "results" / "aep_causal"
pairs     = load_pairs(OUT_DIR / "test_pairs.jsonl")
anchors   = [a for a, _ in pairs]
positives = [p for _, p in pairs]
N         = len(pairs)
hn        = np.load(OUT_DIR / "test_hard_negatives.npy")   # (N,4) MiniLM-L6-v2 semantic hard negs
K         = hn.shape[1]
print(f"[data] N={N}  hard-neg K={K}  miner=all-MiniLM-L6-v2")

# ── load CDEv2 run5c ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(OUT_DIR / "cde_v2_run5c_best.pt", map_location="cpu", weights_only=False)
cfg    = ckpt["cfg"]
tok    = get_tokenizer(cfg["backbone"])
model  = CDEv2(
    cfg["backbone"], cfg["proj_dim"],
    shared_encoder=cfg.get("shared_encoder", False),
    kl_scale=cfg.get("kl_scale", "dim"),
    init_cos_weight=cfg.get("init_cos_weight", 1.0),
    pooling=cfg.get("pooling", "mean"),
    mu_identity=cfg.get("mu_identity", False),
    log_sigma_init=cfg.get("log_sigma_init", 0.0),
).to(device)
# strict=False: run5c predates the MMD params (log_gamma_k/gate_w/gate_b); KL path never uses them
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
ml = cfg.get("max_seq_length", 256)
print(f"[model] CDEv2 run5c  backbone={cfg['backbone']}  pooling={cfg['pooling']}  device={device}")

# ── encode all anchors (cause-side, transported) and positives (effect-side) ──
@torch.no_grad()
def encode(texts, cause):
    M, L = [], []
    for i in range(0, len(texts), 64):
        ids, mask = tokenize_batch(tok, texts[i:i+64], ml)
        ids = ids.to(device); mask = mask.to(device)
        if cause:
            mu, ls, de = model.encode_cause(ids, mask)
            mu, ls = model.transport(mu, ls, de)
        else:
            mu, ls = model.encode_effect(ids, mask)
        M.append(mu.cpu()); L.append(ls.cpu())
    return torch.cat(M), torch.cat(L)

print("[encode] anchors (cause+transport) …")
a_mu, a_ls = encode(anchors, cause=True)
print("[encode] positives (effect) …")
p_mu, p_ls = encode(positives, cause=False)

# ── pointwise CDEv2 score: -KL(N_b || N_a_transported)/D + cosine ─────────────
def score(i, j):
    ma = a_mu[i]; lsa = a_ls[i]
    mb = p_mu[j]; lsb = p_ls[j]
    sa2  = lsa.exp().pow(2)
    sb2  = lsb.exp().pow(2)
    diff = (mb - ma).pow(2)
    kl   = 0.5 * (sb2/sa2 + diff/sa2 - 1.0 - (lsb - lsa)*2).sum()
    cos  = torch.nn.functional.cosine_similarity(ma.unsqueeze(0), mb.unsqueeze(0)).squeeze()
    return float(-kl / ma.shape[0] + cos)

# ── per-query: score [positive, hn0, hn1, hn2, hn3], z-score, collect ─────────
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)

print("[score] 5 candidates per query (positive + 4 hard negs) …")
pos_z, neg_z = [], []
for i in range(N):
    cands = [i] + [int(j) for j in hn[i][:K]]
    sc    = np.array([score(i, j) for j in cands])
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
print("=" * 54)
print(f"  CDEv2 run5c  |  AUC(hard-neg, per-query z) = {auc:.4f}")
print(f"                |  P@1(hard-neg)              = {p1:.4f}")
print(f"  (N={N}, K={K} semantic hard negs, miner=all-MiniLM-L6-v2)")
print("=" * 54)
