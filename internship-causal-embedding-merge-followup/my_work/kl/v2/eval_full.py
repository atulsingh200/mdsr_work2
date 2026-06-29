"""Full evaluation: CDEv2 run5c vs fine-tuned BiEncoder on aep_causal test split.

Three metric families — same protocol for both models, no z-scoring anywhere:
  1. Full N×N retrieval  → MRR, Recall@K, mean/median rank
  2. AUC + P@1 vs 4 RANDOM negatives  (rng seed 0)
  3. AUC + P@1 vs 4 SEMANTIC hard negatives  (pre-mined MiniLM-L6-v2 kNN)

For AUC: raw scores go straight into roc_auc_score, identical to
evaluate_finetune_hardneg.py / evaluate_cde.py — no per-query normalization.

Run:
  cd /mnt/localssd/internship-causal-embedding-merge-followup
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python my_work/kl/v2/eval_full.py
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
REPO   = KL_DIR.parent.parent                     # repo root
sys.path.insert(0, str(KL_DIR))

from dataset import get_tokenizer, load_pairs, tokenize_batch  # noqa: E402

# ── load model_v2 from the branch if not already on disk ─────────────────────
if not (HERE / "model_v2.py").exists():
    import subprocess
    tmp = Path("/tmp/cdev2_eval_full/v2")
    tmp.mkdir(parents=True, exist_ok=True)
    for f in ["model_v2.py", "losses_v2.py"]:
        subprocess.run(f"git show feat/cde-v2:my_work/kl/v2/{f} > {tmp/f}",
                       shell=True, cwd=REPO, check=True)
    (tmp / "__init__.py").touch()
    sys.path.insert(0, str(tmp.parent))
    print("[setup] extracted model_v2.py from feat/cde-v2 into /tmp/cdev2_eval_full")

from v2.model_v2 import CDEv2  # noqa: E402

# ── data ──────────────────────────────────────────────────────────────────────
OUT_DIR  = KL_DIR / "results" / "aep_causal"
BI_CKPT  = REPO / "finetune_eval" / "results" / "aep_causal" / "checkpoint_best.pt"
CDE_CKPT = OUT_DIR / "cde_v2_run5c_best.pt"

pairs     = load_pairs(OUT_DIR / "test_pairs.jsonl")
anchors   = [a for a, _ in pairs]
positives = [p for _, p in pairs]
N         = len(pairs)
hn        = np.load(OUT_DIR / "test_hard_negatives.npy")   # (N, 4)
K         = hn.shape[1]
print(f"[data] N={N}  hard-neg K={K}  miner=all-MiniLM-L6-v2\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── metric helpers ────────────────────────────────────────────────────────────
def full_retrieval(S: torch.Tensor, k_vals=(1, 3, 5, 10)) -> dict:
    """MRR + Recall@K from (N,N) score matrix; diagonal = true positives."""
    diag  = S.diagonal().unsqueeze(1)
    ranks = (S > diag).sum(dim=1).float() + 1.0
    out   = {
        "mrr":         float((1.0 / ranks).mean()),
        "mean_rank":   float(ranks.mean()),
        "median_rank": float(ranks.median()),
        "p@1":         float((ranks <= 1).float().mean()),
    }
    for k in k_vals:
        out[f"r@{k}"] = float((ranks <= k).float().mean())
    return out


def auc_p1_random(pos_sc: np.ndarray, S: np.ndarray, n_neg=4, seed=0) -> dict:
    """AUC + P@1 vs n_neg random negatives per query. Raw scores, no z-score."""
    rng     = np.random.default_rng(seed)
    neg_idx = np.empty((N, n_neg), dtype=np.int64)
    for i in range(N):
        picks = rng.choice(N - 1, size=n_neg, replace=False)
        picks[picks >= i] += 1
        neg_idx[i] = picks
    neg_sc  = S[np.arange(N)[:, None], neg_idx]
    p1      = float(np.mean(pos_sc > neg_sc.max(axis=1)))
    scores  = np.concatenate([pos_sc, neg_sc.ravel()])
    labels  = np.concatenate([np.ones(N), np.zeros(N * n_neg)])
    return {"auc": float(roc_auc_score(labels, scores)), "p@1": p1}


def auc_p1_hardneg(pos_sc: np.ndarray, S: np.ndarray, hn_idx: np.ndarray) -> dict:
    """AUC + P@1 vs pre-mined semantic hard negatives. Raw scores, no z-score."""
    neg_sc = S[np.arange(N)[:, None], hn_idx]
    p1     = float(np.mean(pos_sc > neg_sc.max(axis=1)))
    scores = np.concatenate([pos_sc, neg_sc.ravel()])
    labels = np.concatenate([np.ones(N), np.zeros(N * hn_idx.shape[1])])
    return {"auc": float(roc_auc_score(labels, scores)), "p@1": p1}


def print_block(name, retr, rand, hard):
    w = 60
    print("=" * w)
    print(f"  {name}")
    print("-" * w)
    print(f"  Full N×N retrieval (pool={N})")
    print(f"    MRR        = {retr['mrr']:.4f}")
    print(f"    P@1        = {retr['p@1']:.4f}")
    print(f"    R@1/3/5/10 = {retr['r@1']:.3f} / {retr['r@3']:.3f} / "
          f"{retr['r@5']:.3f} / {retr['r@10']:.3f}")
    print(f"    mean_rank  = {retr['mean_rank']:.1f}   median_rank = {retr['median_rank']:.1f}")
    print(f"  AUC + P@1 vs {K} random negatives")
    print(f"    AUC = {rand['auc']:.4f}   P@1 = {rand['p@1']:.4f}")
    print(f"  AUC + P@1 vs {K} semantic hard negatives (MiniLM-L6-v2)")
    print(f"    AUC = {hard['auc']:.4f}   P@1 = {hard['p@1']:.4f}")
    print("=" * w)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fine-tuned BiEncoder (anchor_encoder + positive_encoder)
# ─────────────────────────────────────────────────────────────────────────────
print("[BiEncoder] loading …")
bi_ckpt = torch.load(BI_CKPT, map_location="cpu", weights_only=False)
bi_cfg  = bi_ckpt["cfg"]
bi_tok  = get_tokenizer(bi_cfg["backbone"])
ml      = bi_cfg.get("max_seq_length", 256)
pool    = bi_cfg.get("pooling", "cls")

def _load_bert(sd, prefix):
    enc = AutoModel.from_pretrained(bi_cfg["backbone"])
    enc.load_state_dict({k[len(prefix):]: v for k, v in sd.items()
                         if k.startswith(prefix)}, strict=True)
    return enc.to(device).eval()

bi_anc = _load_bert(bi_ckpt["model_state_dict"], "anchor_encoder.")
bi_pos = _load_bert(bi_ckpt["model_state_dict"], "positive_encoder.")
print(f"  backbone={bi_cfg['backbone']}  pooling={pool}  val_mrr={bi_ckpt['metrics']['mrr']:.4f}")

def _pool(out, mask, method):
    if method == "cls":
        return out.last_hidden_state[:, 0]
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

@torch.no_grad()
def encode_bi(texts, enc):
    vecs = []
    for i in range(0, len(texts), 64):
        ids, mask = tokenize_batch(bi_tok, texts[i:i+64], ml)
        out = enc(input_ids=ids.to(device), attention_mask=mask.to(device))
        vecs.append(F.normalize(_pool(out, mask.to(device), pool), dim=-1).cpu())
    return torch.cat(vecs)

print("  encoding …")
a_bi = encode_bi(anchors, bi_anc).to(device)
p_bi = encode_bi(positives, bi_pos).to(device)

S_bi    = a_bi @ p_bi.T                          # (N, N) cosine scores
S_bi_np = S_bi.cpu().numpy()
pos_bi  = S_bi_np[np.arange(N), np.arange(N)]

retr_bi = full_retrieval(S_bi)
rand_bi = auc_p1_random(pos_bi, S_bi_np)
hard_bi = auc_p1_hardneg(pos_bi, S_bi_np, hn)

del bi_anc, bi_pos, a_bi, p_bi, S_bi
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 2. CDEv2 run5c
# ─────────────────────────────────────────────────────────────────────────────
print("\n[CDEv2 run5c] loading …")
ckpt = torch.load(CDE_CKPT, map_location="cpu", weights_only=False)
cfg  = ckpt["cfg"]
tok  = get_tokenizer(cfg["backbone"])
ml2  = cfg.get("max_seq_length", 256)

model = CDEv2(
    cfg["backbone"], cfg["proj_dim"],
    shared_encoder=cfg.get("shared_encoder", False),
    kl_scale=cfg.get("kl_scale", "dim"),
    init_cos_weight=cfg.get("init_cos_weight", 1.0),
    pooling=cfg.get("pooling", "mean"),
    mu_identity=cfg.get("mu_identity", False),
    log_sigma_init=cfg.get("log_sigma_init", 0.0),
).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
print(f"  backbone={cfg['backbone']}  pooling={cfg['pooling']}  device={device}")

@torch.no_grad()
def encode_cde(texts, cause):
    M, L = [], []
    for i in range(0, len(texts), 64):
        ids, mask = tokenize_batch(tok, texts[i:i+64], ml2)
        ids = ids.to(device); mask = mask.to(device)
        if cause:
            mu, ls, de = model.encode_cause(ids, mask)
            mu, ls = model.transport(mu, ls, de)
        else:
            mu, ls = model.encode_effect(ids, mask)
        M.append(mu.cpu()); L.append(ls.cpu())
    return torch.cat(M).to(device), torch.cat(L).to(device)

print("  encoding …")
mu_a, ls_a = encode_cde(anchors, cause=True)
mu_p, ls_p = encode_cde(positives, cause=False)

S_cde    = model.score_matrix(mu_a, ls_a, mu_p, ls_p)   # (N, N) raw scores
S_cde_np = S_cde.detach().cpu().numpy()
pos_cde  = S_cde_np[np.arange(N), np.arange(N)]

retr_cde = full_retrieval(S_cde)
rand_cde = auc_p1_random(pos_cde, S_cde_np)
hard_cde = auc_p1_hardneg(pos_cde, S_cde_np, hn)

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
print()
print_block(f"BiEncoder (2×BERT fine-tuned)  —  {bi_cfg['backbone']}", retr_bi, rand_bi, hard_bi)
print_block(f"CDEv2 run5c  —  {cfg['backbone']}", retr_cde, rand_cde, hard_cde)

# compact comparison table
print(f"{'Metric':<28} {'BiEncoder':>10} {'CDEv2 run5c':>12}  {'Delta':>8}")
print("-" * 62)
rows = [
    ("MRR (N×N)",            retr_bi["mrr"],      retr_cde["mrr"]),
    ("P@1 (N×N)",            retr_bi["p@1"],      retr_cde["p@1"]),
    ("R@5 (N×N)",            retr_bi["r@5"],      retr_cde["r@5"]),
    ("R@10 (N×N)",           retr_bi["r@10"],     retr_cde["r@10"]),
    ("AUC vs random negs",   rand_bi["auc"],      rand_cde["auc"]),
    ("P@1 vs random negs",   rand_bi["p@1"],      rand_cde["p@1"]),
    ("AUC vs hard negs",     hard_bi["auc"],      hard_cde["auc"]),
    ("P@1 vs hard negs",     hard_bi["p@1"],      hard_cde["p@1"]),
]
for label, bi, cde in rows:
    delta = cde - bi
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<26} {bi:>10.4f} {cde:>12.4f}  {sign}{delta:>7.4f}")
