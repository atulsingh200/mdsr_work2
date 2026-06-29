"""Evaluate CDE and compare against the Two-Tower BiEncoder baseline.

Loads the CDE checkpoint from results/aep_causal/cde_best.pt and
evaluates on the test split using three metric families:

  1) Full N×N retrieval: MRR, Recall@K, median rank
  2) AUC + P@1 vs 4 random negatives (same protocol as evaluate_finetune.py)
  3) AUC + P@1 vs 4 semantic hard negatives (same as evaluate_finetune_hardneg.py)

For fair comparison the script also loads the existing BiEncoder
checkpoint (if present) and runs the same metrics using dot-product scores.

Usage:
  python evaluate_cde.py
  python evaluate_cde.py --skip-biencoder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

KL_DIR = Path(__file__).resolve().parent
BIENCODER_ROOT = KL_DIR.parent.parent / "internship-causal-embedding"
sys.path.insert(0, str(KL_DIR))
sys.path.insert(0, str(BIENCODER_ROOT))
sys.path.insert(0, str(BIENCODER_ROOT / "src"))

from model import CausalDensityEmbedding   # noqa: E402
from dataset import get_tokenizer, load_pairs, tokenize_batch  # noqa: E402

OUT_DIR = KL_DIR / "results" / "aep_causal"


# ---------------------------------------------------------------------------
# Metric helpers (self-contained so evaluate_cde.py has no extra deps)
# ---------------------------------------------------------------------------

def _kl_score_matrix(
    mu_ta: torch.Tensor,
    log_sigma_ta: torch.Tensor,
    mu_c: torch.Tensor,
    log_sigma_c: torch.Tensor,
) -> torch.Tensor:
    """Score matrix S[i,j] = Score(A_i -> B_j) = -KL(N_Bj || T(N_Ai))."""
    D = mu_ta.shape[-1]
    sigma_q_sq = (2 * log_sigma_ta).exp().clamp(min=1e-9)
    sigma_c_sq = (2 * log_sigma_c).exp()
    inv_q_sq = 1.0 / sigma_q_sq

    t1 = log_sigma_ta.sum(-1).unsqueeze(1) - log_sigma_c.sum(-1).unsqueeze(0)
    t2 = 0.5 * (inv_q_sq @ sigma_c_sq.T)
    half_inv = 0.5 * inv_q_sq
    t3 = (
        half_inv @ (mu_c.pow(2)).T
        - (mu_ta * inv_q_sq) @ mu_c.T
        + (mu_ta.pow(2) * half_inv).sum(-1, keepdim=True)
    )
    kl = t1 + t2 + t3 - 0.5 * D
    return -kl  # (N_q, N_c)


def full_retrieval_metrics(score_mat: torch.Tensor, k_values=(1, 3, 5, 10)) -> dict:
    """MRR, Recall@K, mean/median rank from an (N, N) score matrix (diag = true pos)."""
    N = score_mat.size(0)
    diag = score_mat.diagonal().unsqueeze(1)
    ranks = (score_mat > diag).sum(dim=1).float() + 1.0
    out = {
        "n_test":      N,
        "mrr":         float((1.0 / ranks).mean()),
        "mean_rank":   float(ranks.mean()),
        "median_rank": float(ranks.median()),
    }
    for k in k_values:
        out[f"recall@{k}"] = float((ranks <= k).float().mean())
    return out


def pair_metrics_random(
    query_scores: np.ndarray,   # (N,) positive scores
    cand_scores:  np.ndarray,   # (N, N) full score matrix (or None)
    n_negatives:  int = 4,
    seed: int = 0,
) -> dict:
    """AUC + P@1 vs n_negatives random negatives per query.

    Works on a (N,) positive score vector and the full (N, N) matrix.
    Negative scores for query i are the scores of i against other rows' positives.
    """
    N = len(query_scores)
    rng = np.random.default_rng(seed)
    neg_idx = np.empty((N, n_negatives), dtype=np.int64)
    for i in range(N):
        picks = rng.choice(N - 1, size=n_negatives, replace=False)
        picks[picks >= i] += 1
        neg_idx[i] = picks

    neg_scores = cand_scores[np.arange(N)[:, None], neg_idx]  # (N, n_neg)
    p_at_1 = float(np.mean(query_scores > neg_scores.max(axis=1)))
    all_scores = np.concatenate([query_scores, neg_scores.ravel()])
    all_labels = np.concatenate([np.ones(N), np.zeros(N * n_negatives)])
    auc = float(roc_auc_score(all_labels, all_scores))
    return {"auc": auc, "precision_at_1": p_at_1}


def pair_metrics_hard_neg(
    query_scores: np.ndarray,   # (N,) positive scores for query i
    cand_scores:  np.ndarray,   # (N, N) full score matrix
    hard_neg_idx: np.ndarray,   # (N, K) pre-mined hard negative indices
) -> dict:
    """AUC + P@1 vs semantic hard negatives."""
    N = len(query_scores)
    K = hard_neg_idx.shape[1]
    neg_scores = cand_scores[np.arange(N)[:, None], hard_neg_idx]   # (N, K)
    p_at_1 = float(np.mean(query_scores > neg_scores.max(axis=1)))
    all_scores = np.concatenate([query_scores, neg_scores.ravel()])
    all_labels = np.concatenate([np.ones(N), np.zeros(N * K)])
    auc = float(roc_auc_score(all_labels, all_scores))
    return {"auc": auc, "precision_at_1": p_at_1}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_cde(
    model: CausalDensityEmbedding,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    as_query: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode texts; if as_query=True apply transport (for anchors)."""
    model.eval()
    mus, log_sigmas = [], []
    for i in range(0, len(texts), batch_size):
        ids, mask = tokenize_batch(tokenizer, texts[i: i + batch_size], max_length)
        ids = ids.to(device);  mask = mask.to(device)
        mu, log_sigma, delta = model.encode(ids, mask)
        if as_query:
            mu, log_sigma = model.transport(mu, log_sigma, delta)
        mus.append(mu.cpu())
        log_sigmas.append(log_sigma.cpu())
    return torch.cat(mus), torch.cat(log_sigmas)


@torch.no_grad()
def encode_biencoder(
    model,       # BiEncoder
    tokenizer,
    texts: list[str],
    which: str,  # "anchor" | "positive"
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        emb = model.encode_anchor(ids, mask) if which == "anchor" else model.encode_positive(ids, mask)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-seq-length",  type=int, default=256)
    ap.add_argument("--batch-size",      type=int, default=128)
    ap.add_argument("--n-negatives",     type=int, default=4)
    ap.add_argument("--k",   nargs="+",  type=int, default=[1, 3, 5, 10])
    ap.add_argument("--seed",            type=int, default=0)
    ap.add_argument("--skip-biencoder",  action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = sorted(set(args.k))

    test_pairs_path = OUT_DIR / "test_pairs.jsonl"
    if not test_pairs_path.exists():
        raise SystemExit("Run prepare_data.py first to create test_pairs.jsonl")

    pairs = load_pairs(test_pairs_path)
    anchors  = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    N = len(pairs)
    print(f"[test]  {N} pairs")

    test_hn_path = OUT_DIR / "test_hard_negatives.npy"
    hard_neg_idx = np.load(test_hn_path) if test_hn_path.exists() else None
    if hard_neg_idx is not None:
        print(f"[test]  loaded hard negatives ({hard_neg_idx.shape})")

    all_results: dict[str, dict] = {}

    # ── CDE evaluation ─────────────────────────────────────────────────────
    cde_ckpt_path = OUT_DIR / "cde_best.pt"
    if not cde_ckpt_path.exists():
        print(f"[cde]  checkpoint not found at {cde_ckpt_path}")
    else:
        print(f"\n[CDE]  loading {cde_ckpt_path}")
        ckpt = torch.load(cde_ckpt_path, map_location="cpu", weights_only=False)
        cfg  = ckpt["cfg"]
        backbone  = cfg["backbone"]
        proj_dim  = cfg["proj_dim"]
        max_len   = cfg.get("max_seq_length", args.max_seq_length)

        tokenizer = get_tokenizer(backbone)
        model = CausalDensityEmbedding(backbone, proj_dim).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  backbone={backbone}  proj_dim={proj_dim}  device={device}")

        t0 = time.time()
        mu_ta, log_sigma_ta = encode_cde(model, tokenizer, anchors,
                                          max_len, args.batch_size, device, as_query=True)
        mu_c,  log_sigma_c  = encode_cde(model, tokenizer, positives,
                                          max_len, args.batch_size, device, as_query=False)
        enc_time = time.time() - t0
        print(f"  encoded {N * 2} texts in {enc_time:.1f}s")

        mu_ta = mu_ta.to(device);  log_sigma_ta = log_sigma_ta.to(device)
        mu_c  = mu_c.to(device);   log_sigma_c  = log_sigma_c.to(device)

        S = _kl_score_matrix(mu_ta, log_sigma_ta, mu_c, log_sigma_c)  # (N, N)
        retr = full_retrieval_metrics(S, k_values)

        S_np = S.cpu().numpy()
        pos_scores = S_np[np.arange(N), np.arange(N)]  # diagonal

        rand_m = pair_metrics_random(pos_scores, S_np, args.n_negatives, args.seed)
        hn_m   = (pair_metrics_hard_neg(pos_scores, S_np, hard_neg_idx)
                  if hard_neg_idx is not None else None)

        print(f"  AUC={rand_m['auc']:.4f}  P@1(rand)={rand_m['precision_at_1']:.4f}")
        if hn_m:
            print(f"  AUC(HN)={hn_m['auc']:.4f}  P@1(HN)={hn_m['precision_at_1']:.4f}")
        print(f"  MRR={retr['mrr']:.4f}  mean_rank={retr['mean_rank']:.2f}  "
              f"median_rank={retr['median_rank']:.1f}")
        for k in k_values:
            print(f"  Recall@{k:<3}{retr[f'recall@{k}']:.4f}")

        all_results["cde"] = {
            "model": "CDE",
            "backbone": backbone,
            "n_test_pairs": N,
            "auc_random": rand_m["auc"],
            "p1_random":  rand_m["precision_at_1"],
            "auc_hard_neg": hn_m["auc"] if hn_m else None,
            "p1_hard_neg":  hn_m["precision_at_1"] if hn_m else None,
            **retr,
        }

    # ── BiEncoder comparison ────────────────────────────────────────────────
    bi_ckpt_path = BIENCODER_ROOT / "finetune_eval" / "results" / "aep_causal" / "checkpoint_best.pt"
    if not args.skip_biencoder and bi_ckpt_path.exists():
        try:
            from biencoder.model import BiEncoder, get_tokenizer as bi_get_tok  # noqa: E402
            print(f"\n[BiEncoder]  loading {bi_ckpt_path}")
            bi_ckpt = torch.load(bi_ckpt_path, map_location="cpu", weights_only=False)
            bi_cfg  = bi_ckpt["cfg"]
            bi_tok  = bi_get_tok(bi_cfg["backbone"])
            bi_model = BiEncoder(
                model_name=bi_cfg["backbone"],
                pooling_strategy=bi_cfg["pooling"],
                anchor_prefix="", positive_prefix="",
            ).to(device)
            bi_model.load_state_dict(bi_ckpt["model_state_dict"])
            bi_model.eval()
            print(f"  backbone={bi_cfg['backbone']}  pooling={bi_cfg['pooling']}")

            bi_max_len = bi_cfg.get("max_seq_length", args.max_seq_length)
            t0 = time.time()
            a_emb = encode_biencoder(bi_model, bi_tok, anchors,  "anchor",   bi_max_len, args.batch_size, device)
            p_emb = encode_biencoder(bi_model, bi_tok, positives, "positive", bi_max_len, args.batch_size, device)
            enc_time = time.time() - t0
            print(f"  encoded {N * 2} texts in {enc_time:.1f}s")

            # BiEncoder dot-product score matrix
            a_t = torch.from_numpy(a_emb).to(device)
            p_t = torch.from_numpy(p_emb).to(device)
            S_bi = (a_t @ p_t.T)  # (N, N)
            retr_bi = full_retrieval_metrics(S_bi, k_values)

            S_bi_np = S_bi.cpu().numpy()
            pos_bi  = S_bi_np[np.arange(N), np.arange(N)]

            rand_bi = pair_metrics_random(pos_bi, S_bi_np, args.n_negatives, args.seed)
            hn_bi   = (pair_metrics_hard_neg(pos_bi, S_bi_np, hard_neg_idx)
                       if hard_neg_idx is not None else None)

            print(f"  AUC={rand_bi['auc']:.4f}  P@1(rand)={rand_bi['precision_at_1']:.4f}")
            if hn_bi:
                print(f"  AUC(HN)={hn_bi['auc']:.4f}  P@1(HN)={hn_bi['precision_at_1']:.4f}")
            print(f"  MRR={retr_bi['mrr']:.4f}  mean_rank={retr_bi['mean_rank']:.2f}  "
                  f"median_rank={retr_bi['median_rank']:.1f}")
            for k in k_values:
                print(f"  Recall@{k:<3}{retr_bi[f'recall@{k}']:.4f}")

            all_results["biencoder"] = {
                "model": "BiEncoder (two-tower)",
                "backbone": bi_cfg["backbone"],
                "n_test_pairs": N,
                "auc_random": rand_bi["auc"],
                "p1_random":  rand_bi["precision_at_1"],
                "auc_hard_neg": hn_bi["auc"] if hn_bi else None,
                "p1_hard_neg":  hn_bi["precision_at_1"] if hn_bi else None,
                **retr_bi,
            }
        except Exception as e:
            print(f"  [BiEncoder eval failed: {e}]")

    # ── Summary table ──────────────────────────────────────────────────────
    out_path = OUT_DIR / "cde_eval.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[saved]  {out_path}")

    print()
    print("=" * 100)
    hdr = f"{'model':<24} {'N':>5} {'AUC':>7} {'P@1':>7} {'AUC_HN':>8} {'P@1_HN':>8} "
    hdr += "  " + "  ".join(f"R@{k:<2}" for k in k_values)
    hdr += f"  {'MRR':>7}  {'median':>7}"
    print(hdr)
    print("-" * 100)
    for name, r in all_results.items():
        if "mrr" not in r:
            print(f"{r['model']:<24}  ERROR")
            continue
        row = f"{r['model']:<24} {r['n_test_pairs']:>5d}"
        row += f" {r['auc_random']:>7.4f} {r['p1_random']:>7.4f}"
        row += f" {(r['auc_hard_neg'] or 0.0):>8.4f} {(r['p1_hard_neg'] or 0.0):>8.4f}"
        row += "  " + "  ".join(f"{r[f'recall@{k}']:>5.3f}" for k in k_values)
        row += f"  {r['mrr']:>7.4f}  {r['median_rank']:>7.1f}"
        print(row)


if __name__ == "__main__":
    main()
