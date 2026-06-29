"""Evaluate CDEv2 and compare against BiEncoder baseline.

Loads cde_v2_best.pt and evaluates on the test split using:
  1) Full N×N retrieval: MRR, Recall@K, median rank
  2) AUC + P@1 vs 4 random negatives
  3) AUC + P@1 vs semantic hard negatives

Also loads the BiEncoder checkpoint for a direct side-by-side comparison
using the same evaluation protocol.

Usage:
  python v2/evaluate_v2.py
  python v2/evaluate_v2.py --skip-biencoder
  python v2/evaluate_v2.py --also-v1   # also evaluate CDE v1 for 3-way comparison
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

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
# Project root holds src/{biencoder,evaluation,followup_data} and finetune_eval.
PROJECT_ROOT = KL_DIR.parent.parent
_SIBLING = PROJECT_ROOT.parent / "internship-causal-embedding"
BIENCODER_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "src" / "biencoder").exists() else _SIBLING
sys.path.insert(0, str(KL_DIR))
sys.path.insert(0, str(BIENCODER_ROOT))
sys.path.insert(0, str(BIENCODER_ROOT / "src"))

from dataset import get_tokenizer, load_pairs, tokenize_batch   # noqa: E402
from v2.model_v2 import CDEv2                                   # noqa: E402

OUT_DIR = KL_DIR / "results" / "aep_causal"


# ── Metric helpers ────────────────────────────────────────────────────────────

def full_retrieval_metrics(score_mat: torch.Tensor, k_values=(1, 3, 5, 10)) -> dict:
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
    pos_scores: np.ndarray,
    score_mat:  np.ndarray,
    n_neg: int = 4,
    seed: int = 0,
) -> dict:
    N = len(pos_scores)
    rng = np.random.default_rng(seed)
    neg_idx = np.empty((N, n_neg), dtype=np.int64)
    for i in range(N):
        picks = rng.choice(N - 1, size=n_neg, replace=False)
        picks[picks >= i] += 1
        neg_idx[i] = picks
    neg_scores = score_mat[np.arange(N)[:, None], neg_idx]
    p1 = float(np.mean(pos_scores > neg_scores.max(axis=1)))
    all_s = np.concatenate([pos_scores, neg_scores.ravel()])
    all_l = np.concatenate([np.ones(N), np.zeros(N * n_neg)])
    return {"auc": float(roc_auc_score(all_l, all_s)), "precision_at_1": p1}


def pair_metrics_hard_neg(
    pos_scores:   np.ndarray,
    score_mat:    np.ndarray,
    hard_neg_idx: np.ndarray,
) -> dict:
    N = len(pos_scores)
    K = hard_neg_idx.shape[1]
    neg_s = score_mat[np.arange(N)[:, None], hard_neg_idx]
    p1 = float(np.mean(pos_scores > neg_s.max(axis=1)))
    all_s = np.concatenate([pos_scores, neg_s.ravel()])
    all_l = np.concatenate([np.ones(N), np.zeros(N * K)])
    return {"auc": float(roc_auc_score(all_l, all_s)), "precision_at_1": p1}


# ── Encoding helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def encode_v2(
    model: CDEv2,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    as_cause: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mus, log_sigmas = [], []
    for i in range(0, len(texts), batch_size):
        ids, mask = tokenize_batch(tokenizer, texts[i: i + batch_size], max_length)
        ids = ids.to(device); mask = mask.to(device)
        if as_cause:
            mu, log_sigma, delta = model.encode_cause(ids, mask)
            mu, log_sigma = model.transport(mu, log_sigma, delta)
        else:
            mu, log_sigma = model.encode_effect(ids, mask)
        mus.append(mu.cpu())
        log_sigmas.append(log_sigma.cpu())
    return torch.cat(mus), torch.cat(log_sigmas)


@torch.no_grad()
def encode_biencoder(model, tokenizer, texts, which, max_length, batch_size, device):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--batch-size",     type=int, default=128)
    ap.add_argument("--n-negatives",    type=int, default=4)
    ap.add_argument("--k", nargs="+",   type=int, default=[1, 3, 5, 10])
    ap.add_argument("--seed",           type=int, default=0)
    ap.add_argument("--skip-biencoder", action="store_true")
    ap.add_argument("--also-v1",        action="store_true",
                    help="Also evaluate CDE v1 for 3-way comparison.")
    ap.add_argument("--ckpt-suffix",    default="",
                    help="Suffix of the CDEv2 checkpoint to load, e.g. '_run4b' "
                         "loads cde_v2_run4b_best.pt.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = sorted(set(args.k))

    test_pairs_path = OUT_DIR / "test_pairs.jsonl"
    if not test_pairs_path.exists():
        raise SystemExit("Run prepare_data.py first.")

    pairs    = load_pairs(test_pairs_path)
    anchors  = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    N = len(pairs)
    print(f"[test]  {N} pairs  device={device}")

    hard_neg_idx = None
    for hn_path in [OUT_DIR / "test_hard_negatives.npy"]:
        if hn_path.exists():
            hard_neg_idx = np.load(hn_path)
            print(f"[test]  loaded hard negatives {hard_neg_idx.shape}")

    all_results: dict[str, dict] = {}

    # ── CDEv2 ────────────────────────────────────────────────────────────────
    v2_ckpt = OUT_DIR / f"cde_v2{args.ckpt_suffix}_best.pt"
    if not v2_ckpt.exists():
        print(f"[CDEv2]  checkpoint not found at {v2_ckpt}")
    else:
        print(f"\n[CDEv2]  loading {v2_ckpt}")
        ckpt = torch.load(v2_ckpt, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        backbone = cfg["backbone"]
        proj_dim = cfg["proj_dim"]
        max_len  = cfg.get("max_seq_length", args.max_seq_length)

        tokenizer = get_tokenizer(backbone)
        model = CDEv2(
            backbone, proj_dim,
            shared_encoder=cfg.get("shared_encoder", False),
            kl_scale=cfg.get("kl_scale", "dim"),
            init_cos_weight=cfg.get("init_cos_weight", 1.0),
            pooling=cfg.get("pooling", "mean"),
            mu_identity=cfg.get("mu_identity", False),
            log_sigma_init=cfg.get("log_sigma_init", 0.0),
            score_type=cfg.get("score_type", "kl"),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  backbone={backbone}  proj_dim={proj_dim}  "
              f"cos_weight={model.cos_weight.item():.4f}")

        t0 = time.time()
        mu_ta, lsig_ta = encode_v2(model, tokenizer, anchors,
                                    max_len, args.batch_size, device, as_cause=True)
        mu_b,  lsig_b  = encode_v2(model, tokenizer, positives,
                                    max_len, args.batch_size, device, as_cause=False)
        print(f"  encoded {N * 2} texts in {time.time()-t0:.1f}s")

        mu_ta  = mu_ta.to(device); lsig_ta = lsig_ta.to(device)
        mu_b   = mu_b.to(device);  lsig_b  = lsig_b.to(device)
        with torch.no_grad():
            S  = model.score_matrix(mu_ta, lsig_ta, mu_b, lsig_b)
        retr   = full_retrieval_metrics(S, k_values)
        S_np   = S.cpu().numpy()
        pos_s  = S_np[np.arange(N), np.arange(N)]
        rand_m = pair_metrics_random(pos_s, S_np, args.n_negatives, args.seed)
        hn_m   = pair_metrics_hard_neg(pos_s, S_np, hard_neg_idx) if hard_neg_idx is not None else None

        print(f"  AUC={rand_m['auc']:.4f}  P@1={rand_m['precision_at_1']:.4f}")
        if hn_m:
            print(f"  AUC_HN={hn_m['auc']:.4f}  P@1_HN={hn_m['precision_at_1']:.4f}")
        print(f"  MRR={retr['mrr']:.4f}  mean_rank={retr['mean_rank']:.2f}  "
              f"median_rank={retr['median_rank']:.1f}")
        for k in k_values:
            print(f"  Recall@{k:<3}{retr[f'recall@{k}']:.4f}")

        all_results["cde_v2"] = {
            "model": "CDEv2",
            "backbone": backbone,
            "n_test_pairs": N,
            "auc_random":   rand_m["auc"],
            "p1_random":    rand_m["precision_at_1"],
            "auc_hard_neg": hn_m["auc"] if hn_m else None,
            "p1_hard_neg":  hn_m["precision_at_1"] if hn_m else None,
            **retr,
        }

    # ── CDE v1 (optional) ────────────────────────────────────────────────────
    if args.also_v1:
        from model import CausalDensityEmbedding   # noqa: E402 (v1 model)

        def _kl_score_mat(mu_ta, lsig_ta, mu_c, lsig_c):
            D = mu_ta.shape[-1]
            sq = (2 * lsig_ta).exp().clamp(min=1e-9)
            sc = (2 * lsig_c).exp()
            inv = 1.0 / sq
            t1 = lsig_ta.sum(-1).unsqueeze(1) - lsig_c.sum(-1).unsqueeze(0)
            t2 = 0.5 * (inv @ sc.T)
            h  = 0.5 * inv
            t3 = h @ (mu_c.pow(2)).T - (mu_ta * inv) @ mu_c.T + (mu_ta.pow(2) * h).sum(-1, keepdim=True)
            return -(t1 + t2 + t3 - 0.5 * D)

        v1_ckpt = OUT_DIR / "cde_best.pt"
        if v1_ckpt.exists():
            print(f"\n[CDEv1]  loading {v1_ckpt}")
            ck1  = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
            c1   = ck1["cfg"]
            tok1 = get_tokenizer(c1["backbone"])
            m1   = CausalDensityEmbedding(c1["backbone"], c1["proj_dim"]).to(device)
            m1.load_state_dict(ck1["model_state_dict"])
            m1.eval()

            @torch.no_grad()
            def enc1(texts, as_query):
                ms, ls = [], []
                ml = c1.get("max_seq_length", args.max_seq_length)
                for i in range(0, len(texts), args.batch_size):
                    ids, msk = tokenize_batch(tok1, texts[i:i+args.batch_size], ml)
                    ids = ids.to(device); msk = msk.to(device)
                    mu, ls_, de = m1.encode(ids, msk)
                    if as_query:
                        mu, ls_ = m1.transport(mu, ls_, de)
                    ms.append(mu.cpu()); ls.append(ls_.cpu())
                return torch.cat(ms).to(device), torch.cat(ls).to(device)

            mta1, lta1 = enc1(anchors, True)
            mb1, lb1   = enc1(positives, False)
            S1 = _kl_score_mat(mta1, lta1, mb1, lb1)
            r1 = full_retrieval_metrics(S1, k_values)
            S1np = S1.cpu().numpy()
            ps1  = S1np[np.arange(N), np.arange(N)]
            rm1  = pair_metrics_random(ps1, S1np, args.n_negatives, args.seed)
            hn1  = pair_metrics_hard_neg(ps1, S1np, hard_neg_idx) if hard_neg_idx is not None else None
            print(f"  v1 MRR={r1['mrr']:.4f}  AUC={rm1['auc']:.4f}")
            all_results["cde_v1"] = {
                "model": "CDEv1",
                "backbone": c1["backbone"],
                "n_test_pairs": N,
                "auc_random": rm1["auc"],
                "p1_random":  rm1["precision_at_1"],
                "auc_hard_neg": hn1["auc"] if hn1 else None,
                "p1_hard_neg":  hn1["precision_at_1"] if hn1 else None,
                **r1,
            }

    # ── BiEncoder ────────────────────────────────────────────────────────────
    bi_ckpt = BIENCODER_ROOT / "finetune_eval" / "results" / "aep_causal" / "checkpoint_best.pt"
    if not args.skip_biencoder and bi_ckpt.exists():
        try:
            from biencoder.model import BiEncoder, get_tokenizer as bi_tok  # noqa: E402
            print(f"\n[BiEncoder]  loading {bi_ckpt}")
            bi_c   = torch.load(bi_ckpt, map_location="cpu", weights_only=False)
            bi_cfg = bi_c["cfg"]
            bi_t   = bi_tok(bi_cfg["backbone"])
            bi_m   = BiEncoder(
                model_name=bi_cfg["backbone"],
                pooling_strategy=bi_cfg["pooling"],
                anchor_prefix="", positive_prefix="",
            ).to(device)
            bi_m.load_state_dict(bi_c["model_state_dict"])
            bi_m.eval()
            bi_ml = bi_cfg.get("max_seq_length", args.max_seq_length)
            t0 = time.time()
            a_emb = encode_biencoder(bi_m, bi_t, anchors,   "anchor",   bi_ml, args.batch_size, device)
            p_emb = encode_biencoder(bi_m, bi_t, positives, "positive", bi_ml, args.batch_size, device)
            print(f"  encoded in {time.time()-t0:.1f}s")
            a_t = torch.from_numpy(a_emb).to(device)
            p_t = torch.from_numpy(p_emb).to(device)
            S_bi = a_t @ p_t.T
            r_bi = full_retrieval_metrics(S_bi, k_values)
            Sbn  = S_bi.cpu().numpy()
            psb  = Sbn[np.arange(N), np.arange(N)]
            rmb  = pair_metrics_random(psb, Sbn, args.n_negatives, args.seed)
            hnb  = pair_metrics_hard_neg(psb, Sbn, hard_neg_idx) if hard_neg_idx is not None else None
            print(f"  BiEncoder MRR={r_bi['mrr']:.4f}  AUC={rmb['auc']:.4f}")
            all_results["biencoder"] = {
                "model": "BiEncoder",
                "backbone": bi_cfg["backbone"],
                "n_test_pairs": N,
                "auc_random": rmb["auc"],
                "p1_random":  rmb["precision_at_1"],
                "auc_hard_neg": hnb["auc"] if hnb else None,
                "p1_hard_neg":  hnb["precision_at_1"] if hnb else None,
                **r_bi,
            }
        except Exception as e:
            print(f"  [BiEncoder failed: {e}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "cde_v2_eval.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[saved]  {out_path}")

    print()
    print("=" * 110)
    hdr = f"{'model':<22} {'AUC':>7} {'P@1':>7} {'AUC_HN':>8} {'P@1_HN':>8}"
    hdr += "  " + "  ".join(f"R@{k:<2}" for k in k_values)
    hdr += f"  {'MRR':>7}  {'median':>7}"
    print(hdr)
    print("-" * 110)
    for name, r in all_results.items():
        if "mrr" not in r:
            continue
        row  = f"{r['model']:<22} {r['auc_random']:>7.4f} {r['p1_random']:>7.4f}"
        row += f" {(r['auc_hard_neg'] or 0.0):>8.4f} {(r['p1_hard_neg'] or 0.0):>8.4f}"
        row += "  " + "  ".join(f"{r[f'recall@{k}']:>5.3f}" for k in k_values)
        row += f"  {r['mrr']:>7.4f}  {r['median_rank']:>7.1f}"
        print(row)

    # Print improvement vs BiEncoder if both present.
    if "cde_v2" in all_results and "biencoder" in all_results:
        v2 = all_results["cde_v2"]
        bi = all_results["biencoder"]
        mrr_lift = (v2["mrr"] - bi["mrr"]) / bi["mrr"] * 100
        auc_lift = (v2["auc_random"] - bi["auc_random"]) / bi["auc_random"] * 100
        r1_lift  = (v2["recall@1"] - bi["recall@1"]) / max(bi["recall@1"], 1e-9) * 100
        print()
        print(f"  CDEv2 vs BiEncoder:  MRR {mrr_lift:+.1f}%  AUC {auc_lift:+.1f}%  R@1 {r1_lift:+.1f}%")
        target = bi["mrr"] * 1.10
        print(f"  10% target MRR: {target:.4f}  achieved: {'YES' if v2['mrr'] >= target else 'NO'}")


if __name__ == "__main__":
    main()
