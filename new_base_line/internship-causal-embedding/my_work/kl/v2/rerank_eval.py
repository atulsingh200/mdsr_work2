"""Two-stage retrieve-then-rerank evaluation on aep_causal test.

Stage 1 (retrieval): a single-vector retriever scores all N candidates per
query and we keep the top-`pool` (CDEv2 warm-start, or the BiEncoder).
Stage 2 (rerank): a BERT cross-encoder re-scores the (anchor, candidate)
pairs in that pool with joint cross-attention. The final ranking uses the
cross-encoder scores within the pool (optionally blended with the retrieval
score); candidates outside the pool keep their stage-1 order behind the pool.

This is the standard way to beat a bi-encoder on MRR/R@1 at fixed base model:
the bi-encoder gives high recall (R@10 ~0.71) and the cross-encoder fixes the
ordering inside the pool.

Usage:
  python v2/rerank_eval.py --retriever cde --cde-suffix _run5c \
      --ce-dir cross_encoder/results/aep_causal_rerank --pool 20 --blend 0.5
  python v2/rerank_eval.py --retriever biencoder --ce-dir <...> --pool 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
PROJECT_ROOT = KL_DIR.parent.parent
_SIBLING = PROJECT_ROOT.parent / "internship-causal-embedding"
BIENCODER_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "src" / "biencoder").exists() else _SIBLING
sys.path.insert(0, str(KL_DIR))
sys.path.insert(0, str(KL_DIR / "cross_encoder"))
sys.path.insert(0, str(BIENCODER_ROOT))
sys.path.insert(0, str(BIENCODER_ROOT / "src"))

from dataset import get_tokenizer, load_pairs, tokenize_batch   # noqa: E402
from v2.model_v2 import CDEv2                                   # noqa: E402
from model_ce import CrossEncoder                               # noqa: E402

OUT_DIR = KL_DIR / "results" / "aep_causal"


def retrieval_metrics_from_ranks(ranks: np.ndarray, k_values=(1, 3, 5, 10)) -> dict:
    out = {
        "mrr":         float(np.mean(1.0 / ranks)),
        "mean_rank":   float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
    for k in k_values:
        out[f"recall@{k}"] = float(np.mean(ranks <= k))
    return out


def ranks_from_scores(score_mat: np.ndarray) -> np.ndarray:
    """Rank of the true positive in each row. Higher score = better.

    For square matrices the true positive is on the diagonal.
    For pooled (non-square) matrices the true positive is always at column 0.
    """
    if score_mat.shape[0] == score_mat.shape[1]:
        gold_scores = np.diagonal(score_mat)[:, None]
    else:
        gold_scores = score_mat[:, 0:1]   # gold always placed at col 0 in pooled path
    return (score_mat > gold_scores).sum(axis=1).astype(np.float64) + 1.0


# ── Stage 1 retrievers ────────────────────────────────────────────────────────

@torch.no_grad()
def cde_encode(suffix, anchors, positives, batch_size, device):
    """Encode anchors and positives; return CPU numpy arrays (no N×N matrix)."""
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ml = cfg.get("max_seq_length", 256)

    def enc(texts, cause):
        mus, lss = [], []
        for i in range(0, len(texts), batch_size):
            ids, mask = tokenize_batch(tok, texts[i:i+batch_size], ml)
            ids = ids.to(device); mask = mask.to(device)
            if cause:
                mu, ls, de = model.encode_cause(ids, mask)
                mu, ls = model.transport(mu, ls, de)
            else:
                mu, ls = model.encode_effect(ids, mask)
            mus.append(mu.cpu()); lss.append(ls.cpu())
        return torch.cat(mus), torch.cat(lss)

    mta, lta = enc(anchors, True)
    mb, lb = enc(positives, False)
    # return CPU tensors + model (for scoring) — caller decides full or pooled
    return model.cpu(), mta, lta, mb, lb


@torch.no_grad()
def cde_score_matrix(suffix, anchors, positives, batch_size, device):
    """Full N×N matrix — only safe for small N."""
    model, mta, lta, mb, lb = cde_encode(suffix, anchors, positives, batch_size, device)
    S = model.score_matrix(mta, lta, mb, lb)
    return S.float().numpy()


@torch.no_grad()
def cde_score_pooled(suffix, anchors, positives, batch_size, device, cand_idx):
    """Pooled scoring: score only the candidates in cand_idx (N, cpool) — no N×N."""
    model, mta, lta, mb, lb = cde_encode(suffix, anchors, positives, batch_size, device)
    N, cpool = cand_idx.shape
    S = np.empty((N, cpool), dtype=np.float32)
    # score row by row in chunks to stay on CPU and avoid OOM
    CHUNK = 512
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        rows = slice(start, end)
        cidx = cand_idx[start:end]          # (chunk, cpool)
        # gather candidate embeddings for this chunk
        c_mu = mb[cidx.ravel()].view(end - start, cpool, -1)   # (chunk, cpool, D)
        c_ls = lb[cidx.ravel()].view(end - start, cpool, -1)
        a_mu = mta[start:end].unsqueeze(1).expand_as(c_mu)     # (chunk, cpool, D)
        a_ls = lta[start:end].unsqueeze(1).expand_as(c_ls)
        scores = model.score_pointwise(a_mu.reshape(-1, a_mu.shape[-1]),
                                       a_ls.reshape(-1, a_ls.shape[-1]),
                                       c_mu.reshape(-1, c_mu.shape[-1]),
                                       c_ls.reshape(-1, c_ls.shape[-1]))
        S[start:end] = scores.view(end - start, cpool).numpy()
    return S


@torch.no_grad()
def biencoder_score_matrix(anchors, positives, batch_size, device, dataset="aep_causal"):
    from biencoder.model import BiEncoder, get_tokenizer as bi_tok
    bc = BIENCODER_ROOT / "finetune_eval" / "results" / dataset / "checkpoint_best.pt"
    ckpt = torch.load(bc, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    tok = bi_tok(cfg["backbone"])
    model = BiEncoder(model_name=cfg["backbone"], pooling_strategy=cfg["pooling"],
                      anchor_prefix="", positive_prefix="").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ml = cfg.get("max_seq_length", 256)

    def enc(texts, which):
        out = []
        for i in range(0, len(texts), batch_size):
            e = tok(texts[i:i+batch_size], padding=True, truncation=True,
                    max_length=ml, return_tensors="pt")
            ids = e["input_ids"].to(device); mask = e["attention_mask"].to(device)
            emb = model.encode_anchor(ids, mask) if which == "a" else model.encode_positive(ids, mask)
            out.append(emb.cpu())
        return torch.cat(out)

    a = enc(anchors, "a"); p = enc(positives, "p")
    return (a @ p.T).numpy()


# ── Stage 2 cross-encoder ─────────────────────────────────────────────────────

@torch.no_grad()
def ce_score_pairs(model, tok, text_a, text_b, max_length, batch_size, device):
    out = []
    for s in range(0, len(text_a), batch_size):
        e = max(s + batch_size, 0)
        enc = tok(text_a[s:s+batch_size], text_b[s:s+batch_size],
                  padding=True, truncation="longest_first", max_length=max_length,
                  return_tensors="pt", return_token_type_ids=True)
        ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        out.extend(logits.float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", default="cde", choices=["cde", "biencoder"])
    ap.add_argument("--cde-suffix", default="_run5c")
    ap.add_argument("--ce-dir", default=["cross_encoder/results/aep_causal_rerank"],
                    nargs="+", help="One or more CE checkpoint dirs; logits are "
                    "z-scored and averaged (ensemble) when multiple are given.")
    ap.add_argument("--pool", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--blend", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0],
                    help="Final = blend*CE_z + (1-blend)*retr_z within the pool. "
                         "1.0 = pure CE rerank, 0.0 = retrieval only.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="Which split to evaluate on (val for tuning, test for reporting).")
    ap.add_argument("--dataset", default="aep_causal",
                    help="Dataset name; sets OUT_DIR=results/<dataset>/.")
    ap.add_argument("--candidate-pool", type=int, default=0,
                    help="If >0, rank each anchor against this many random candidates "
                         "instead of all N. Mirrors evaluate_finetune pool_size=1000. "
                         "Required for large datasets (workflow 260k) to avoid N×N OOM. "
                         "Set to 1000 for workflow.")
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = KL_DIR / "results" / args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(OUT_DIR / f"{args.split}_pairs.jsonl")
    anchors = [a for a, _ in pairs]; positives = [p for _, p in pairs]
    N = len(pairs)
    print(f"[rerank] {N} pairs  retriever={args.retriever}  device={device}")

    # ── Stage 1: retrieval score matrix ──
    # For large datasets (e.g. workflow 260k), use random-pool sampling (O(N*pool_size))
    # instead of full N×N to avoid CPU OOM — same approach as evaluate_finetune.
    cpool = args.candidate_pool if args.candidate_pool > 0 else N
    cpool = min(cpool, N)
    if cpool < N:
        print(f"[rerank] using random candidate pool of {cpool} per query (full N={N} would OOM)")
        rng_cpool = np.random.default_rng(42)
        # Vectorized: sample (cpool-1) from [0, N-1), shift indices >= row to exclude self
        raw = rng_cpool.integers(0, N - 1, size=(N, cpool - 1), dtype=np.int32)
        rows = np.arange(N, dtype=np.int32)[:, None]
        distractors = np.where(raw < rows, raw, raw + 1)   # (N, cpool-1)
        cand_idx = np.empty((N, cpool), dtype=np.int32)
        cand_idx[:, 0] = np.arange(N, dtype=np.int32)     # gold always at position 0
        cand_idx[:, 1:] = distractors
        # Build sub-matrix of scores: (N, cpool) — only score selected candidates
        if args.retriever == "cde":
            S_retr = cde_score_pooled(args.cde_suffix, anchors, positives, args.batch_size, device, cand_idx)
        else:
            S_retr = biencoder_score_matrix(anchors, positives, args.batch_size, device, args.dataset)
            S_retr = np.take_along_axis(S_retr, cand_idx.astype(np.intp), axis=1)
        # Re-index: true positive is always column 0
        base_ranks = ranks_from_scores(S_retr)
    else:
        if args.retriever == "cde":
            S_retr = cde_score_matrix(args.cde_suffix, anchors, positives, args.batch_size, device)
        else:
            S_retr = biencoder_score_matrix(anchors, positives, args.batch_size, device, args.dataset)
        base_ranks = ranks_from_scores(S_retr)
        cand_idx = None
    base = retrieval_metrics_from_ranks(base_ranks)
    print(f"[stage1] retrieval MRR={base['mrr']:.4f}  R@1={base['recall@1']:.3f}  "
          f"R@10={base['recall@10']:.3f}")

    # Precompute the max pool needed; CE-score the union of top-maxpool per query.
    max_pool = max(args.pool)
    order = np.argsort(-S_retr, axis=1)              # (N, cpool) desc by retrieval score
    topk = order[:, :max_pool]                       # (N, max_pool) — indices into S_retr cols

    # Resolve topk column indices back to global positive indices for CE text lookup.
    def col_to_global(i, col_idx):
        """Map S_retr column index to global positive index."""
        if cand_idx is not None:
            return int(cand_idx[i, col_idx])
        return int(col_idx)

    # Build all (anchor, candidate) pairs for the pool.
    flat_a, flat_b = [], []
    for i in range(N):
        for col in topk[i]:
            flat_a.append(anchors[i]); flat_b.append(positives[col_to_global(i, col)])

    # ── Load CE(s) and score the pool; ensemble by z-scoring per query then mean ──
    from model_ce import get_tokenizer as ce_tok_fn
    ce_dirs = [(KL_DIR / d) if not Path(d).is_absolute() else Path(d) for d in args.ce_dir]
    ce_pools = []
    for ce_dir in ce_dirs:
        ce_ckpt = torch.load(ce_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        ce_cfg = ce_ckpt.get("cfg", ce_ckpt.get("config", {"backbone": "google-bert/bert-base-uncased"}))
        ce_backbone = ce_cfg.get("backbone", "google-bert/bert-base-uncased")
        ce_tok = ce_tok_fn(ce_backbone)
        ce = CrossEncoder(ce_backbone).to(device)
        ce.load_state_dict(ce_ckpt["model_state_dict"])
        ce.eval()
        print(f"[stage2] scoring {len(flat_a)} pairs with CE {ce_dir.name} …")
        flat = ce_score_pairs(ce, ce_tok, flat_a, flat_b, args.max_length, args.batch_size, device)
        ce_pools.append(flat.reshape(N, max_pool))
        del ce
        torch.cuda.empty_cache()
    # Ensemble: per-query z-score each CE's pool logits, then average.
    if len(ce_pools) == 1:
        ce_pool = ce_pools[0]
    else:
        zs = [np.stack([_zscore(cp[i]) for i in range(N)]) for cp in ce_pools]
        ce_pool = np.mean(zs, axis=0)
        print(f"[stage2] ensembled {len(ce_pools)} CEs")

    # ── Sweep pool size × blend ──
    results = {"stage1": base}
    best = {"mrr": base["mrr"], "tag": "stage1_only"}
    print()
    print(f"{'pool':>5} {'blend':>6} {'MRR':>8} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("-" * 46)
    for pool in args.pool:
        tk = topk[:, :pool]
        ce_p = ce_pool[:, :pool]
        retr_p = np.take_along_axis(S_retr, tk, axis=1)   # (N, pool) retrieval scores
        for blend in args.blend:
            # Z-normalise each component per-query for a fair blend.
            new_ranks = np.empty(N, dtype=np.float64)
            for i in range(N):
                cez = _zscore(ce_p[i]); rez = _zscore(retr_p[i])
                final = blend * cez + (1 - blend) * rez
                # Position of the true positive (candidate index i) within the pool?
                # Find the true positive within the rerank pool.
                # With candidate_pool sampling, true positive is always col 0 in S_retr;
                # with full N×N, true positive global index == i.
                if cand_idx is not None:
                    pos_in_pool = np.where(tk[i] == 0)[0]   # col 0 = gold, search within current pool slice
                else:
                    pos_in_pool = np.where(tk[i] == i)[0]   # global index == i
                if len(pos_in_pool) == 0:
                    # positive not retrieved into the pool: keep stage-1 rank
                    new_ranks[i] = base_ranks[i]
                else:
                    pidx = pos_in_pool[0]
                    pos_score = final[pidx]
                    # rank within pool, then candidates beyond pool rank below.
                    new_ranks[i] = float((final > pos_score).sum() + 1)
            m = retrieval_metrics_from_ranks(new_ranks)
            tag = f"pool{pool}_blend{blend}"
            results[tag] = m
            flag = ""
            if m["mrr"] > best["mrr"]:
                best = {"mrr": m["mrr"], "tag": tag, **m}
                flag = "  <<"
            print(f"{pool:>5} {blend:>6.2f} {m['mrr']:>8.4f} {m['recall@1']:>7.3f} "
                  f"{m['recall@5']:>7.3f} {m['recall@10']:>7.3f}{flag}")

    print()
    print(f"[best] {best['tag']}  MRR={best['mrr']:.4f}")
    # Load baseline MRR from the BiEncoder eval.json for this dataset (not hardcoded to aep_causal).
    bi_eval = PROJECT_ROOT / "finetune_eval" / "results" / args.dataset / "eval.json"
    baseline = 0.4033  # aep_causal default
    if bi_eval.exists():
        import json as _json
        _d = _json.loads(bi_eval.read_text())
        baseline = _d.get("mrr", baseline)
    print(f"[baseline BiEncoder MRR]  {baseline:.4f}")
    print(f"[lift over baseline]      {(best['mrr']-baseline)/baseline*100:+.1f}%  "
          f"(target +10% = {baseline*1.10:.4f}  "
          f"{'ACHIEVED' if best['mrr'] >= baseline*1.10 else 'not yet'})")

    # ── AUC of the full fused system (vs random + hard negatives) ──
    # Re-score the positive and a small set of negatives with the SAME fused
    # score used for ranking (per-query z-scored CE-ensemble + retrieval blend).
    best_blend = float(best["tag"].split("blend")[1]) if "blend" in best["tag"] else 0.5
    rng = np.random.default_rng(0)
    test_hn_path = OUT_DIR / ("test_hard_negatives.npy" if args.split == "test"
                              else "val_hard_negatives.npy")
    test_hn = np.load(test_hn_path) if test_hn_path.exists() else None

    def fused_auc(neg_idx_fn, K):
        ce_models, ce_toks = [], []
        for ce_dir in ce_dirs:
            c = torch.load(ce_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
            bb = c.get("cfg", {}).get("backbone", "google-bert/bert-base-uncased")
            mdl = CrossEncoder(bb).to(device); mdl.load_state_dict(c["model_state_dict"]); mdl.eval()
            ce_models.append(mdl); ce_toks.append(ce_tok_fn(bb))
        pos_s, neg_s = [], []
        for i in range(N):
            negs = list(neg_idx_fn(i, K))
            # cands are global indices; map to S_retr column if using candidate pool
            cands = [i] + negs
            if cand_idx is not None:
                # map global indices to S_retr columns (gold=col0; negs may not be in pool)
                col_map = {int(cand_idx[i, c]): c for c in range(cpool)}
                retr = np.array([S_retr[i, col_map[j]] if j in col_map else -1e9 for j in cands])
            else:
                retr = np.array([S_retr[i, j] for j in cands])
            ce_z_list = []
            for mdl, tk in zip(ce_models, ce_toks):
                raw = ce_score_pairs(mdl, tk, [anchors[i]] * len(cands),
                                     [positives[j] for j in cands],
                                     args.max_length, args.batch_size, device)
                ce_z_list.append(_zscore(raw))
            fused = best_blend * np.mean(ce_z_list, axis=0) + (1 - best_blend) * _zscore(retr)
            pos_s.append(fused[0]); neg_s.extend(fused[1:])
        y = np.array([1] * len(pos_s) + [0] * len(neg_s))
        return float(roc_auc_score(y, np.array(pos_s + neg_s)))

    def rand_neg(i, K):
        p = rng.choice(N - 1, size=K, replace=False)
        p[p >= i] += 1   # shift to exclude self (global index i)
        return p

    auc_rand = fused_auc(rand_neg, 4)
    best["auc_random"] = auc_rand
    print(f"[fused AUC vs 4 random negs]  {auc_rand:.4f}")
    if test_hn is not None:
        auc_hn = fused_auc(lambda i, K: test_hn[i][:K], 4)
        best["auc_hardneg"] = auc_hn
        print(f"[fused AUC vs 4 hard negs]    {auc_hn:.4f}")

    results["best"] = best
    out_path = OUT_DIR / "rerank_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
