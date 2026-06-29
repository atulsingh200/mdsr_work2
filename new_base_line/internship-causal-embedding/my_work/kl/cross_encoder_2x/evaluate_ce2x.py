"""Evaluate the CrossEncoder-2x on a dataset's test split.

Identical metric groups to evaluate_ce.py so results are directly comparable:
  1. AUC + P@1 vs 4 RANDOM negatives per anchor.
  2. AUC + P@1 vs 4 SEMANTIC HARD negatives per anchor (pre-mined MiniLM kNN).
  3. Random-pool retrieval: rank gold among (pool_size - 1) random distractors.
     Reports MRR + Recall@K + median rank.

Outputs:
  results/<dataset>/eval.json
  results/<dataset>/eval_hardneg.json
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

HERE = Path(__file__).resolve().parent
CE_DIR = HERE.parent / "cross_encoder"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CE_DIR))

from model_ce2x import CrossEncoder2x, get_tokenizer
from dataset_ce import load_pairs  # reuse 1x dataset loader


DEFAULT_DATA_ROOT = Path("/mnt/localssd/base_line/my_work/kl/cross_encoder/data")


def auto_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def score_pairs(
    model: CrossEncoder2x, tokenizer,
    text_a: list[str], text_b: list[str],
    max_length: int, batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval()
    out: list[float] = []
    for start in range(0, len(text_a), batch_size):
        end = min(start + batch_size, len(text_a))
        enc = tokenizer(
            text_a[start:end], text_b[start:end],
            padding=True, truncation="longest_first",
            max_length=max_length, return_tensors="pt",
            return_token_type_ids=True,
        )
        ids = enc["input_ids"].to(device, non_blocking=True)
        mask = enc["attention_mask"].to(device, non_blocking=True)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        out.extend(logits.float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def auc_p1(pos_sims: np.ndarray, neg_sims: np.ndarray) -> dict:
    p_at_1 = float(np.mean(pos_sims > neg_sims.max(axis=1)))
    scores = np.concatenate([pos_sims, neg_sims.ravel()])
    labels = np.concatenate([
        np.ones(len(pos_sims), dtype=np.int8),
        np.zeros(neg_sims.size, dtype=np.int8),
    ])
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "p_at_1": p_at_1, "n_anchors": int(len(pos_sims)),
            "n_negatives_per_anchor": int(neg_sims.shape[1])}


def pool_retrieval(
    model: CrossEncoder2x, tokenizer,
    anchors: list[str], positives: list[str],
    query_indices: np.ndarray,
    pool_size: int, k_values: list[int],
    batch_size: int, max_length: int, device: torch.device, seed: int,
) -> dict:
    N = len(positives)
    actual_pool = min(pool_size, N)
    Q = len(query_indices)
    rng = np.random.default_rng(seed)

    if actual_pool >= N:
        cand_idx = np.empty((Q, N), dtype=np.int64)
        for q, i in enumerate(query_indices):
            cand_idx[q, 0] = i
            cand_idx[q, 1:] = np.concatenate([np.arange(i), np.arange(i + 1, N)])
    else:
        n_distractors = actual_pool - 1
        cand_idx = np.empty((Q, actual_pool), dtype=np.int64)
        for q, i in enumerate(query_indices):
            picks = rng.choice(N - 1, size=n_distractors, replace=False)
            picks[picks >= i] += 1
            cand_idx[q, 0]  = i
            cand_idx[q, 1:] = picks

    print(f"  [pool] scoring {Q * actual_pool:,} (anchor, candidate) pairs …")
    t0 = time.time()
    flat_a: list[str] = []
    flat_b: list[str] = []
    for q, i in enumerate(query_indices):
        flat_a.extend([anchors[i]] * actual_pool)
        flat_b.extend([positives[j] for j in cand_idx[q]])
    flat_scores = score_pairs(model, tokenizer, flat_a, flat_b,
                              max_length, batch_size, device)
    print(f"  [pool] done in {time.time() - t0:.1f}s")

    scores = flat_scores.reshape(Q, actual_pool)
    gold_scores = scores[:, 0]
    ranks = (scores[:, 1:] > gold_scores[:, None]).sum(axis=1) + 1
    recall = {k: float((ranks <= k).mean()) for k in k_values}
    return {
        "n_queries": Q,
        "n_candidates_per_query": actual_pool,
        "mrr": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "recall_at_k": recall,
        "hit_at_k": recall,
    }


def evaluate(args) -> None:
    if args.out_dir is None:
        args.out_dir = str(HERE / "results" / args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint_best.pt"
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = auto_device()
    tokenizer = get_tokenizer(cfg["backbone"])
    model = CrossEncoder2x(
        backbone=cfg["backbone"],
        n_layers=cfg.get("n_layers", 24),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  backbone={cfg['backbone']}  layers={cfg.get('n_layers', 24)}  "
          f"params={n_params:,}  device={device}")

    ds_dir = Path(args.data_dir) / args.dataset
    test_pairs_path = ds_dir / "test_pairs.jsonl"
    test_hn_path    = ds_dir / "test_hard_negatives.npy"
    if not test_pairs_path.exists() or not test_hn_path.exists():
        raise FileNotFoundError(
            f"missing test artifacts for {args.dataset} at {ds_dir}.\n"
            f"Mine with:  mine_hard_negatives.py --dataset {args.dataset} --split test --k 4"
        )

    pairs = load_pairs(test_pairs_path)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    test_hn = np.load(test_hn_path)
    N = len(pairs)
    print(f"  test pairs: {N}    hard-neg matrix: {test_hn.shape}")

    print("\n[score] positive pairs …")
    t0 = time.time()
    pos_sims = score_pairs(model, tokenizer, anchors, positives,
                           args.max_length, args.batch_size, device)
    print(f"  done in {time.time() - t0:.1f}s  pos_mean={pos_sims.mean():.3f}")

    rng = np.random.default_rng(args.seed)
    n_neg = args.n_negatives
    rand_neg_idx = np.empty((N, n_neg), dtype=np.int64)
    for i in range(N):
        picks = rng.choice(N - 1, size=n_neg, replace=False)
        picks[picks >= i] += 1
        rand_neg_idx[i] = picks
    flat_a, flat_b = [], []
    for i in range(N):
        flat_a.extend([anchors[i]] * n_neg)
        flat_b.extend([positives[j] for j in rand_neg_idx[i]])
    print(f"\n[score] {len(flat_a):,} pairs vs RANDOM negatives …")
    t0 = time.time()
    rand_neg_sims = score_pairs(model, tokenizer, flat_a, flat_b,
                                args.max_length, args.batch_size, device).reshape(N, n_neg)
    print(f"  done in {time.time() - t0:.1f}s")
    rand_metrics = auc_p1(pos_sims, rand_neg_sims)
    print(f"  AUC(random)={rand_metrics['auc']:.4f}  P@1(random)={rand_metrics['p_at_1']:.4f}")

    K_hn = test_hn.shape[1]
    flat_a, flat_b = [], []
    for i in range(N):
        flat_a.extend([anchors[i]] * K_hn)
        flat_b.extend([positives[int(j)] for j in test_hn[i]])
    print(f"\n[score] {len(flat_a):,} pairs vs SEMANTIC HARD negatives …")
    t0 = time.time()
    hn_neg_sims = score_pairs(model, tokenizer, flat_a, flat_b,
                              args.max_length, args.batch_size, device).reshape(N, K_hn)
    print(f"  done in {time.time() - t0:.1f}s")
    hn_metrics = auc_p1(pos_sims, hn_neg_sims)
    print(f"  AUC(hardneg)={hn_metrics['auc']:.4f}  P@1(hardneg)={hn_metrics['p_at_1']:.4f}")

    if args.pool_size > 0:
        if args.max_queries > 0 and args.max_queries < N:
            rng2 = np.random.default_rng(args.seed)
            query_indices = np.sort(rng2.choice(N, size=args.max_queries, replace=False))
            print(f"\n[retrieve] sampled {len(query_indices)} of {N} anchors  "
                  f"(pool_size={args.pool_size})")
        else:
            query_indices = np.arange(N)
            print(f"\n[retrieve] all {N} anchors  (pool_size={args.pool_size})")
        pool_metrics = pool_retrieval(
            model, tokenizer, anchors, positives, query_indices,
            pool_size=args.pool_size, k_values=sorted(set(args.k)),
            batch_size=args.batch_size, max_length=args.max_length,
            device=device, seed=args.seed,
        )
        print(f"  MRR={pool_metrics['mrr']:.4f}  "
              f"mean_rank={pool_metrics['mean_rank']:.2f}  "
              f"median_rank={pool_metrics['median_rank']:.1f}")
        for k in sorted(set(args.k)):
            print(f"  Recall@{k:<3}{pool_metrics['recall_at_k'][k]:.4f}")
    else:
        pool_metrics = None

    eval_payload = {
        "model": "CrossEncoder2x",
        "n_params": n_params,
        "n_layers": cfg.get("n_layers", 24),
        "dataset": args.dataset,
        "checkpoint": str(ckpt_path),
        "backbone": cfg["backbone"],
        "n_test_pairs": N,
        "auc":            rand_metrics["auc"],
        "precision_at_1": rand_metrics["p_at_1"],
        "n_negatives":    rand_metrics["n_negatives_per_anchor"],
        "eval_type":      "random_negatives",
    }
    if pool_metrics is not None:
        eval_payload.update({
            "n_candidates_per_query": pool_metrics["n_candidates_per_query"],
            "n_queries":              pool_metrics["n_queries"],
            "mrr":         pool_metrics["mrr"],
            "mean_rank":   pool_metrics["mean_rank"],
            "median_rank": pool_metrics["median_rank"],
            "recall_at_k": pool_metrics["recall_at_k"],
            "hit_at_k":    pool_metrics["hit_at_k"],
        })
    (out_dir / "eval.json").write_text(json.dumps(eval_payload, indent=2))

    hn_payload = {
        "model": "CrossEncoder2x",
        "n_params": n_params,
        "n_layers": cfg.get("n_layers", 24),
        "dataset": args.dataset,
        "checkpoint": str(ckpt_path),
        "backbone": cfg["backbone"],
        "n_test_pairs": N,
        "auc":            hn_metrics["auc"],
        "precision_at_1": hn_metrics["p_at_1"],
        "n_negatives":    hn_metrics["n_negatives_per_anchor"],
        "eval_type":      "semantic_hard_negatives",
        "miner":          "sentence-transformers/all-MiniLM-L6-v2",
    }
    (out_dir / "eval_hardneg.json").write_text(json.dumps(hn_payload, indent=2))
    print(f"\n[saved] {out_dir / 'eval.json'}")
    print(f"[saved] {out_dir / 'eval_hardneg.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="Defaults to results/<dataset>/checkpoint_best.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--max-queries", type=int, default=2000)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Smaller than 1x CE due to 2x model size.")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_ROOT))
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
