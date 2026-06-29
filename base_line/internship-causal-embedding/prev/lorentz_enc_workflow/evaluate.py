"""Evaluate a LorentzBiEncoder checkpoint on the workflow test split.

Matches the output schema of finetune_eval/evaluate_finetune.py and
evaluate_finetune_hardneg.py so results are directly comparable.

Two output files in results/workflow/:
  eval.json         — AUC/P@1 (4 random negs) + MRR/R@k (pool=1000)
  eval_hardneg.json — AUC/P@1 (4 semantic hard negs, pre-mined)

Scoring uses the ASYMMETRIC causal_similarity_matrix rather than plain
cosine, so results reflect the time-ordering signal.

Usage:
  uv run python lorentz_enc_workflow/evaluate.py \\
      --results-dir lorentz_enc_workflow/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lorentz_enc_workflow.model import LorentzBiEncoder, get_tokenizer, tokenize  # noqa: E402
from lorentz_enc_workflow.scoring import causal_similarity_matrix                  # noqa: E402

from evaluation_6.metrics_extra import (                                            # noqa: E402
    auc_precision_at_1_with_semantic_hard_negatives,
)
from finetune_eval.data import load_pairs                                           # noqa: E402
from finetune_eval.datasets import split_path                                       # noqa: E402


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_all(
    model: LorentzBiEncoder,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts → (space_emb, time_emb) numpy arrays."""
    model.eval()
    space_out, time_out = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        ids, mask = tokenize(tokenizer, chunk, max_length, device)
        s, t = model(ids, mask)
        space_out.append(s.float().cpu().numpy())
        time_out.append(t.float().cpu().numpy())
    return (
        np.concatenate(space_out, axis=0),
        np.concatenate(time_out, axis=0),
    )


def causal_auc_p1_random(
    space_a: np.ndarray,
    time_a: np.ndarray,
    space_p: np.ndarray,
    time_p: np.ndarray,
    n_negatives: int = 4,
    seed: int = 0,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 10.0,
) -> dict:
    """AUC and P@1 using the asymmetric causal score vs random negatives."""
    from sklearn.metrics import roc_auc_score

    N = len(space_a)
    rng = np.random.default_rng(seed)

    # Positive scores
    sa = torch.from_numpy(space_a)
    ta = torch.from_numpy(time_a)
    sp = torch.from_numpy(space_p)
    tp = torch.from_numpy(time_p)
    pos_sims = causal_similarity_matrix(sa, ta, sp, tp, alpha=alpha, beta=beta, tau=tau)
    pos_scores = pos_sims.diag().numpy()          # (N,)

    # Random negative scores
    neg_idx = np.empty((N, n_negatives), dtype=np.int64)
    for i in range(N):
        picks = rng.choice(N - 1, size=n_negatives, replace=False)
        picks[picks >= i] += 1
        neg_idx[i] = picks

    neg_scores = np.empty((N, n_negatives), dtype=np.float32)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    chunk = 512
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        q_s = sa[start:end].to(device_str)
        q_t = ta[start:end].to(device_str)
        for j in range(n_negatives):
            nidx = neg_idx[start:end, j]
            k_s = sp[nidx].to(device_str)
            k_t = tp[nidx].to(device_str)
            from lorentz_enc_workflow.scoring import causal_similarity
            neg_scores[start:end, j] = causal_similarity(
                q_s, q_t, k_s, k_t, alpha=alpha, beta=beta, tau=tau,
            ).float().cpu().numpy()

    p_at_1 = float(np.mean(pos_scores > neg_scores.max(axis=1)))
    scores = np.concatenate([pos_scores, neg_scores.ravel()])
    labels = np.concatenate([np.ones(N, dtype=np.int8), np.zeros(N * n_negatives, dtype=np.int8)])
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "precision_at_1": p_at_1,
            "n_anchors": N, "n_negatives_per_anchor": n_negatives}


def causal_random_pool_retrieval(
    space_a: np.ndarray,
    time_a: np.ndarray,
    space_p: np.ndarray,
    time_p: np.ndarray,
    pool_size: int = 1000,
    k_values: list[int] | None = None,
    seed: int = 0,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 10.0,
    batch_size: int = 128,
) -> dict:
    """MRR / Recall@k using causal asymmetric score, random distractor pool."""
    if k_values is None:
        k_values = [1, 3, 5, 10]
    k_values = sorted(set(k_values))

    N = len(space_a)
    actual_pool = min(pool_size, N)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sa = torch.from_numpy(space_a).to(device)
    ta = torch.from_numpy(time_a).to(device)
    sp = torch.from_numpy(space_p).to(device)
    tp = torch.from_numpy(time_p).to(device)

    rng = np.random.default_rng(seed)
    if actual_pool >= N:
        # Full NxN
        ranks = torch.zeros(N, dtype=torch.long, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            sim_batch = causal_similarity_matrix(
                sa[start:end], ta[start:end], sp, tp,
                alpha=alpha, beta=beta, tau=tau,
            )                                            # (b, N)
            pos_score = sim_batch[
                torch.arange(end - start, device=device),
                torch.arange(start, end, device=device),
            ].unsqueeze(1)                               # (b, 1)
            ranks[start:end] = (sim_batch > pos_score).sum(dim=1) + 1
        n_candidates = N
    else:
        n_distractors = actual_pool - 1
        rand = rng.integers(0, N - 1, size=(N, n_distractors), dtype=np.int64)
        rand = rand + (rand >= np.arange(N)[:, None])
        d_idx = torch.from_numpy(rand).to(device)

        ranks = torch.zeros(N, dtype=torch.long, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_d = d_idx[start:end]                   # (b, n_dist)
            # positive score
            pos_s = causal_similarity_matrix(
                sa[start:end], ta[start:end],
                sp[torch.arange(start, end, device=device)],
                tp[torch.arange(start, end, device=device)],
                alpha=alpha, beta=beta, tau=tau,
            ).diag().unsqueeze(1)                        # (b, 1)

            # distractor scores — iterate over rows to avoid OOM on large N
            # Use gathered approach: batch_d is (b, n_dist)
            flat_d = batch_d.reshape(-1)                 # (b*n_dist,)
            neg_s = causal_similarity_matrix(
                sa[start:end].repeat_interleave(n_distractors, dim=0),
                ta[start:end].repeat_interleave(n_distractors, dim=0),
                sp[flat_d], tp[flat_d],
                alpha=alpha, beta=beta, tau=tau,
            ).view(end - start, n_distractors)           # (b, n_dist)
            ranks[start:end] = (neg_s > pos_s).sum(dim=1) + 1
        n_candidates = actual_pool

    ranks_np = ranks.cpu().numpy()
    recall = {k: float((ranks_np <= k).mean()) for k in k_values}
    return {
        "n_queries": N,
        "n_candidates_per_query": n_candidates,
        "mrr":         float((1.0 / ranks_np).mean()),
        "mean_rank":   float(ranks_np.mean()),
        "median_rank": float(np.median(ranks_np)),
        "recall_at_k": recall,
        "hit_at_k":    recall,
    }


def causal_auc_p1_hardneg(
    space_a: np.ndarray,
    time_a: np.ndarray,
    space_p: np.ndarray,
    time_p: np.ndarray,
    hard_neg_idx: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 10.0,
) -> dict:
    """AUC and P@1 using the asymmetric causal score vs semantic hard negatives."""
    from sklearn.metrics import roc_auc_score

    N = len(space_a)
    K = hard_neg_idx.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sa = torch.from_numpy(space_a).to(device)
    ta = torch.from_numpy(time_a).to(device)
    sp = torch.from_numpy(space_p).to(device)
    tp = torch.from_numpy(time_p).to(device)

    from lorentz_enc_workflow.scoring import causal_similarity
    pos_scores = causal_similarity(sa, ta, sp, tp, alpha=alpha, beta=beta, tau=tau).cpu().numpy()

    neg_scores = np.empty((N, K), dtype=np.float32)
    chunk = 512
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        for k in range(K):
            nidx = hard_neg_idx[start:end, k]
            neg_scores[start:end, k] = causal_similarity(
                sa[start:end], ta[start:end],
                sp[torch.from_numpy(nidx).to(device)],
                tp[torch.from_numpy(nidx).to(device)],
                alpha=alpha, beta=beta, tau=tau,
            ).cpu().numpy()

    p_at_1 = float(np.mean(pos_scores > neg_scores.max(axis=1)))
    scores = np.concatenate([pos_scores, neg_scores.ravel()])
    labels = np.concatenate([np.ones(N, dtype=np.int8), np.zeros(N * K, dtype=np.int8)])
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "precision_at_1": p_at_1,
            "n_anchors": N, "n_negatives_per_anchor": K}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "lorentz_enc_workflow/results"))
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds_dir = Path(args.results_dir) / "workflow"
    ckpt_path = ds_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}\n"
            "Run train.py first."
        )

    print(f"\n[workflow] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    device = auto_device()
    tokenizer = get_tokenizer(cfg["backbone"])
    model = LorentzBiEncoder(
        backbone_name=cfg["backbone"],
        space_dim=cfg["space_dim"],
        time_dim=cfg["time_dim"],
        space_hidden=cfg.get("space_hidden", 1024),
        time_hidden=cfg.get("time_hidden", 512),
        dropout=0.0,
        pooling=cfg["pooling"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  backbone: {cfg['backbone']}  space_dim={cfg['space_dim']}  time_dim={cfg['time_dim']}")

    alpha = cfg.get("alpha", 1.0)
    beta = cfg.get("beta", 0.5)
    tau = cfg.get("tau", 10.0)

    # ---- Load test pairs ----
    test_path = split_path("workflow", "test")
    if test_path is None or not test_path.exists():
        raise FileNotFoundError(f"No test file: {test_path}")
    pairs = load_pairs(test_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    print(f"  test pairs: {len(pairs):,}")

    t0 = time.time()
    space_a, time_a = encode_all(model, tokenizer, anchors,   cfg["max_seq_length"], args.batch_size, device)
    space_p, time_p = encode_all(model, tokenizer, positives, cfg["max_seq_length"], args.batch_size, device)
    enc_time = time.time() - t0
    print(f"  encoded {len(pairs)*2:,} texts in {enc_time:.1f}s")

    k_values = sorted(set(args.k))

    # ---- eval.json: random pool ----
    print("\n  [random negatives]")
    pair_m = causal_auc_p1_random(
        space_a, time_a, space_p, time_p,
        n_negatives=args.n_negatives, seed=args.seed,
        alpha=alpha, beta=beta, tau=tau,
    )
    print(f"  AUC={pair_m['auc']:.4f}  P@1={pair_m['precision_at_1']:.4f}")

    print("\n  [random pool retrieval]")
    retr_m = causal_random_pool_retrieval(
        space_a, time_a, space_p, time_p,
        pool_size=args.pool_size, k_values=k_values, seed=args.seed,
        alpha=alpha, beta=beta, tau=tau,
        batch_size=args.batch_size,
    )
    print(f"  MRR={retr_m['mrr']:.4f}  mean_rank={retr_m['mean_rank']:.2f}  "
          f"median_rank={retr_m['median_rank']:.1f}")
    for k in k_values:
        print(f"  Recall@{k:<3}{retr_m['recall_at_k'][k]:.4f}")

    row_eval = {
        "dataset":                "workflow",
        "checkpoint":             str(ckpt_path),
        "backbone":               cfg["backbone"],
        "pooling":                cfg["pooling"],
        "space_dim":              cfg["space_dim"],
        "time_dim":               cfg["time_dim"],
        "alpha":                  alpha, "beta": beta, "tau": tau,
        "n_test_pairs":           len(pairs),
        "n_candidates_per_query": retr_m["n_candidates_per_query"],
        "encoding_seconds":       enc_time,
        "auc":                    pair_m["auc"],
        "precision_at_1":         pair_m["precision_at_1"],
        "n_negatives":            args.n_negatives,
        "mrr":                    retr_m["mrr"],
        "mean_rank":              retr_m["mean_rank"],
        "median_rank":            retr_m["median_rank"],
        "recall_at_k":            retr_m["recall_at_k"],
        "hit_at_k":               retr_m["hit_at_k"],
    }
    (ds_dir / "eval.json").write_text(json.dumps(row_eval, indent=2))
    print(f"\n  saved -> {ds_dir / 'eval.json'}")

    # ---- eval_hardneg.json: semantic hard negatives ----
    hn_path = ds_dir / "test_hard_negatives.npy"
    test_pairs_path = ds_dir / "test_pairs.jsonl"
    if not hn_path.exists() or not test_pairs_path.exists():
        print(
            "\n  [skip hard-neg eval] test_hard_negatives.npy not found. Run:\n"
            "    uv run python finetune_eval/mine_hard_negatives.py "
            "--dataset workflow --split test --out-dir lorentz_enc_workflow/results"
        )
    else:
        test_pairs_mined = load_pairs(test_pairs_path, cap=None)
        hard_neg_idx = np.load(hn_path)
        if hard_neg_idx.shape[0] != len(test_pairs_mined):
            print(f"  [skip hard-neg eval] index mismatch: "
                  f"{hard_neg_idx.shape[0]} != {len(test_pairs_mined)}")
        else:
            print("\n  [semantic hard negatives]")
            # Use mined test pairs (may differ slightly from test_path due to sampling)
            anc_m = [a for a, _ in test_pairs_mined]
            pos_m = [p for _, p in test_pairs_mined]
            space_am, time_am = encode_all(model, tokenizer, anc_m, cfg["max_seq_length"], args.batch_size, device)
            space_pm, time_pm = encode_all(model, tokenizer, pos_m, cfg["max_seq_length"], args.batch_size, device)

            hn_m = causal_auc_p1_hardneg(
                space_am, time_am, space_pm, time_pm,
                hard_neg_idx, alpha=alpha, beta=beta, tau=tau,
            )
            print(f"  AUC={hn_m['auc']:.4f}  P@1={hn_m['precision_at_1']:.4f}  "
                  f"(vs {hn_m['n_negatives_per_anchor']} semantic hard negs)")

            row_hn = {
                "dataset":           "workflow",
                "checkpoint":        str(ckpt_path),
                "backbone":          cfg["backbone"],
                "pooling":           cfg["pooling"],
                "space_dim":         cfg["space_dim"],
                "time_dim":          cfg["time_dim"],
                "alpha":             alpha, "beta": beta, "tau": tau,
                "n_test_pairs":      len(test_pairs_mined),
                "n_negatives":       int(hn_m["n_negatives_per_anchor"]),
                "auc":               hn_m["auc"],
                "precision_at_1":    hn_m["precision_at_1"],
                "eval_type":         "semantic_hard_negatives",
                "miner":             "sentence-transformers/all-MiniLM-L6-v2",
            }
            (ds_dir / "eval_hardneg.json").write_text(json.dumps(row_hn, indent=2))
            print(f"  saved -> {ds_dir / 'eval_hardneg.json'}")

    # ---- Comparison table ----
    print(f"\n{'='*80}")
    print("Results summary (vs finetune_eval/results/workflow baseline)")
    print(f"{'='*80}")
    print(f"{'metric':<20} {'lorentz_enc':>14} {'finetune_eval':>14}")
    print(f"{'-'*50}")
    baseline = {"auc": 0.9662, "precision_at_1": 0.9104, "mrr": 0.5500,
                "recall@1": 0.4656, "recall@5": 0.6428, "recall@10": 0.7055}
    print(f"{'AUC (random)':<20} {pair_m['auc']:>14.4f} {baseline['auc']:>14.4f}")
    print(f"{'P@1 (random)':<20} {pair_m['precision_at_1']:>14.4f} {baseline['precision_at_1']:>14.4f}")
    print(f"{'MRR':<20} {retr_m['mrr']:>14.4f} {baseline['mrr']:>14.4f}")
    for k in [1, 5, 10]:
        if k in retr_m["recall_at_k"]:
            print(f"{'Recall@'+str(k):<20} {retr_m['recall_at_k'][k]:>14.4f} {baseline.get('recall@'+str(k), float('nan')):>14.4f}")


if __name__ == "__main__":
    main()
