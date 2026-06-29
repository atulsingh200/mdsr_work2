"""Evaluation for Lorentz encoder. Handles e5/BGE prefix conventions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation_6.metrics_extra import (  # noqa
    auc_precision_at_1_with_random_negatives,
    random_pool_retrieval_metrics,
)
from finetune_eval.data import load_pairs  # noqa
from finetune_eval.datasets import split_path  # noqa
from ..model.encoder import LorentzEncoder


def auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def encode_all_fn(model, tokenizer, texts, device, max_length=256, batch_size=64,
                  which="anchor", prefix=""):
    model.eval()
    space_all, time_all = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        if prefix:
            chunk = [prefix + t for t in chunk]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        if which == "positive":
            s, t = model.encode_positive(ids, mask)
        else:
            s, t = model.encode_anchor(ids, mask)
        space_all.append(s.float().cpu().numpy())
        time_all.append(t.float().cpu().numpy())
    return (np.concatenate(space_all) if space_all else np.zeros((0, model.space_dim)),
            np.concatenate(time_all) if time_all else np.zeros((0, model.time_dim)))


def evaluate_lorentz(
    dataset: str,
    ckpt_path: str,
    pool_size: int = 1000,
    n_negatives: int = 4,
    k_values=(1, 3, 5, 10),
    batch_size: int = 128,
    seed: int = 0,
    use_time: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 10.0,
    test_path_override: str = None,
):
    device = auto_device()
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    anchor_prefix = cfg.get("anchor_prefix", "")
    positive_prefix = cfg.get("positive_prefix", "")

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    is_tied = cfg.get("tied", False)
    model = LorentzEncoder(
        backbone_name=cfg["backbone"],
        space_dim=cfg["space_dim"],
        time_dim=cfg["time_dim"],
        space_hidden=cfg["space_hidden"],
        time_hidden=cfg["time_hidden"],
        time_layers=cfg.get("time_layers", 2),
        pooling=cfg["pooling"],
        tied=is_tied,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  backbone={cfg['backbone']}  space_dim={cfg['space_dim']}  "
          f"tied={is_tied}  anchor_prefix={anchor_prefix!r}")

    if test_path_override:
        test_path = Path(test_path_override)
    else:
        test_path = split_path(dataset, "test")
    pairs = load_pairs(test_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]

    t0 = time.time()
    a_space, a_time = encode_all_fn(model, tokenizer, anchors, device,
                                    max_length=cfg.get("max_length", 256), batch_size=batch_size,
                                    which="anchor", prefix=anchor_prefix)
    p_space, p_time = encode_all_fn(model, tokenizer, positives, device,
                                    max_length=cfg.get("max_length", 256), batch_size=batch_size,
                                    which="positive", prefix=positive_prefix)
    enc_time = time.time() - t0

    pair_metrics = auc_precision_at_1_with_random_negatives(
        a_space, p_space, n_negatives=n_negatives, seed=seed,
    )
    retrieval_metrics = random_pool_retrieval_metrics(
        a_space, p_space,
        pool_size=pool_size, k_values=list(k_values), seed=seed,
    )

    row = {
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "backbone": cfg["backbone"],
        "space_dim": cfg["space_dim"],
        "time_dim": cfg["time_dim"],
        "n_test_pairs": len(pairs),
        "encoding_seconds": enc_time,
        "auc": pair_metrics["auc"],
        "precision_at_1": pair_metrics["precision_at_1"],
        "mrr": retrieval_metrics["mrr"],
        "mean_rank": retrieval_metrics["mean_rank"],
        "median_rank": retrieval_metrics["median_rank"],
        "recall_at_k": retrieval_metrics["recall_at_k"],
        "hit_at_k": retrieval_metrics["hit_at_k"],
        "n_candidates_per_query": retrieval_metrics["n_candidates_per_query"],
    }

    # ---- Time-head diagnostics (only meaningful if time head was trained) ----
    beta_cfg = cfg.get("beta", 0.0)
    tau_score = cfg.get("tau_score", 10.0)
    use_time = cfg.get("use_time", False)
    if use_time:
        import torch as _t
        ta = _t.tensor(a_time)
        tp = _t.tensor(p_time)
        # 1) Directional accuracy: for true pairs (a, b), does the model place
        #    b "after" a more than a "after" b?  score_fwd vs score_rev.
        #    score = mean_k sigmoid(tau * (t_b - t_a)).
        fwd = _t.sigmoid(tau_score * (tp - ta)).mean(-1)   # a -> b
        rev = _t.sigmoid(tau_score * (ta - tp)).mean(-1)   # b -> a
        row["directional_acc"] = float((fwd > rev).float().mean())
        row["directional_margin"] = float((fwd - rev).mean())

        # 2) Combined space+time retrieval MRR (use best beta from cfg, fallback 0.5)
        beta_eval = beta_cfg if beta_cfg > 0 else 0.5
        comb = _combined_retrieval_metrics(
            a_space, p_space, a_time, p_time,
            beta=beta_eval, tau_score=tau_score,
            pool_size=pool_size, k_values=list(k_values), seed=seed,
        )
        row["mrr_combined"] = comb["mrr"]
        row["median_rank_combined"] = comb["median_rank"]
        row["recall_at_k_combined"] = comb["recall_at_k"]
        row["beta_eval"] = beta_eval

    print(f"[{dataset}] AUC={row['auc']:.4f}  P@1={row['precision_at_1']:.4f}  "
          f"MRR={row['mrr']:.4f}  med_rank={row['median_rank']:.0f}")
    if use_time:
        print(f"  [time] dir_acc={row['directional_acc']:.4f}  "
              f"dir_margin={row['directional_margin']:+.4f}  "
              f"MRR_combined={row['mrr_combined']:.4f} (β={row['beta_eval']})")
    for k in k_values:
        print(f"  R@{k}={retrieval_metrics['recall_at_k'][k]:.4f}", end="  ")
    print()
    return row


def _combined_retrieval_metrics(a_space, p_space, a_time, p_time,
                                 beta, tau_score, pool_size, k_values, seed):
    """Retrieval metrics using combined space + time score over a random pool.

    Mirrors random_pool_retrieval_metrics but the score is
        cos(space) + beta * mean_k sigmoid(tau*(t_cand - t_query)).
    """
    import numpy as _np
    N = a_space.shape[0]
    rng = _np.random.default_rng(seed)
    pool = min(pool_size, N)
    ranks = []
    a_s = a_space; p_s = p_space
    a_t = a_time; p_t = p_time
    for i in range(N):
        # build candidate pool: gold (i) + (pool-1) random distractors
        cand = rng.choice(N, size=pool, replace=False)
        if i not in cand:
            cand[0] = i
        # space cosine
        s_score = a_s[i] @ p_s[cand].T                      # (pool,)
        # time term
        diff = p_t[cand] - a_t[i][None, :]                  # (pool, Dt)
        t_score = (1.0 / (1.0 + _np.exp(-tau_score * diff))).mean(-1)
        score = s_score + beta * t_score
        gold_pos = _np.where(cand == i)[0][0]
        rank = 1 + int((score > score[gold_pos]).sum())
        ranks.append(rank)
    ranks = _np.array(ranks)
    out = {"mrr": float((1.0 / ranks).mean()),
           "median_rank": float(_np.median(ranks)),
           "recall_at_k": {k: float((ranks <= k).mean()) for k in k_values}}
    return out
