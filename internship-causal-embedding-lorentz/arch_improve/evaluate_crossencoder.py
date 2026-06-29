"""Evaluate the cross-encoder on the test split and compare to the BiEncoder.

Builds the full N×N cross-encoder score matrix (anchor_i vs every test
positive_j), then computes the same metrics as the baseline:
  * N×N MRR, Recall@K, mean/median rank (rank true positive among all positives)
  * AUC and P@1 vs 4 random negatives (sampled from the score matrix)

Base model is bert-base-uncased — identical to the BiEncoder baseline — so any
gain is purely from the cross-encoder architecture.

Usage:
  PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY arch_improve/evaluate_crossencoder.py --dataset aep_causal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finetune_eval.data import load_pairs                      # noqa: E402
from finetune_eval.datasets import split_path                  # noqa: E402
from arch_improve.train_crossencoder import (                  # noqa: E402
    CrossEncoder, score_matrix, retrieval_from_scores, check_memory,
)

# Matched baseline (bert-base-uncased BiEncoder, same eval code) on aep_causal test
BASELINE = {"aep_causal": {
    "nn_mrr": 0.4033, "pool_mrr": 0.3574, "auc": 0.9688, "p_at_1": 0.9146,
    "recall@1": 0.1833, "recall@5": 0.5712, "recall@10": 0.7153}}


def auc_p1_from_scores(S, n_neg=4, seed=0):
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    pos = np.diag(S)
    neg = np.empty((n, n_neg), dtype=np.float32)
    for i in range(n):
        picks = rng.choice(n - 1, size=n_neg, replace=False)
        picks[picks >= i] += 1
        neg[i] = S[i, picks]
    p1 = float(np.mean(pos > neg.max(axis=1)))
    scores = np.concatenate([pos, neg.ravel()])
    labels = np.concatenate([np.ones(n), np.zeros(n * n_neg)])
    return {"auc": float(roc_auc_score(labels, scores)), "precision_at_1": p1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aep_causal")
    ap.add_argument("--results-dir", default=str(ROOT / "arch_improve/results"))
    ap.add_argument("--batch-pairs", type=int, default=256)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.results_dir) / f"{args.dataset}_crossencoder"
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    print(f"[eval cross-encoder]  best epoch={ckpt['epoch']+1}  val MRR={ckpt['val_metrics']['mrr']:.4f}")
    check_memory("start")

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    model = CrossEncoder(cfg["backbone"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_pairs = load_pairs(split_path(args.dataset, "test"), cap=None)
    anchors = [a for a, _ in test_pairs]
    cands = [p for _, p in test_pairs]
    print(f"  test pairs: {len(test_pairs)}  → scoring {len(anchors)}×{len(cands)} matrix")

    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
          if (args.bf16 and device.type == "cuda") else (lambda: nullcontext())

    t0 = time.time()
    S = score_matrix(model, tokenizer, anchors, cands, cfg["max_seq_length"],
                     device, batch_pairs=args.batch_pairs, amp_ctx=amp)
    print(f"  scored in {time.time()-t0:.1f}s")
    check_memory("scored")

    ret = retrieval_from_scores(S, k_values=(1, 3, 5, 10))
    pr = auc_p1_from_scores(S, n_neg=args.n_negatives, seed=args.seed)

    row = {"dataset": args.dataset, "arch": "cross_encoder", "backbone": cfg["backbone"],
           "n_test": int(S.shape[0]), **ret, **pr}
    (run_dir / "eval.json").write_text(json.dumps(row, indent=2))

    b = BASELINE.get(args.dataset, {})
    print()
    print("=" * 82)
    print(f"{'Metric':<22} {'Cross-encoder':>14} {'BERT BiEncoder':>16} {'Δ':>9} {'Δ%':>8}")
    print("-" * 82)

    def line(name, ours, base):
        if base:
            d = ours - base; pct = 100 * d / base
            print(f"  {name:<20} {ours:>14.4f} {base:>16.4f} {d:>+9.4f} {pct:>+7.1f}%")
        else:
            print(f"  {name:<20} {ours:>14.4f} {'—':>16}")

    line("N×N MRR",   ret["mrr"],            b.get("nn_mrr"))
    line("AUC",       pr["auc"],             b.get("auc"))
    line("P@1",       pr["precision_at_1"],  b.get("p_at_1"))
    line("Recall@1",  ret["recall@1"],       b.get("recall@1"))
    line("Recall@5",  ret["recall@5"],       b.get("recall@5"))
    line("Recall@10", ret["recall@10"],      b.get("recall@10"))
    print(f"  {'mean rank':<20} {ret['mean_rank']:>14.2f}")
    print(f"  {'median rank':<20} {ret['median_rank']:>14.1f}")
    print("=" * 82)
    if b.get("nn_mrr"):
        pct = 100 * (ret["mrr"] - b["nn_mrr"]) / b["nn_mrr"]
        ok = "✓✓" if pct >= 10 else ("✓" if pct > 0 else "✗")
        print(f"  {ok}  N×N MRR {pct:+.1f}%  vs BiEncoder (same bert-base-uncased)  [target ≥ +10%]")


if __name__ == "__main__":
    main()
