"""Evaluate an e2e checkpoint (gnn/train_gnn_e2e.py) on the test split.

Same metrics as finetune_eval/evaluate_finetune.py + full N×N MRR, so results
are directly comparable to the bert-base-uncased two-tower baseline.

Usage:
  PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  $PY gnn/evaluate_e2e.py --run-dir gnn/results/aep_causal_e2e_gnn_bge-base-en-v1.5
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

from gnn.train_gnn import load_pairs_jsonl, build_graph, SPLITS_CFG, DATA_DIR, check_memory  # noqa: E402
from gnn.train_gnn_e2e import (E2EModel, make_tokenizer, encode_all)                          # noqa: E402
from evaluation_6.metrics_extra import (                                                       # noqa: E402
    auc_precision_at_1_with_random_negatives, random_pool_retrieval_metrics,
)

# Baseline numbers (bert-base-uncased two-tower, finetune_eval) on aep_causal test
BASELINES = {
    "aep_causal": {
        "n_nn_mrr": 0.4033, "pool_mrr": 0.3574, "auc": 0.9688, "p_at_1": 0.9146,
        "recall@1": 0.1833, "recall@5": 0.5712, "recall@10": 0.7153,
    },
}


@torch.no_grad()
def evaluate(run_dir: Path, dataset: str, pool_size=1000, n_neg=4,
             k_values=(1, 3, 5, 10), seed=0):
    check_memory("eval start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]; id2idx = ckpt["id2idx"]
    print(f"\n[evaluate_e2e]  {run_dir.name}  best epoch={ckpt['epoch']+1}  "
          f"val MRR={ckpt['val_metrics']['mrr']:.4f}")

    model = E2EModel(
        cfg["encoder"], cfg["max_seq_length"], use_gnn=cfg["use_gnn"],
        gnn_dim=cfg["gnn_dim"], num_gnn_layers=cfg["num_gnn_layers"],
        heads=cfg["heads"], dropout=cfg["dropout"], gate_init=cfg["gate_init"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tok = make_tokenizer(cfg["encoder"])

    dd = DATA_DIR / dataset
    s = SPLITS_CFG[dataset]
    train_pairs = load_pairs_jsonl(dd / s["train"])
    edge_index, _ = build_graph(train_pairs, id2idx)
    edge_index = edge_index.to(device)

    x_text = None
    if cfg["use_gnn"]:
        doc_cache = run_dir / "doc_embs.npy"
        x_text = torch.from_numpy(np.load(doc_cache)).float()

    test_pairs = load_pairs_jsonl(dd / s["test"])
    anchors = [a for a, _, _ in test_pairs]
    positives = [p for _, p, _ in test_pairs]

    t0 = time.time()
    a_all = torch.from_numpy(encode_all(model.anchor_tower, tok, anchors, cfg["max_seq_length"], device)).float()
    p_all = torch.from_numpy(encode_all(model.positive_tower, tok, positives, cfg["max_seq_length"], device)).float()
    h_nodes = model.encode_graph(x_text.to(device), edge_index) if cfg["use_gnn"] else None
    enc_time = time.time() - t0
    check_memory("after encode test")

    cs, es = [], []
    for i, (_, _, m) in enumerate(test_pairs):
        sid, tid = m["source_doc_id"], m["target_doc_id"]
        if sid not in id2idx or tid not in id2idx:
            continue
        a_e = a_all[i:i + 1].to(device); p_e = p_all[i:i + 1].to(device)
        if cfg["use_gnn"]:
            cs.append(model.cause_from(a_e, h_nodes[id2idx[sid]].unsqueeze(0)).cpu())
            es.append(model.effect_from(p_e, h_nodes[id2idx[tid]].unsqueeze(0)).cpu())
        else:
            cs.append(a_e.cpu()); es.append(p_e.cpu())

    A = torch.cat(cs).numpy(); P = torch.cat(es).numpy()
    sim = torch.from_numpy(A) @ torch.from_numpy(P).T
    ranks = (sim > sim.diag().unsqueeze(1)).sum(1).float() + 1
    nn_mrr = float((1 / ranks).mean())
    nn_r1 = float((ranks <= 1).float().mean())
    nn_r5 = float((ranks <= 5).float().mean())

    pm = auc_precision_at_1_with_random_negatives(A, P, n_neg, seed)
    rm = random_pool_retrieval_metrics(A, P, pool_size, list(k_values), seed)

    row = {
        "run": run_dir.name, "dataset": dataset, "n_test": len(A),
        "encoding_seconds": enc_time,
        "nn_mrr": nn_mrr, "nn_recall@1": nn_r1, "nn_recall@5": nn_r5,
        "auc": pm["auc"], "precision_at_1": pm["precision_at_1"],
        "pool_mrr": rm["mrr"], "mean_rank": rm["mean_rank"], "median_rank": rm["median_rank"],
        "recall_at_k": rm["recall_at_k"], "hit_at_k": rm["hit_at_k"],
    }
    (run_dir / "eval.json").write_text(json.dumps(row, indent=2))

    b = BASELINES.get(dataset, {})
    print()
    print("=" * 82)
    print(f"{'Metric':<22} {'E2E (ours)':>14} {'BERT two-tower':>16} {'Δ':>9} {'Δ%':>8}")
    print("-" * 82)

    def line(name, ours, base):
        if base:
            d = ours - base
            pct = 100 * d / base if base else 0
            print(f"  {name:<20} {ours:>14.4f} {base:>16.4f} {d:>+9.4f} {pct:>+7.1f}%")
        else:
            print(f"  {name:<20} {ours:>14.4f} {'—':>16}")

    line("N×N MRR",   nn_mrr,               b.get("n_nn_mrr"))
    line("pool MRR",  rm["mrr"],            b.get("pool_mrr"))
    line("AUC",       pm["auc"],            b.get("auc"))
    line("P@1",       pm["precision_at_1"], b.get("p_at_1"))
    line("pool R@1",  rm["recall_at_k"][1], b.get("recall@1"))
    line("pool R@5",  rm["recall_at_k"][5], b.get("recall@5"))
    line("pool R@10", rm["recall_at_k"][10],b.get("recall@10"))
    print("=" * 82)
    if b.get("n_nn_mrr"):
        pct = 100 * (nn_mrr - b["n_nn_mrr"]) / b["n_nn_mrr"]
        pct_pool = 100 * (rm["mrr"] - b["pool_mrr"]) / b["pool_mrr"]
        ok = "✓✓" if (pct >= 10 or pct_pool >= 10) else ("✓" if pct > 0 else "✗")
        print(f"  {ok}  N×N MRR {pct:+.1f}%  |  pool MRR {pct_pool:+.1f}%  vs baseline "
              f"(target ≥ +10%)")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="aep_causal")
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evaluate(Path(args.run_dir), args.dataset, args.pool_size, args.n_negatives, seed=args.seed)


if __name__ == "__main__":
    main()
