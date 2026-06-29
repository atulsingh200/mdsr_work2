"""Evaluate a fine-tuned Hybrid GNN checkpoint.

Same metrics as finetune_eval/evaluate_finetune.py for direct comparison.

Usage:
  PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  $PYTHON gnn/evaluate_gnn_finetune.py --dataset aep_causal
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

from gnn.model import HybridCauseEffectGNN             # noqa: E402
from gnn.train_gnn import (                            # noqa: E402
    load_pairs_jsonl, build_graph, SPLITS_CFG, DATA_DIR, check_memory,
)
from gnn.train_gnn_finetune import TextEncoder         # noqa: E402
from evaluation_6.metrics_extra import (               # noqa: E402
    auc_precision_at_1_with_random_negatives,
    random_pool_retrieval_metrics,
)


def evaluate_finetune_checkpoint(
    ckpt_path: Path,
    dataset: str,
    pool_size: int = 1000,
    n_negatives: int = 4,
    k_values: list[int] = (1, 3, 5, 10),
    seed: int = 0,
) -> dict:
    check_memory("eval start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[evaluate_gnn_finetune]  {ckpt_path.name}  device={device}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    id2idx = ckpt["id2idx"]

    encoder = TextEncoder(cfg["encoder_model"], max_length=256).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    gnn_model = HybridCauseEffectGNN(
        text_dim=cfg["text_dim"], gnn_dim=cfg["gnn_dim"],
        out_dim=cfg["out_dim"], num_gnn_layers=cfg["num_gnn_layers"],
        heads=cfg["heads"], dropout=cfg["dropout"],
    ).to(device)
    gnn_model.load_state_dict(ckpt["gnn_state"])
    gnn_model.eval()

    data_dir = DATA_DIR / dataset
    cfg_s = SPLITS_CFG[dataset]
    train_pairs = load_pairs_jsonl(data_dir / cfg_s["train"])
    edge_index, _ = build_graph(train_pairs, id2idx)
    edge_index = edge_index.to(device)

    # Rebuild doc embeddings with fine-tuned encoder
    from gnn.train_gnn import build_doc_registry
    _, _, doc_texts = build_doc_registry(train_pairs,
                                          load_pairs_jsonl(data_dir / cfg_s["val"]) +
                                          load_pairs_jsonl(data_dir / cfg_s["test"]))
    print(f"  re-encoding {len(doc_texts)} docs with fine-tuned encoder …")
    check_memory("before doc encode")
    x_text = torch.from_numpy(
        encoder.encode_texts(doc_texts, batch_size=64, device=device)
    ).float()
    check_memory("after doc encode")

    test_pairs = load_pairs_jsonl(data_dir / cfg_s["test"])
    anchors    = [a for a, _, _ in test_pairs]
    positives  = [p for _, p, _ in test_pairs]

    t0 = time.time()
    all_h = gnn_model.encode_graph(x_text.to(device), edge_index)
    with torch.no_grad():
        a_emb_enc = torch.from_numpy(encoder.encode_texts(anchors, device=device)).float()
        p_emb_enc = torch.from_numpy(encoder.encode_texts(positives, device=device)).float()
    enc_time = time.time() - t0

    cause_list, effect_list = [], []
    with torch.no_grad():
        for i, (_, _, meta) in enumerate(test_pairs):
            sid, tid = meta["source_doc_id"], meta["target_doc_id"]
            if sid not in id2idx or tid not in id2idx: continue
            h_c = all_h[id2idx[sid]].unsqueeze(0)
            h_e = all_h[id2idx[tid]].unsqueeze(0)
            cause_list.append(gnn_model.fuse_cause(a_emb_enc[i:i+1].to(device), h_c).cpu())
            effect_list.append(gnn_model.fuse_effect(p_emb_enc[i:i+1].to(device), h_e).cpu())

    a_emb = torch.cat(cause_list).numpy()
    p_emb = torch.cat(effect_list).numpy()
    print(f"  a_emb={a_emb.shape}  encoded in {enc_time:.1f}s")

    pair_m = auc_precision_at_1_with_random_negatives(a_emb, p_emb, n_negatives, seed)
    retr_m = random_pool_retrieval_metrics(a_emb, p_emb, pool_size, k_values, seed)

    row = {
        "dataset": dataset, "n_test": len(a_emb),
        "n_candidates": retr_m["n_candidates_per_query"],
        "encoding_seconds": enc_time,
        "auc": pair_m["auc"], "precision_at_1": pair_m["precision_at_1"],
        "mrr": retr_m["mrr"], "mean_rank": retr_m["mean_rank"],
        "median_rank": retr_m["median_rank"],
        "recall_at_k": retr_m["recall_at_k"],
        "hit_at_k": retr_m["hit_at_k"],
    }
    out = ckpt_path.parent / "gnn_ft_eval.json"
    out.write_text(json.dumps(row, indent=2))
    print(f"  results → {out}")

    # Print comparison
    baseline_path = ROOT / "finetune_eval/results" / dataset / "eval.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    print()
    print("=" * 80)
    print(f"{'Metric':<22} {'GNN fine-tuned':>14} {'BERT two-tower':>16}  {'Δ':>8}")
    print("-" * 80)
    def row_p(name, gv, bv):
        if bv is not None:
            sign = "+" if gv - bv >= 0 else ""
            print(f"  {name:<20} {gv:>14.4f} {bv:>16.4f}  {sign}{gv-bv:>7.4f}")
        else:
            print(f"  {name:<20} {gv:>14.4f}")
    rk = row["recall_at_k"]
    brk = baseline.get("recall_at_k", {})
    def bget(k): return brk.get(k) or brk.get(str(k))
    row_p("AUC",       row["auc"],            baseline.get("auc"))
    row_p("P@1",       row["precision_at_1"], baseline.get("precision_at_1"))
    row_p("MRR",       row["mrr"],            baseline.get("mrr"))
    row_p("Recall@1",  rk.get(1, 0),          bget(1))
    row_p("Recall@3",  rk.get(3, 0),          bget(3))
    row_p("Recall@5",  rk.get(5, 0),          bget(5))
    row_p("Recall@10", rk.get(10, 0),         bget(10))
    row_p("Mean rank", row["mean_rank"],       baseline.get("mean_rank"))
    print("=" * 80)
    if baseline.get("mrr"):
        gap = row["mrr"] - baseline["mrr"]
        sym = "✓" if gap >= 0 else "✗"
        print(f"  {sym} GNN fine-tuned {'beats' if gap>=0 else 'trails'} BERT two-tower by {abs(gap):.4f} MRR")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aep_causal")
    ap.add_argument("--results-dir", default=str(ROOT / "gnn/results"))
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) / f"{args.dataset}_finetune"
    ckpt_path = results_dir / "gnn_ft_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint: {ckpt_path}\nRun train_gnn_finetune.py first.")

    evaluate_finetune_checkpoint(
        ckpt_path, args.dataset,
        args.pool_size, args.n_negatives,
        sorted(set(args.k)), args.seed,
    )


if __name__ == "__main__":
    main()
