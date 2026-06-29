"""Evaluate a trained Hybrid Cause→Effect GNN checkpoint.

Computes the same metrics as finetune_eval/evaluate_finetune.py so that
results are directly comparable to the two-tower BERT baseline:
  * AUC and Precision@1  (anchor vs 4 random negatives)
  * MRR, Recall@K, Hit@K, mean/median rank  (pool of up to 1000 candidates)

The comparison table is printed alongside the BERT two-tower scores from
finetune_eval/results/<dataset>/eval.json.

Memory guard:
  Checks CPU RAM before heavy steps; aborts if < 4 GB available.

Usage:
  PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  $PYTHON gnn/evaluate_gnn.py --dataset aep_causal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gnn.model import HybridCauseEffectGNN    # noqa: E402
from gnn.train_gnn import (                   # noqa: E402
    load_pairs_jsonl, build_graph, encode_pair_texts,
    SPLITS_CFG, DATA_DIR, check_memory,
)
from evaluation_6.metrics_extra import (      # noqa: E402
    auc_precision_at_1_with_random_negatives,
    random_pool_retrieval_metrics,
)


@torch.no_grad()
def get_embeddings(
    model: HybridCauseEffectGNN,
    pairs: list[tuple[str, str, dict]],
    anchor_embs: torch.Tensor,    # [N, text_dim]
    positive_embs: torch.Tensor,  # [N, text_dim]
    id2idx: dict[str, int],
    x_text: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cause_emb, effect_emb) arrays of shape [N_valid, out_dim]."""
    model.eval()
    all_h = model.encode_graph(x_text.to(device), edge_index.to(device))

    cause_list, effect_list = [], []
    for i, (_, _, meta) in enumerate(pairs):
        sid, tid = meta["source_doc_id"], meta["target_doc_id"]
        if sid not in id2idx or tid not in id2idx:
            continue
        h_c = all_h[id2idx[sid]].unsqueeze(0)   # [1, gnn_dim]
        h_e = all_h[id2idx[tid]].unsqueeze(0)   # [1, gnn_dim]
        a_t = anchor_embs[i].unsqueeze(0).to(device)
        p_t = positive_embs[i].unsqueeze(0).to(device)

        cause_list.append(model.fuse_cause(a_t, h_c).cpu())
        effect_list.append(model.fuse_effect(p_t, h_e).cpu())

    if not cause_list:
        return np.zeros((0, model.out_dim)), np.zeros((0, model.out_dim))

    a_emb = torch.cat(cause_list, dim=0).numpy()
    p_emb = torch.cat(effect_list, dim=0).numpy()
    return a_emb, p_emb


def evaluate_checkpoint(
    ckpt_path: Path,
    dataset: str,
    pool_size: int = 1000,
    n_negatives: int = 4,
    k_values: list[int] = (1, 3, 5, 10),
    seed: int = 0,
) -> dict:
    check_memory("eval start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[evaluate_gnn]  {ckpt_path.name}  device={device}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    id2idx: dict[str, int] = ckpt["id2idx"]

    model = HybridCauseEffectGNN(
        text_dim=cfg["text_dim"],
        gnn_dim=cfg["gnn_dim"],
        out_dim=cfg["out_dim"],
        num_gnn_layers=cfg["num_gnn_layers"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded  gnn_dim={cfg['gnn_dim']}  out_dim={cfg['out_dim']}  "
          f"layers={cfg['num_gnn_layers']}")

    results_dir = ckpt_path.parent
    doc_cache = results_dir / "doc_embeddings.npy"
    if not doc_cache.exists():
        raise FileNotFoundError(f"doc embeddings cache missing: {doc_cache}")
    check_memory("before load embeddings")
    x_text = torch.from_numpy(np.load(doc_cache)).float()
    print(f"  doc embeddings: {x_text.shape}")
    check_memory("after load embeddings")

    data_dir = DATA_DIR / dataset
    cfg_splits = SPLITS_CFG[dataset]
    train_pairs = load_pairs_jsonl(data_dir / cfg_splits["train"])
    edge_index, _ = build_graph(train_pairs, id2idx)
    edge_index = edge_index.to(device)

    test_pairs = load_pairs_jsonl(data_dir / cfg_splits["test"])
    print(f"  test pairs: {len(test_pairs)}")

    # Encode test pair texts with the same encoder used for training
    test_anchors   = [a for a, _, _ in test_pairs]
    test_positives = [p for _, p, _ in test_pairs]
    test_a_cache = results_dir / "test_anchor_embs.npy"
    test_p_cache = results_dir / "test_positive_embs.npy"

    t0 = time.time()
    test_a_embs, test_p_embs = encode_pair_texts(
        test_anchors, test_positives,
        test_a_cache, test_p_cache,
        model_name=cfg["encoder_model"],
    )
    enc_time = time.time() - t0
    print(f"  encoded test in {enc_time:.1f}s")
    check_memory("after encode test")

    a_emb, p_emb = get_embeddings(
        model, test_pairs,
        test_a_embs, test_p_embs,
        id2idx, x_text, edge_index, device,
    )
    print(f"  a_emb={a_emb.shape}  p_emb={p_emb.shape}")

    if len(a_emb) == 0:
        print("  WARNING: no test pairs resolved to known docs")
        return {}

    pair_metrics = auc_precision_at_1_with_random_negatives(
        a_emb, p_emb, n_negatives=n_negatives, seed=seed,
    )
    retrieval_metrics = random_pool_retrieval_metrics(
        a_emb, p_emb, pool_size=pool_size, k_values=k_values, seed=seed,
    )

    row = {
        "dataset":                dataset,
        "checkpoint":             str(ckpt_path),
        "n_test_pairs":           len(a_emb),
        "n_candidates_per_query": retrieval_metrics["n_candidates_per_query"],
        "encoding_seconds":       enc_time,
        "auc":                    pair_metrics["auc"],
        "precision_at_1":         pair_metrics["precision_at_1"],
        "mrr":                    retrieval_metrics["mrr"],
        "mean_rank":              retrieval_metrics["mean_rank"],
        "median_rank":            retrieval_metrics["median_rank"],
        "recall_at_k":            retrieval_metrics["recall_at_k"],
        "hit_at_k":               retrieval_metrics["hit_at_k"],
    }
    out_path = results_dir / "gnn_eval.json"
    out_path.write_text(json.dumps(row, indent=2))
    print(f"  results → {out_path}")
    return row


def print_comparison(gnn_row: dict, baseline_path: Path | None) -> None:
    print()
    print("=" * 80)
    print(f"{'Metric':<22} {'GNN (hybrid)':>14} {'BERT two-tower':>16}  {'Δ':>8}")
    print("-" * 80)

    baseline: dict = {}
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())

    def _get(d, *keys):
        for k in keys:
            if d is None:
                return None
            d = d.get(k) if isinstance(d, dict) else None
        return d

    def row(name, gnn_val, base_val):
        if base_val is not None:
            delta = gnn_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<20} {gnn_val:>14.4f} {base_val:>16.4f}  {sign}{delta:>7.4f}")
        else:
            print(f"  {name:<20} {gnn_val:>14.4f} {'—':>16}")

    gnn_rk = gnn_row.get("recall_at_k", {})
    base_rk_raw = baseline.get("recall_at_k", {})
    # recall_at_k keys may be int or str depending on source
    def rk(d, k):
        return d.get(k) or d.get(str(k))

    row("AUC",       gnn_row.get("auc", 0),              baseline.get("auc"))
    row("P@1",       gnn_row.get("precision_at_1", 0),   baseline.get("precision_at_1"))
    row("MRR",       gnn_row.get("mrr", 0),              baseline.get("mrr"))
    row("Recall@1",  rk(gnn_rk, 1) or 0,                rk(base_rk_raw, 1))
    row("Recall@3",  rk(gnn_rk, 3) or 0,                rk(base_rk_raw, 3))
    row("Recall@5",  rk(gnn_rk, 5) or 0,                rk(base_rk_raw, 5))
    row("Recall@10", rk(gnn_rk, 10) or 0,               rk(base_rk_raw, 10))
    row("Mean rank", gnn_row.get("mean_rank", 0),        baseline.get("mean_rank"))
    row("Median rank",gnn_row.get("median_rank", 0),     baseline.get("median_rank"))
    print("=" * 80)

    mrr_g = gnn_row.get("mrr", 0)
    mrr_b = baseline.get("mrr", 0)
    if mrr_b:
        if mrr_g > mrr_b:
            print(f"\n  ✓ GNN hybrid beats BERT two-tower by +{mrr_g - mrr_b:.4f} MRR")
        else:
            print(f"\n  ✗ GNN hybrid trails BERT two-tower by {mrr_b - mrr_g:.4f} MRR")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal")
    ap.add_argument("--results-dir", default=str(ROOT / "gnn/results"))
    ap.add_argument("--baseline-eval", default=None)
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) / args.dataset
    ckpt_path = results_dir / "gnn_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint found: {ckpt_path}\nRun train_gnn.py first.")

    gnn_row = evaluate_checkpoint(
        ckpt_path=ckpt_path,
        dataset=args.dataset,
        pool_size=args.pool_size,
        n_negatives=args.n_negatives,
        k_values=sorted(set(args.k)),
        seed=args.seed,
    )

    baseline_path = (Path(args.baseline_eval) if args.baseline_eval
                     else ROOT / "finetune_eval/results" / args.dataset / "eval.json")
    print_comparison(gnn_row, baseline_path)


if __name__ == "__main__":
    main()
