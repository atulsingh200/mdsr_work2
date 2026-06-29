"""Generate final comparison table: Lorentz encoder vs two-tower BERT baseline."""

from __future__ import annotations

import sys
import json
import torch
from pathlib import Path

ROOT_MINE = Path(__file__).resolve().parent.parent
ROOT_ORIG = Path('/mnt/localssd/internship-causal-embedding-merge-followup')

sys.path.insert(0, str(ROOT_ORIG))
sys.path.insert(0, str(ROOT_ORIG / 'src'))
sys.path.insert(0, str(ROOT_MINE))

from finetune_eval.data import load_pairs
from finetune_eval.datasets import split_path
from evaluation_6.metrics_extra import (
    auc_precision_at_1_with_random_negatives,
    random_pool_retrieval_metrics,
)
from lorentz_enc.eval.retrieval import encode_all_fn, auto_device
from biencoder.model import get_tokenizer
from lorentz_enc.model.encoder import LorentzEncoder


BASELINE = {
    "aep_causal":   {"mrr": 0.3525, "auc": 0.9712, "recall_at_k": {"1": 0.1477, "5": 0.4573, "10": 0.5943}},
    "followupqg":   {"mrr": 0.5844, "auc": 0.9827, "recall_at_k": {"1": 0.4630, "5": 0.8303, "10": 0.8822}},
    "multiwoz_v24": {"mrr": 0.2493, "auc": 0.9526, "recall_at_k": {}},
    "qrecc":        {"mrr": 0.2758, "auc": 0.9227, "recall_at_k": {}},
    "workflow":     {"mrr": 0.5500, "auc": 0.9662, "recall_at_k": {}},
}


def evaluate_dataset(dataset, ckpt_path, test_path, which_anchor="anchor", which_positive="positive"):
    device = auto_device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    tokenizer = get_tokenizer(cfg["backbone"])
    model = LorentzEncoder(
        backbone_name=cfg["backbone"],
        space_dim=cfg["space_dim"], time_dim=cfg["time_dim"],
        space_hidden=cfg["space_hidden"], time_hidden=cfg["time_hidden"],
        pooling=cfg["pooling"], tied=cfg.get("tied", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pairs = load_pairs(test_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]

    a_space, _ = encode_all_fn(model, tokenizer, anchors, device, max_length=256, batch_size=128, which=which_anchor)
    p_space, _ = encode_all_fn(model, tokenizer, positives, device, max_length=256, batch_size=128, which=which_positive)

    pair_m = auc_precision_at_1_with_random_negatives(a_space, p_space, n_negatives=4, seed=0)
    ret_m = random_pool_retrieval_metrics(a_space, p_space, pool_size=1000, k_values=[1, 3, 5, 10], seed=0)
    return pair_m, ret_m, len(pairs)


def main():
    results_dir = ROOT_MINE / "lorentz_enc_results"

    configs = [
        ("aep_causal",   results_dir / "aep_causal"   / "lorentz_best.pt", split_path("aep_causal", "test")),
        ("followupqg",   results_dir / "followupqg"   / "lorentz_best.pt", split_path("followupqg", "test")),
        ("multiwoz_v24", results_dir / "multiwoz_v24" / "lorentz_best.pt", split_path("multiwoz_v24", "test")),
        ("qrecc",        results_dir / "qrecc"         / "lorentz_best.pt", split_path("qrecc", "test")),
        ("workflow",     results_dir / "workflow"      / "lorentz_best.pt",
         ROOT_ORIG / "data_6" / "workflow" / "test_pairs.jsonl"),
    ]

    rows = []
    for ds, ckpt, tpath in configs:
        if not ckpt.exists():
            print(f"[{ds}] skip: no checkpoint at {ckpt}")
            continue
        if not Path(tpath).exists():
            print(f"[{ds}] skip: no test file at {tpath}")
            continue
        print(f"\n[{ds}] evaluating …")
        try:
            pair_m, ret_m, n = evaluate_dataset(ds, ckpt, tpath)
            rows.append({
                "dataset": ds, "n": n,
                "auc": pair_m["auc"],
                "mrr": ret_m["mrr"],
                "recall_at_k": ret_m["recall_at_k"],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 100)
    print("FINAL COMPARISON: Lorentz Encoder vs Two-Tower BERT Baseline")
    print("=" * 100)
    print(f"{'Dataset':<14} {'N':>7}  {'Model':<10}  {'MRR':>7}  {'Δ_MRR':>8}  {'AUC':>7}  {'R@10':>7}  Result")
    print("-" * 100)

    total_wins = 0
    for row in rows:
        ds = row["dataset"]
        b = BASELINE.get(ds, {})
        b_mrr = b.get("mrr", 0)
        b_auc = b.get("auc", 0)
        delta_mrr = row["mrr"] - b_mrr
        win = delta_mrr > 0

        if win:
            total_wins += 1

        r10 = row["recall_at_k"].get(10, 0) if isinstance(row["recall_at_k"], dict) else 0

        # Lorentz row
        print(f"{ds:<14} {row['n']:>7d}  {'Lorentz':<10}  {row['mrr']:>7.4f}  {delta_mrr:>+8.4f}  {row['auc']:>7.4f}  {r10:>7.4f}  {'✓ WIN' if win else '✗ LOSS'}")
        # Baseline row
        print(f"{'':14} {'':7}   {'BiEncoder':<10}  {b_mrr:>7.4f}  {'':>8}  {b_auc:>7.4f}")

    print("-" * 100)
    print(f"\nLorentz Encoder wins on {total_wins}/{len(rows)} datasets")
    print(f"\nKey innovation: Semantic hard negatives with K=7 and anchor-similarity filter (threshold=0.85)")
    print(f"  - K=7 vs K=4 (baseline): more hard negatives → harder InfoNCE → better representations")
    print(f"  - Anchor-sim filter: avoids false negatives by skipping candidates whose anchors")
    print(f"    are too similar (cos > 0.85) to the query anchor")
    print(f"  - Untied two-tower encoder with 768-dim (no bottleneck) for direct comparison")
    print(f"  - Time head (20-dim LayerNorm) as auxiliary causal ordering signal")

    # Save
    (results_dir / "final_comparison.json").write_text(json.dumps({
        "lorentz": rows,
        "baseline": BASELINE,
    }, indent=2))
    print(f"\n  Saved to {results_dir / 'final_comparison.json'}")


if __name__ == "__main__":
    main()
