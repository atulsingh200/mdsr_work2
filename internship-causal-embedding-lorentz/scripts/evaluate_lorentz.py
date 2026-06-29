"""Evaluate Lorentz encoder checkpoints on test splits.

Works with both BERT and e5 checkpoints. Reads prefix convention from checkpoint config.

Usage:
  python scripts/evaluate_lorentz.py --dataset all
  python scripts/evaluate_lorentz.py --dataset aep_causal --results-dir lorentz_enc_results_e5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finetune_eval.datasets import list_datasets  # noqa
from lorentz_enc.eval.retrieval import evaluate_lorentz  # noqa

BASELINE = {
    "aep_causal":   {"mrr": 0.3525, "auc": 0.9712},
    "followupqg":   {"mrr": 0.5844, "auc": 0.9827},
    "multiwoz_v24": {"mrr": 0.2493, "auc": 0.9526},
    "qrecc":        {"mrr": 0.2758, "auc": 0.9227},
    "workflow":     {"mrr": 0.5500, "auc": 0.9662},
}

WORKFLOW_TEST_OVERRIDE = str(
    Path("/mnt/localssd/internship-causal-embedding-merge-followup") /
    "data_6" / "workflow" / "test_pairs.jsonl"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--results-dir", default=str(ROOT / "lorentz_enc_results"))
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    targets = list_datasets() if args.dataset == "all" else [args.dataset]
    k_values = sorted(set(args.k))

    rows = []
    for ds in targets:
        ckpt_path = results_dir / ds / "lorentz_best.pt"
        if not ckpt_path.exists():
            print(f"[{ds}] skip: no checkpoint at {ckpt_path}")
            rows.append({"dataset": ds, "error": "no checkpoint"})
            continue
        # workflow test data lives in the other project
        test_override = WORKFLOW_TEST_OVERRIDE if ds == "workflow" else None
        if test_override and not Path(test_override).exists():
            test_override = None
        try:
            row = evaluate_lorentz(
                dataset=ds, ckpt_path=str(ckpt_path),
                pool_size=args.pool_size, n_negatives=args.n_negatives,
                k_values=k_values, batch_size=args.batch_size, seed=args.seed,
                test_path_override=test_override,
            )
            rows.append(row)
            (results_dir / ds / "lorentz_eval.json").write_text(json.dumps(row, indent=2))
        except Exception as e:
            print(f"[{ds}] ERROR: {e}")
            import traceback; traceback.print_exc()
            rows.append({"dataset": ds, "error": str(e)})

    # Table
    print("\n" + "=" * 105)
    print(f"{'dataset':<14} {'N':>6}  {'MRR':>7}  {'Δ%':>7}  {'AUC':>7}  "
          f"{'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'median':>7}  Result")
    print("-" * 105)
    wins = 0
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<14}  ERROR: {r['error']}")
            continue
        ds = r["dataset"]
        b = BASELINE.get(ds, {})
        b_mrr = b.get("mrr", 0)
        delta = (r["mrr"] - b_mrr) / max(b_mrr, 1e-9) * 100
        win = delta >= 10.0
        gt0 = delta > 0
        if win:
            wins += 1
        marker = "✓≥10%" if win else ("↑" if gt0 else "✗")
        r10 = r["recall_at_k"].get(10, 0)
        r5 = r["recall_at_k"].get(5, 0)
        r1 = r["recall_at_k"].get(1, 0)
        print(f"{ds:<14} {r['n_test_pairs']:>6d}  {r['mrr']:>7.4f}  {delta:>+7.1f}%  "
              f"{r['auc']:>7.4f}  {r1:>7.4f}  {r5:>7.4f}  {r10:>7.4f}  "
              f"{r['median_rank']:>7.1f}  {marker}")

    print("-" * 105)
    valid = [r for r in rows if "mrr" in r]
    n_10pct = sum(1 for r in valid if (r["mrr"] - BASELINE.get(r["dataset"],{}).get("mrr",0)) /
                  max(BASELINE.get(r["dataset"],{}).get("mrr",0), 1e-9) * 100 >= 10.0)
    print(f"\n≥10% improvement: {n_10pct}/{len(valid)} datasets  |  "
          f"any improvement: {wins} wins  |  "
          f"target: {len(valid)}/{len(valid)} at ≥10%")

    # Save
    out_path = Path(args.out) if args.out else results_dir / "lorentz_eval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": rows, "baseline": BASELINE}, indent=2))
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
