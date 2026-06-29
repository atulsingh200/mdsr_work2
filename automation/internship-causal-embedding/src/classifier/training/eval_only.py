#!/usr/bin/env python3
"""
Evaluate a trained directional classifier checkpoint on the test split.

No training happens here. The checkpoint (best.pt / final.pt) carries its own
`cfg`, so the model is rebuilt exactly as trained, weights are loaded, and the
test set is scored with the same `evaluate()` used at the end of training.

Reproduces the numbers in the run's README / test_metrics.json:
  overall acc / auc / f1 / loss + per-tier-pair accuracy.

Invoke (from repo root):
  python3 -m src.classifier.training.eval_only \
      --ckpt runs/classifier/best_mlp_bge_small/best.pt \
      --data-dir data/aep_causal_classification

Or by file path:
  python3 src/classifier/training/eval_only.py --ckpt <path/to/best.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Support both `python -m src.classifier.training.eval_only` and direct file run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.classifier.model import DirectionalClassifier
    from src.classifier.training.data import Collate, PairDataset
    from src.classifier.training.train import evaluate
else:
    from ..model import DirectionalClassifier
    from .data import Collate, PairDataset
    from .train import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-only for a trained classifier checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to best.pt / final.pt")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Dir with directional_test.jsonl. Default = cfg['data_dir'] from the checkpoint.",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--write-out",
        action="store_true",
        help="Write <split>_metrics_eval.json and <split>_predictions_eval.jsonl "
             "next to the checkpoint (does NOT overwrite the originals).",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"

    ckpt_path = Path(args.ckpt)
    print(f"[eval] loading checkpoint {ckpt_path}", file=sys.stderr)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    base_model = cfg["base_model"]
    head_cfg = cfg["head_cfg"]
    n_tiers = cfg.get("n_tiers", 0) or (15 if cfg.get("tier_aux_weight", 0.0) > 0 else 0)
    max_seq_len = cfg["max_seq_len"]
    data_dir = Path(args.data_dir or cfg["data_dir"])

    print(f"[eval] base_model={base_model}  head_cfg={head_cfg}  "
          f"n_tiers={n_tiers}  max_seq_len={max_seq_len}", file=sys.stderr)
    print(f"[eval] data_dir={data_dir}  split={args.split}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    ds = PairDataset(data_dir / f"directional_{args.split}.jsonl")
    print(f"[eval] {args.split} examples: {len(ds)}", file=sys.stderr)

    collate = Collate(tokenizer, max_seq_len)
    dl = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = DirectionalClassifier(base_model, head_cfg=head_cfg, n_tiers=n_tiers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    metrics, preds_rows = evaluate(
        model, dl, device, use_bf16, loss_fn,
        dataset=ds, return_predictions=True,
    )

    # Per-tier-pair accuracy (same computation as train.py).
    pair_stats: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in preds_rows:
        key = (min(row["tier_1"], row["tier_2"]), max(row["tier_1"], row["tier_2"]))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int((row["prob"] > 0.5) == (row["label"] == 1))
    per_pair = {
        f"T{a}-T{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
        for (a, b), s in sorted(pair_stats.items())
    }

    print("\n==================== TEST RESULTS ====================")
    print(f"  split    : {args.split}")
    print(f"  n        : {metrics['n']}")
    print(f"  accuracy : {metrics['acc']:.4f}")
    print(f"  auc      : {metrics['auc']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  loss     : {metrics['loss']:.4f}")
    print("======================================================\n")

    if args.write_out:
        out_metrics = ckpt_path.parent / f"{args.split}_metrics_eval.json"
        out_metrics.write_text(json.dumps({**metrics, "per_tier_pair": per_pair}, indent=2))
        out_preds = ckpt_path.parent / f"{args.split}_predictions_eval.jsonl"
        with out_preds.open("w") as f:
            for row in preds_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[eval] wrote {out_metrics}\n[eval] wrote {out_preds}", file=sys.stderr)


if __name__ == "__main__":
    main()
