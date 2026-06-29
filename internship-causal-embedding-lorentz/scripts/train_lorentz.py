"""Train Lorentz encoder (Stage 3 v2): learnable temperature, bidirectional NCE, prefix support.

Usage:
  # e5-large-v2 (strongest, recommended)
  python scripts/train_lorentz.py --dataset aep_causal --backbone intfloat/e5-large-v2 \
      --pooling mean --anchor-prefix "query: " --positive-prefix "passage: " \
      --batch-size 64 --epochs 5

  # e5-base-v2 (faster, still strong)
  python scripts/train_lorentz.py --dataset all --backbone intfloat/e5-base-v2 \
      --pooling mean --anchor-prefix "query: " --positive-prefix "passage: " --batch-size 128

  # BERT baseline (for comparison)
  python scripts/train_lorentz.py --dataset aep_causal --backbone google-bert/bert-base-uncased \
      --pooling cls --batch-size 128 --bidirectional
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finetune_eval.datasets import list_datasets   # noqa
from lorentz_enc.train.stage3_contrastive import run_stage3  # noqa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal", help="Dataset name or 'all'.")
    ap.add_argument("--results-dir", default=str(ROOT / "lorentz_enc_results"))
    ap.add_argument("--backbone", default="intfloat/e5-base-v2")
    ap.add_argument("--anchor-prefix", default="query: ")
    ap.add_argument("--positive-prefix", default="passage: ")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--space-dim", type=int, default=0, help="0 = use backbone hidden size (no bottleneck)")
    ap.add_argument("--time-dim", type=int, default=20)
    ap.add_argument("--space-hidden", type=int, default=768)
    ap.add_argument("--time-hidden", type=int, default=256)
    ap.add_argument("--time-layers", type=int, default=2,
                    help="Number of residual blocks in the time head (deeper = more expressive).")
    ap.add_argument("--max-negatives", type=int, default=None,
                    help="Cap hard negatives per anchor (e.g. 4 to match baseline recipe).")
    ap.add_argument("--select-on", default="mrr", choices=["mrr", "combined"],
                    help="Checkpoint selection metric: space-only MRR or combined space+time MRR.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--grad-accumulation", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature-lr", type=float, default=1e-3)
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--tau-score", type=float, default=10.0)
    ap.add_argument("--nce-margin", type=float, default=0.02)
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--lambda-ord", type=float, default=0.0)
    ap.add_argument("--lambda-reg", type=float, default=0.0)
    ap.add_argument("--lambda-dir", type=float, default=0.0,
                    help="Weight on the directional margin loss (push score(a->b) > score(b->a)). "
                         "Trains only the time head; space cosine is detached so retrieval is protected.")
    ap.add_argument("--dir-margin", type=float, default=0.2,
                    help="Margin for the directional loss.")
    ap.add_argument("--dir-beta", type=float, default=1.0,
                    help="Weight of the time term in the directional/combined score.")
    ap.add_argument("--no-bidirectional", action="store_true")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-val-samples", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-every-n-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-neg-subdir", default=None,
                    help="Override subdir name (default: dataset name). Useful when hard negs "
                         "were mined with a different model.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    targets = list_datasets() if args.dataset == "all" else [args.dataset]

    for ds in targets:
        subdir = args.hard_neg_subdir or ds
        hard_neg_dir = results_dir / subdir
        if not (hard_neg_dir / "hard_negatives.npy").exists():
            print(f"[{ds}] skip: no hard_negatives.npy at {hard_neg_dir}. "
                  f"Run mine_lorentz_hard_negatives.py first.")
            continue

        print(f"\n{'='*70}")
        print(f"Training Lorentz encoder v2 on {ds}")
        print(f"{'='*70}")

        run_stage3(
            dataset=ds,
            hard_neg_dir=str(hard_neg_dir),
            out_dir=str(hard_neg_dir),
            backbone=args.backbone,
            anchor_prefix=args.anchor_prefix,
            positive_prefix=args.positive_prefix,
            space_dim=args.space_dim,
            time_dim=args.time_dim,
            space_hidden=args.space_hidden,
            time_hidden=args.time_hidden,
            time_layers=args.time_layers,
            max_negatives=args.max_negatives,
            select_on=args.select_on,
            dropout=args.dropout,
            pooling=args.pooling,
            max_length=args.max_length,
            batch_size=args.batch_size,
            grad_accumulation=args.grad_accumulation,
            epochs=args.epochs,
            lr=args.lr,
            temperature_lr=args.temperature_lr,
            temperature_init=args.temperature_init,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_clip=args.grad_clip,
            alpha=args.alpha,
            beta=args.beta,
            tau_score=args.tau_score,
            nce_margin=args.nce_margin,
            lambda_nce=args.lambda_nce,
            lambda_ord=args.lambda_ord,
            lambda_dir=args.lambda_dir,
            dir_margin=args.dir_margin,
            dir_beta=args.dir_beta,
            bidirectional=not args.no_bidirectional,
            bf16=not args.no_bf16,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            val_frac=args.val_frac,
            max_val_samples=args.max_val_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            val_every_n_steps=args.val_every_n_steps,
        )


if __name__ == "__main__":
    main()
