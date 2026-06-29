"""Bi-encoder training entry point.

Trains a two-tower bi-encoder on any registered dataset (see
`followup_data`) using InfoNCE contrastive loss with in-batch negatives.

The script is dataset-agnostic — it only reads `PairExample.anchor` and
`PairExample.positive`, which every loader produces. Pick the dataset
with `--dataset NAME` and (optionally) the val split with `--val-split`.
If no val split exists, hold out 10% of train as validation.

Example:
    uv run train-biencoder \\
        --dataset aep_causal --val-split val \\
        --backbone bge-small --epochs 15 --batch-size 64 \\
        --out-dir runs/aep_causal_bge

CLI flags are documented via --help.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset

from followup_data import default_root, load
from followup_data.base import DatasetNotDownloaded

from ..config import DEFAULT_MODEL_KEY, MODEL_CONFIGS, get_model_config
from ..losses import InfoNCELoss
from ..model import BiEncoder, get_tokenizer
from .data import PairDataset, make_collate_fn
from .validation import validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(dataset_name: str, split: str, root: Path | None):
    """Load a dataset split; ensure it's downloaded."""
    effective_root = root if root is not None else default_root()
    ds = load(dataset_name, root=effective_root, split=split)
    try:
        if not ds.is_downloaded():
            ds.download()
    except DatasetNotDownloaded as e:
        raise SystemExit(f"dataset not downloaded: {e}")
    return ds


def split_train_val(
    train_ds: PairDataset, val_frac: float, seed: int
) -> tuple[Subset, Subset]:
    """Random hold-out: split a PairDataset into (train, val) by indices."""
    n = len(train_ds)
    n_val = max(1, int(n * val_frac))
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(train_ds, train_idx), Subset(train_ds, val_idx)


def save_checkpoint(
    path: Path,
    model: BiEncoder,
    loss_fn: InfoNCELoss,
    optimizer,
    epoch: int,
    metrics: dict,
    cfg: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "cfg": cfg,
        },
        path,
    )


def maybe_resume(path: Path, model: BiEncoder, loss_fn: InfoNCELoss, optimizer) -> int:
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("epoch", 0)) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a bi-encoder on any registered follow-up dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- Data ----
    ap.add_argument("--dataset", required=True,
                    help="Registered dataset name (e.g. aep_causal, qrecc, clariq).")
    ap.add_argument("--train-split", default="train",
                    help="Split name to use for training.")
    ap.add_argument("--val-split", default=None,
                    help="Split name to use for validation. "
                         "If omitted, holds out --val-frac of train.")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="If --val-split is omitted, fraction of train held out for val.")
    ap.add_argument("--root", default=None,
                    help="Override data root (default: repo-provided).")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Cap on train samples (useful for smoke tests).")
    ap.add_argument("--max-val-samples", type=int, default=None,
                    help="Cap on val samples (useful for smoke tests).")
    # ---- Model ----
    ap.add_argument("--backbone", default=DEFAULT_MODEL_KEY,
                    choices=list(MODEL_CONFIGS),
                    help="Backbone key from biencoder.config.MODEL_CONFIGS.")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Override max sequence length (defaults to backbone config).")
    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override batch size (defaults to backbone's suggested_batch_size).")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Encoder learning rate.")
    ap.add_argument("--temperature-lr", type=float, default=1e-3,
                    help="Higher LR for the (learnable) InfoNCE temperature.")
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--fixed-temperature", action="store_true",
                    help="Treat temperature as a fixed (non-learnable) buffer.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1,
                    help="Fraction of total steps used for linear warmup.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # ---- I/O ----
    ap.add_argument("--out-dir", default="runs/biencoder",
                    help="Where to save checkpoints and training_history.json.")
    ap.add_argument("--run-name", default=None,
                    help="If set, checkpoints go under <out-dir>/<run-name>.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <out-dir>/biencoder_latest.pt if present.")
    # ---- Early stopping ----
    ap.add_argument("--early-stopping-patience", type=int, default=5,
                    help="Stop if val MRR does not improve for N consecutive epochs.")

    args = ap.parse_args()

    # ---- seed ----
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- backbone config ----
    mcfg = get_model_config(args.backbone)
    max_seq_length = args.max_seq_length or mcfg["max_seq_length"]
    batch_size = args.batch_size or mcfg["suggested_batch_size"]

    device = auto_device()
    print(f"[device] {device}")

    # ---- data ----
    root = Path(args.root) if args.root else None
    print(f"[data] loading dataset={args.dataset!r} split={args.train_split!r}")
    train_loader_ds = load_split(args.dataset, args.train_split, root)
    train_pairs = PairDataset(train_loader_ds, max_examples=args.max_train_samples)
    print(f"  train pairs: {len(train_pairs)}")

    if args.val_split:
        print(f"[data] loading val dataset={args.dataset!r} split={args.val_split!r}")
        val_loader_ds = load_split(args.dataset, args.val_split, root)
        val_pairs = PairDataset(val_loader_ds, max_examples=args.max_val_samples)
        train_subset = train_pairs
        val_subset = val_pairs
        print(f"  val pairs: {len(val_subset)}")
    else:
        print(f"[data] no --val-split given; holding out {args.val_frac:.0%} of train as val")
        train_subset, val_subset = split_train_val(train_pairs, args.val_frac, args.seed)
        print(f"  train: {len(train_subset)}  val: {len(val_subset)}")

    if len(train_subset) == 0 or len(val_subset) == 0:
        raise SystemExit("empty train or val split; aborting")

    # ---- model + tokenizer ----
    print(f"[model] initializing bi-encoder from {mcfg['model_name']!r}")
    tokenizer = get_tokenizer(mcfg["model_name"])
    model = BiEncoder(
        model_name=mcfg["model_name"],
        pooling_strategy=mcfg["pooling"],
        anchor_prefix=mcfg["anchor_prefix"],
        positive_prefix=mcfg["positive_prefix"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {total_params:,} total, {trainable_params:,} trainable")

    collate = make_collate_fn(
        tokenizer,
        max_length=max_seq_length,
        anchor_prefix=mcfg["anchor_prefix"],
        positive_prefix=mcfg["positive_prefix"],
    )
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        drop_last=False, collate_fn=collate, num_workers=args.num_workers,
    )

    # ---- loss + optimizer ----
    loss_fn = InfoNCELoss(
        temperature_init=args.temperature_init,
        learnable=not args.fixed_temperature,
    ).to(device)

    encoder_params = list(model.anchor_encoder.parameters()) + list(model.positive_encoder.parameters())
    param_groups: list[dict] = [
        {"params": encoder_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ]
    temp_lrs = []
    if loss_fn.learnable:
        param_groups.append(
            {"params": [loss_fn.log_temperature], "lr": args.temperature_lr, "weight_decay": 0.0}
        )
        temp_lrs.append(args.temperature_lr)
    optimizer = AdamW(param_groups)

    total_steps = len(train_loader) * args.epochs
    max_lrs = [args.lr] + temp_lrs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

    # ---- run directory ----
    run_dir = Path(args.out_dir)
    if args.run_name:
        run_dir = run_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[io] run_dir={run_dir}")

    # Persist the resolved config (so the run is self-describing).
    cfg = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "val_frac": args.val_frac if not args.val_split else None,
        "backbone": args.backbone,
        "model_name": mcfg["model_name"],
        "pooling": mcfg["pooling"],
        "anchor_prefix": mcfg["anchor_prefix"],
        "positive_prefix": mcfg["positive_prefix"],
        "max_seq_length": max_seq_length,
        "embedding_dim": mcfg["embedding_dim"],
        "batch_size": batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "temperature_init": args.temperature_init,
        "temperature_lr": args.temperature_lr,
        "fixed_temperature": args.fixed_temperature,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": str(device),
        "n_train_pairs": len(train_subset),
        "n_val_pairs": len(val_subset),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- resume ----
    latest_path = run_dir / "biencoder_latest.pt"
    best_path = run_dir / "biencoder_best.pt"
    start_epoch = 0
    if args.resume and latest_path.exists():
        start_epoch = maybe_resume(latest_path, model, loss_fn, optimizer)
        print(f"[resume] continuing from epoch {start_epoch}")

    # ---- training loop ----
    history: list[dict] = []
    best_mrr = -1.0
    patience = 0

    print()
    print("=" * 70)
    print(f"training: {args.epochs} epochs, batch_size={batch_size}, lr={args.lr}")
    print(f"steps/epoch: {len(train_loader)}, total: {total_steps}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        last_stats: dict = {}

        for a_ids, a_mask, p_ids, p_mask in train_loader:
            a_ids = a_ids.to(device, non_blocking=True)
            a_mask = a_mask.to(device, non_blocking=True)
            p_ids = p_ids.to(device, non_blocking=True)
            p_mask = p_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            a, p = model(a_ids, a_mask, p_ids, p_mask)
            loss, stats = loss_fn(a, p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            last_stats = stats

        train_loss = running_loss / max(n_batches, 1)

        # Validation
        val_metrics = validate(model, val_loader, loss_fn, device)

        epoch_time = time.time() - t0
        print(
            f"\nepoch {epoch + 1}/{args.epochs} ({epoch_time:.0f}s)"
            f"\n  train loss={train_loss:.4f}  τ={last_stats.get('temperature', 0):.4f}"
            f"  pos_sim={last_stats.get('mean_pos_sim', 0):.3f}"
            f"  neg_sim={last_stats.get('mean_neg_sim', 0):.3f}"
            f"  gap={last_stats.get('sim_gap', 0):.3f}"
            f"\n  val   loss={val_metrics['loss']:.4f}"
            f"  MRR={val_metrics['mrr']:.4f}"
            f"  R@1={val_metrics.get('recall@1', 0):.3f}"
            f"  R@5={val_metrics.get('recall@5', 0):.3f}"
            f"  R@10={val_metrics.get('recall@10', 0):.3f}"
            f"  med.rank={val_metrics['median_rank']:.0f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_stats": last_stats,
                "val_metrics": val_metrics,
                "epoch_time": epoch_time,
            }
        )

        save_checkpoint(latest_path, model, loss_fn, optimizer, epoch, val_metrics, cfg)

        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            patience = 0
            save_checkpoint(best_path, model, loss_fn, optimizer, epoch, val_metrics, cfg)
            print(f"  -> new best MRR: {best_mrr:.4f} (saved)")
        else:
            patience += 1
            print(f"  -> no improvement ({patience}/{args.early_stopping_patience} patience)")

        if patience >= args.early_stopping_patience:
            print(f"\n[early stopping] stopping at epoch {epoch + 1}")
            break

    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print()
    print("=" * 70)
    print(f"done. best val MRR={best_mrr:.4f}")
    print(f"  best:  {best_path}")
    print(f"  latest:{latest_path}")
    print(f"  history: {run_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
