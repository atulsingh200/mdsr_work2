"""Fine-tune a bi-encoder with semantic hard negatives.

Reuses the two-tower `BiEncoder` from `src/biencoder/model.py` and adds
explicit hard negatives via `HardNegativeInfoNCELoss`.

For each (anchor x, positive y) train pair, K hard negatives y* are
pre-mined (semantic-similar OTHER positives — see mine_hard_negatives.py).
The contrastive loss treats each anchor's correct partner as its in-batch
diagonal positive, with B-1 in-batch negatives plus K explicit hard
negatives in the softmax denominator.

Usage:
  .venv/bin/python finetune_eval/train_finetune.py --dataset aep_causal
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from contextlib import nullcontext

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from biencoder.model import BiEncoder, get_tokenizer                       # noqa: E402

from finetune_eval.data import TripleDataset, load_pairs, make_collate_fn   # noqa: E402
from finetune_eval.datasets import SPLITS, list_datasets, split_path        # noqa: E402
from finetune_eval.losses import HardNegativeInfoNCELoss                    # noqa: E402


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model: BiEncoder, loader, device: torch.device, k_values=(1, 3, 5, 10)) -> dict:
    """Encode the val set, compute MRR + Recall@K on the full N×N matrix."""
    model.eval()
    all_a: list[torch.Tensor] = []
    all_p: list[torch.Tensor] = []
    for a_ids, a_mask, p_ids, p_mask, _n_ids, _n_mask in loader:
        a_ids = a_ids.to(device, non_blocking=True)
        a_mask = a_mask.to(device, non_blocking=True)
        p_ids = p_ids.to(device, non_blocking=True)
        p_mask = p_mask.to(device, non_blocking=True)
        a = model.encode_anchor(a_ids, a_mask)
        p = model.encode_positive(p_ids, p_mask)
        all_a.append(a.cpu())
        all_p.append(p.cpu())
    A = torch.cat(all_a, dim=0)
    P = torch.cat(all_p, dim=0)
    sim = A @ P.T
    diag = sim.diag().unsqueeze(1)
    ranks = (sim > diag).sum(dim=1).float() + 1
    out = {
        "n_val":       int(A.size(0)),
        "mrr":         float((1.0 / ranks).mean()),
        "mean_rank":   float(ranks.mean()),
        "median_rank": float(ranks.median()),
    }
    for k in k_values:
        out[f"recall@{k}"] = float((ranks <= k).float().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list_datasets())
    ap.add_argument("--results-dir", default=str(ROOT / "finetune_eval/results"),
                    help="Where mining outputs were saved AND where checkpoints go.")
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased",
                    help="HF model id used as both anchor/positive encoder init.")
    ap.add_argument("--pooling", default="cls", choices=["mean", "cls"])
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature-lr", type=float, default=1e-3)
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="If the dataset has no --val split, hold out this fraction of train.")
    ap.add_argument("--max-val-samples", type=int, default=2000,
                    help="Cap on val pairs (full N²-sim val matrix; keeps RAM bounded).")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--early-stopping-patience", type=int, default=2,
                    help="Number of consecutive validation checks without an "
                         "improvement >= --early-stopping-min-delta before stopping.")
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.001,
                    help="Minimum increase in val MRR that counts as an improvement. "
                         "Smaller deltas are treated as 'no improvement' and count "
                         "toward the patience window.")
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 autocast for encoder forwards (A100-friendly, "
                         "~2x throughput and ~2x effective memory).")
    ap.add_argument("--val-every-n-steps", type=int, default=None,
                    help="Run validation every N training steps in addition to "
                         "the end-of-epoch validation. Useful when 1 epoch is "
                         "very long (e.g. workflow @ 2.1M pairs).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = auto_device()
    results_dir = Path(args.results_dir) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[run dir] {results_dir}")

    # ---- Load pre-mined train pairs + hard negatives ----
    train_pairs_path = results_dir / "train_pairs.jsonl"
    hard_neg_path = results_dir / "hard_negatives.npy"
    if not train_pairs_path.exists() or not hard_neg_path.exists():
        raise SystemExit(
            f"missing mined artifacts. Run mine_hard_negatives.py for {args.dataset} first.\n"
            f"  expected: {train_pairs_path}\n  and:      {hard_neg_path}"
        )

    print(f"[data] train pairs ← {train_pairs_path}")
    train_pairs = load_pairs(train_pairs_path, cap=None)
    hard_negatives = np.load(hard_neg_path)
    print(f"  {len(train_pairs)} train pairs  K={hard_negatives.shape[1]} hard negs/anchor")

    val_path = split_path(args.dataset, "val")
    train_active: list[int] | None = None
    if val_path is not None and val_path.exists():
        val_pairs_all = load_pairs(val_path, cap=None)
        # Cap val for tractable validation each epoch (random subset).
        if args.max_val_samples and len(val_pairs_all) > args.max_val_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(val_pairs_all), size=args.max_val_samples, replace=False)
            idx.sort()
            val_pairs = [val_pairs_all[i] for i in idx]
        else:
            val_pairs = val_pairs_all
        print(f"[data] val pairs ← {val_path}  ({len(val_pairs)} of {len(val_pairs_all)})")
    else:
        # No val split — hold out --val-frac of train *by index*. We keep the
        # full train_pairs (and hard_negatives) intact so the hard-neg matrix
        # indices stay valid; the split is enforced via active_indices below.
        n_val = max(1, int(len(train_pairs) * args.val_frac))
        n_val = min(n_val, args.max_val_samples or n_val)
        rng = random.Random(args.seed)
        idx = list(range(len(train_pairs)))
        rng.shuffle(idx)
        val_idx_list = sorted(idx[:n_val])
        train_active = sorted(idx[n_val:])
        val_pairs = [train_pairs[i] for i in val_idx_list]
        print(f"[data] no val split; held out {len(val_pairs)} pairs from train")

    # ---- Build datasets ----
    train_ds = TripleDataset(train_pairs, hard_negatives, active_indices=train_active)
    val_ds = TripleDataset(val_pairs, hard_negatives=None)  # val: no hard negs needed

    # ---- Model + tokenizer ----
    print(f"[model] BiEncoder({args.backbone!r}, pooling={args.pooling!r})")
    tokenizer = get_tokenizer(args.backbone)
    model = BiEncoder(
        model_name=args.backbone,
        pooling_strategy=args.pooling,
        anchor_prefix="",
        positive_prefix="",
    ).to(device)
    params_total = sum(p.numel() for p in model.parameters())
    print(f"  params: {params_total:,}")

    collate = make_collate_fn(tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Loss + optimizer ----
    loss_fn = HardNegativeInfoNCELoss(
        temperature_init=args.temperature_init, learnable=True,
    ).to(device)
    encoder_params = list(model.anchor_encoder.parameters()) + list(model.positive_encoder.parameters())
    optimizer = AdamW([
        {"params": encoder_params,           "lr": args.lr,             "weight_decay": args.weight_decay},
        {"params": [loss_fn.log_temperature], "lr": args.temperature_lr, "weight_decay": 0.0},
    ])
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.temperature_lr],
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

    # ---- Persist config ----
    cfg = {
        "dataset":              args.dataset,
        "backbone":             args.backbone,
        "pooling":              args.pooling,
        "max_seq_length":       args.max_seq_length,
        "batch_size":           args.batch_size,
        "epochs":               args.epochs,
        "lr":                   args.lr,
        "temperature_lr":       args.temperature_lr,
        "temperature_init":     args.temperature_init,
        "weight_decay":         args.weight_decay,
        "warmup_ratio":         args.warmup_ratio,
        "grad_clip":            args.grad_clip,
        "val_frac":             args.val_frac,
        "max_val_samples":      args.max_val_samples,
        "bf16":                 args.bf16,
        "early_stopping_patience":  args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "val_every_n_steps":    args.val_every_n_steps,
        "num_workers":          args.num_workers,
        "seed":                 args.seed,
        "n_train":              len(train_ds),
        "n_val":                len(val_ds),
        "k_hard_negs":          train_ds.n_hard_negatives,
        "device":               str(device),
    }
    (results_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- Training loop ----
    K = train_ds.n_hard_negatives
    history: list[dict] = []
    best_mrr = -1.0
    patience = 0
    best_path = results_dir / "checkpoint_best.pt"
    latest_path = results_dir / "checkpoint_latest.pt"

    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
              if (args.bf16 and device.type == "cuda") else (lambda: nullcontext())

    def run_validation_and_maybe_save(tag: str, epoch_idx: int, step_idx: int):
        nonlocal best_mrr, patience
        val_metrics = validate(model, val_loader, device)
        improved = val_metrics["mrr"] > best_mrr + args.early_stopping_min_delta
        new_best = val_metrics["mrr"] > best_mrr
        if new_best:
            best_mrr = val_metrics["mrr"]
        if improved:
            patience = 0
            torch.save({
                "epoch":              epoch_idx,
                "global_step":        step_idx,
                "model_state_dict":   model.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "metrics":            val_metrics,
                "cfg":                cfg,
            }, best_path)
            tail = f"  -> new best MRR={best_mrr:.4f}  (≥ +{args.early_stopping_min_delta} improvement, saved)"
        else:
            patience += 1
            tail = (f"  -> no significant improvement "
                    f"(best={best_mrr:.4f}, Δ<{args.early_stopping_min_delta}, "
                    f"patience {patience}/{args.early_stopping_patience})")
        history.append({
            "tag":         tag,
            "epoch":       epoch_idx + 1,
            "global_step": step_idx,
            "val_metrics": val_metrics,
        })
        torch.save({
            "epoch":                epoch_idx,
            "global_step":          step_idx,
            "model_state_dict":     model.state_dict(),
            "loss_fn_state_dict":   loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics":              val_metrics,
            "cfg":                  cfg,
        }, latest_path)
        print(
            f"  [{tag}] step={step_idx} epoch={epoch_idx + 1}  "
            f"MRR={val_metrics['mrr']:.4f}  "
            f"R@1={val_metrics['recall@1']:.3f}  "
            f"R@10={val_metrics['recall@10']:.3f}  "
            f"med_rank={val_metrics['median_rank']:.0f}"
        )
        print(tail)
        return val_metrics

    print()
    print("=" * 78)
    print(f"training: {args.epochs} epochs · batch={args.batch_size} · K={K} hard negs · lr={args.lr}")
    print(f"steps/epoch={len(train_loader)} · total_steps={total_steps}")
    print(f"bf16={args.bf16}  num_workers={args.num_workers}  "
          f"early_stop_min_delta={args.early_stopping_min_delta}  "
          f"val_every_n_steps={args.val_every_n_steps}")
    print("=" * 78)

    global_step = 0
    stop_training = False
    for epoch in range(args.epochs):
        if stop_training:
            break
        t0 = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0
        last_stats: dict = {}

        for a_ids, a_mask, p_ids, p_mask, n_ids, n_mask in train_loader:
            a_ids = a_ids.to(device, non_blocking=True)
            a_mask = a_mask.to(device, non_blocking=True)
            p_ids = p_ids.to(device, non_blocking=True)
            p_mask = p_mask.to(device, non_blocking=True)

            with amp_ctx():
                a = model.encode_anchor(a_ids, a_mask)
                p = model.encode_positive(p_ids, p_mask)
                if K > 0 and n_ids.numel() > 0:
                    n_ids = n_ids.to(device, non_blocking=True)
                    n_mask = n_mask.to(device, non_blocking=True)
                    n = model.encode_positive(n_ids, n_mask)      # (B*K, D)
                    B = a.size(0)
                    n = n.view(B, K, -1)                           # (B, K, D)
                else:
                    n = None

            optimizer.zero_grad()
            # Loss in fp32 (autocast for bf16 only covers matmul-heavy ops;
            # cross-entropy will upcast internally anyway).
            loss, stats = loss_fn(a.float(), p.float(),
                                  n.float() if n is not None else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            last_stats = stats
            global_step += 1

            # Intra-epoch validation
            if args.val_every_n_steps and global_step % args.val_every_n_steps == 0:
                run_validation_and_maybe_save("intra_epoch", epoch, global_step)
                model.train()  # validate() puts model in eval mode
                if patience >= args.early_stopping_patience:
                    print(f"\n[early stop] no MRR improvement ≥ "
                          f"{args.early_stopping_min_delta} for "
                          f"{patience} consecutive validations")
                    stop_training = True
                    break

        if stop_training:
            break

        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - t0
        print(
            f"\nepoch {epoch + 1}/{args.epochs}  ({epoch_time:.0f}s)"
            f"\n  train loss={train_loss:.4f}  τ={last_stats.get('temperature', 0):.4f}"
            f"  pos={last_stats.get('mean_pos_sim', 0):.3f}"
            f"  in-batch_neg={last_stats.get('mean_inbatch_neg', 0):.3f}"
            f"  hard_neg={last_stats.get('mean_hardneg_sim', 0):.3f}"
        )

        run_validation_and_maybe_save("end_of_epoch", epoch, global_step)
        if patience >= args.early_stopping_patience:
            print(f"\n[early stop] no MRR improvement ≥ "
                  f"{args.early_stopping_min_delta} for "
                  f"{patience} consecutive validations")
            break

    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print()
    print("=" * 78)
    print(f"[done] best val MRR={best_mrr:.4f}")
    print(f"  best:    {best_path}")
    print(f"  latest:  {latest_path}")
    print(f"  history: {results_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
