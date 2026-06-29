"""Train LorentzBiEncoder on the full workflow dataset.

Reads pre-mined hard negatives from mine_hard_negatives.py (--no-cap run).
Uses three-term loss: causal InfoNCE + ordering + reverse-pair penalty.

Usage:
  uv run python lorentz_enc_workflow/train.py \\
      --backbone BAAI/bge-m3 \\
      --results-dir lorentz_enc_workflow/results \\
      --batch-size 32 --epochs 1 --bf16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lorentz_enc_workflow.model import LorentzBiEncoder, get_tokenizer, tokenize  # noqa: E402
from lorentz_enc_workflow.losses import LorentzLoss                                # noqa: E402
from lorentz_enc_workflow.scoring import causal_similarity_matrix                  # noqa: E402

from finetune_eval.data import TripleDataset, load_pairs, make_collate_fn          # noqa: E402
from finetune_eval.datasets import split_path                                      # noqa: E402


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(
    model: LorentzBiEncoder,
    loader: DataLoader,
    device: torch.device,
    k_values: tuple = (1, 3, 5, 10),
) -> dict:
    """N×N validation on the space component (symmetric cosine, fast)."""
    model.eval()
    all_s_a, all_s_p = [], []
    for a_ids, a_mask, p_ids, p_mask, _n_ids, _n_mask in loader:
        a_ids = a_ids.to(device, non_blocking=True)
        a_mask = a_mask.to(device, non_blocking=True)
        p_ids = p_ids.to(device, non_blocking=True)
        p_mask = p_mask.to(device, non_blocking=True)
        s_a, _ = model(a_ids, a_mask)
        s_p, _ = model(p_ids, p_mask)
        all_s_a.append(s_a.cpu())
        all_s_p.append(s_p.cpu())
    A = torch.cat(all_s_a)
    P = torch.cat(all_s_p)
    sim = A @ P.t()
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
    ap.add_argument("--results-dir", default=str(ROOT / "lorentz_enc_workflow/results"))
    ap.add_argument("--backbone", default="BAAI/bge-m3")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--space-dim", type=int, default=1004)
    ap.add_argument("--time-dim", type=int, default=20)
    ap.add_argument("--space-hidden", type=int, default=1024)
    ap.add_argument("--time-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature-lr", type=float, default=1e-3)
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="Space weight in scoring")
    ap.add_argument("--beta", type=float, default=0.5, help="Time weight in scoring")
    ap.add_argument("--tau", type=float, default=10.0, help="SteepSigmoid steepness")
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--lambda-ord", type=float, default=0.5)
    ap.add_argument("--lambda-rev", type=float, default=1.0)
    ap.add_argument("--ord-margin", type=float, default=0.5)
    ap.add_argument("--ord-slack", type=float, default=0.1)
    ap.add_argument("--margin-rev", type=float, default=0.0)
    ap.add_argument("--max-val-samples", type=int, default=4000)
    ap.add_argument("--val-every-n-steps", type=int, default=5000)
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    ap.add_argument("--max-train-steps", type=int, default=None,
                    help="Stop after N steps (useful for smoke-testing).")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = auto_device()
    ds_dir = Path(args.results_dir) / "workflow"
    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[run dir] {ds_dir}")

    # ---- Load pre-mined artifacts ----
    train_pairs_path = ds_dir / "train_pairs.jsonl"
    hard_neg_path = ds_dir / "hard_negatives.npy"
    if not train_pairs_path.exists() or not hard_neg_path.exists():
        raise SystemExit(
            "Missing mined artifacts. Run:\n"
            "  uv run python finetune_eval/mine_hard_negatives.py "
            "--dataset workflow --no-cap --out-dir lorentz_enc_workflow/results"
        )

    print(f"[data] train pairs <- {train_pairs_path}")
    train_pairs = load_pairs(train_pairs_path, cap=None)
    hard_negatives = np.load(hard_neg_path)
    K = int(hard_negatives.shape[1])
    print(f"  {len(train_pairs):,} train pairs  K={K} hard negs/anchor")

    # ---- Validation data ----
    val_path = split_path("workflow", "val")
    if val_path is not None and val_path.exists():
        val_pairs_all = load_pairs(val_path, cap=None)
        if args.max_val_samples and len(val_pairs_all) > args.max_val_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(val_pairs_all), size=args.max_val_samples, replace=False)
            idx.sort()
            val_pairs = [val_pairs_all[i] for i in idx]
        else:
            val_pairs = val_pairs_all
        print(f"[data] val pairs <- {val_path}  ({len(val_pairs):,})")
    else:
        raise SystemExit(f"Val split not found at {val_path}")

    train_ds = TripleDataset(train_pairs, hard_negatives, active_indices=None)
    val_ds = TripleDataset(val_pairs, hard_negatives=None)

    # ---- Model + tokenizer ----
    print(f"[model] LorentzBiEncoder({args.backbone!r})")
    tokenizer = get_tokenizer(args.backbone)
    model = LorentzBiEncoder(
        backbone_name=args.backbone,
        space_dim=args.space_dim,
        time_dim=args.time_dim,
        space_hidden=args.space_hidden,
        time_hidden=args.time_hidden,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    print(f"  space_dim={args.space_dim}  time_dim={args.time_dim}")

    collate = make_collate_fn(tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Loss + optimizer ----
    loss_fn = LorentzLoss(
        alpha=args.alpha,
        beta=args.beta,
        tau_score=args.tau,
        temperature_init=args.temperature_init,
        lambda_nce=args.lambda_nce,
        lambda_ord=args.lambda_ord,
        lambda_rev=args.lambda_rev,
        ord_margin=args.ord_margin,
        ord_slack=args.ord_slack,
        margin_rev=args.margin_rev,
    ).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.space_head.parameters()) +
        list(model.time_head.parameters())
    )
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr,             "weight_decay": args.weight_decay},
        {"params": head_params,     "lr": args.lr * 5,         "weight_decay": args.weight_decay},
        {"params": [loss_fn.log_temperature], "lr": args.temperature_lr, "weight_decay": 0.0},
    ])
    total_steps = max(len(train_loader) * args.epochs, 1)
    if args.max_train_steps:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.lr * 5, args.temperature_lr],
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

    # ---- Config ----
    cfg = {
        "backbone": args.backbone, "pooling": args.pooling,
        "space_dim": args.space_dim, "time_dim": args.time_dim,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "lr": args.lr, "temperature_lr": args.temperature_lr,
        "temperature_init": args.temperature_init,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "grad_clip": args.grad_clip,
        "alpha": args.alpha, "beta": args.beta, "tau": args.tau,
        "lambda_nce": args.lambda_nce, "lambda_ord": args.lambda_ord,
        "lambda_rev": args.lambda_rev,
        "ord_margin": args.ord_margin, "ord_slack": args.ord_slack,
        "margin_rev": args.margin_rev,
        "bf16": args.bf16,
        "val_every_n_steps": args.val_every_n_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "k_hard_negs": K, "device": str(device),
    }
    (ds_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- Training loop ----
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
              if (args.bf16 and device.type == "cuda") else (lambda: nullcontext())

    best_mrr = -1.0
    patience_count = 0
    best_path = ds_dir / "checkpoint_best.pt"
    latest_path = ds_dir / "checkpoint_latest.pt"
    history: list[dict] = []

    def save_checkpoint(tag: str, epoch_idx: int, step_idx: int):
        nonlocal best_mrr, patience_count
        val_metrics = validate(model, val_loader, device)
        improved = val_metrics["mrr"] > best_mrr + args.early_stopping_min_delta
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
        if improved:
            patience_count = 0
            torch.save({
                "epoch": epoch_idx, "global_step": step_idx,
                "model_state_dict": model.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "metrics": val_metrics, "cfg": cfg,
            }, best_path)
            suffix = f"  -> new best MRR={best_mrr:.4f} (saved)"
        else:
            patience_count += 1
            suffix = (f"  -> no improvement (best={best_mrr:.4f}, "
                      f"patience {patience_count}/{args.early_stopping_patience})")
        history.append({"tag": tag, "epoch": epoch_idx + 1,
                        "global_step": step_idx, "val_metrics": val_metrics})
        torch.save({
            "epoch": epoch_idx, "global_step": step_idx,
            "model_state_dict": model.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": val_metrics, "cfg": cfg,
        }, latest_path)
        print(f"  [{tag}] step={step_idx} epoch={epoch_idx+1}  "
              f"MRR={val_metrics['mrr']:.4f}  "
              f"R@1={val_metrics['recall@1']:.3f}  "
              f"R@10={val_metrics['recall@10']:.3f}  "
              f"med_rank={val_metrics['median_rank']:.0f}")
        print(suffix)
        return val_metrics

    print(f"\n{'='*78}")
    print(f"training: {args.epochs} epoch(s) · batch={args.batch_size} · "
          f"K={K} hard negs · lr={args.lr} · bf16={args.bf16}")
    print(f"steps/epoch={len(train_loader):,} · total_steps={total_steps:,}")
    print(f"{'='*78}\n")

    global_step = 0
    stop_training = False

    for epoch in range(args.epochs):
        if stop_training:
            break
        t0 = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0

        for a_ids, a_mask, p_ids, p_mask, n_ids, n_mask in train_loader:
            a_ids = a_ids.to(device, non_blocking=True)
            a_mask = a_mask.to(device, non_blocking=True)
            p_ids = p_ids.to(device, non_blocking=True)
            p_mask = p_mask.to(device, non_blocking=True)

            with amp_ctx():
                space_a, time_a = model(a_ids, a_mask)
                space_p, time_p = model(p_ids, p_mask)
                if K > 0 and n_ids.numel() > 0:
                    n_ids = n_ids.to(device, non_blocking=True)
                    n_mask = n_mask.to(device, non_blocking=True)
                    space_n, time_n = model(n_ids, n_mask)  # (B*K, D_s), (B*K, D_t)
                else:
                    space_n = time_n = None

            optimizer.zero_grad()
            loss, stats = loss_fn(
                space_a.float(), time_a.float(),
                space_p.float(), time_p.float(),
                space_n.float() if space_n is not None else None,
                time_n.float() if time_n is not None else None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"  step {global_step:6d}  loss={loss.item():.4f}  "
                    f"nce={stats['loss_nce']:.3f}  ord={stats['loss_ord']:.3f}  "
                    f"rev={stats['loss_rev']:.3f}  τ={stats['temperature']:.4f}  "
                    f"pos={stats['mean_pos_sim']:.3f}  hn={stats['mean_hardneg_sim']:.3f}  "
                    f"rev_sim={stats['mean_rev_sim']:.3f}"
                )

            if args.val_every_n_steps and global_step % args.val_every_n_steps == 0:
                save_checkpoint("intra_epoch", epoch, global_step)
                model.train()
                if patience_count >= args.early_stopping_patience:
                    print(f"\n[early stop] patience exhausted")
                    stop_training = True
                    break

            if args.max_train_steps and global_step >= args.max_train_steps:
                print(f"\n[max_train_steps={args.max_train_steps}] stopping early")
                stop_training = True
                break

        if stop_training:
            break

        epoch_time = time.time() - t0
        avg_loss = running_loss / max(n_batches, 1)
        print(f"\nepoch {epoch+1}/{args.epochs}  ({epoch_time:.0f}s)  avg_loss={avg_loss:.4f}")
        save_checkpoint("end_of_epoch", epoch, global_step)
        if patience_count >= args.early_stopping_patience:
            print(f"\n[early stop] no improvement for {patience_count} checks")
            break

    (ds_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n{'='*78}")
    print(f"[done] best val MRR={best_mrr:.4f}")
    print(f"  best:    {best_path}")
    print(f"  latest:  {latest_path}")
    print(f"  history: {ds_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
