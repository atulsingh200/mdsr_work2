"""Fine-tune a bi-encoder with a pairwise preference (DPO-style) loss.

For each (anchor=A, positive=B) training pair the model learns that the
forward causal direction A→B is preferred over the reverse direction B→A:

    s(A,B) = enc_anchor(A) · enc_positive(B)   ← forward, should be HIGH
    s(B,A) = enc_anchor(B) · enc_positive(A)   ← reverse, should be LOW

    L = -log σ( (s(A,B) - s(B,A)) / τ )

This is a strict pairwise preference over directions — no hard-negative
mining needed, no separate negative text. The same (A, B) pair provides
both the positive signal (A→B) and the negative signal (B→A) via swapped
encoder roles.

Artifacts are read from / written to finetune_eval/results_reverse/.
Run mine_hard_negatives_reverse.py first to produce train_pairs.jsonl.

Usage:
  uv run python finetune_eval/train_finetune_reverse.py --dataset aep_causal
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts         # noqa: E402
from finetune_eval.datasets import list_datasets, split_path                 # noqa: E402


# ---------------------------------------------------------------------------
# Dataset: plain (anchor, positive) pairs — no hard_neg field needed
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Each item: (anchor_text, positive_text). The pairwise loss derives
    both the forward (A→B) and reverse (B→A) signals from the same pair."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.rows[idx]


def load_pairs_simple(
    path: Path, cap: int | None = None, seed: int = 0,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            a = (d.get("anchor") or "").strip()
            p = (d.get("positive") or "").strip()
            if a and p:
                rows.append((a, p))
    if cap is not None and len(rows) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=cap, replace=False)
        idx.sort()
        rows = [rows[i] for i in idx]
    return rows


def make_collate_fn(tokenizer, max_length: int):
    """Returns batches of (a_ids, a_mask, p_ids, p_mask) only."""
    def collate(batch):
        anchors   = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        a_ids, a_mask = tokenize_texts(tokenizer, anchors,   max_length)
        p_ids, p_mask = tokenize_texts(tokenizer, positives, max_length)
        return a_ids, a_mask, p_ids, p_mask

    return collate


# ---------------------------------------------------------------------------
# Validation (same as train_finetune.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: BiEncoder, loader, device: torch.device, k_values=(1, 3, 5, 10)) -> dict:
    model.eval()
    all_a, all_p = [], []
    for a_ids, a_mask, p_ids, p_mask in loader:
        a = model.encode_anchor(a_ids.to(device), a_mask.to(device))
        p = model.encode_positive(p_ids.to(device), p_mask.to(device))
        all_a.append(a.cpu())
        all_p.append(p.cpu())
    A = torch.cat(all_a)
    P = torch.cat(all_p)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list_datasets())
    ap.add_argument("--results-dir", default=str(ROOT / "finetune_eval/results_reverse"))
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased")
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
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-val-samples", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--early-stopping-patience", type=int, default=2)
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--val-every-n-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else torch.device("cpu"))

    results_dir = Path(args.results_dir) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[run dir] {results_dir}")

    train_pairs_path = results_dir / "train_pairs.jsonl"
    if not train_pairs_path.exists():
        raise SystemExit(
            f"missing mined artifacts. Run mine_hard_negatives_reverse.py for {args.dataset} first.\n"
            f"  expected: {train_pairs_path}"
        )

    print(f"[data] train pairs ← {train_pairs_path}")
    train_rows = load_pairs_simple(train_pairs_path, cap=None)
    print(f"  {len(train_rows)} train pairs  (pairwise DPO-style loss: A→B preferred over B→A)")

    # Val split: use dataset val file if available, else hold out val_frac of train
    val_path = split_path(args.dataset, "val")
    if val_path is not None and val_path.exists():
        from finetune_eval.data import load_pairs as _load_pairs
        val_pairs_all = _load_pairs(val_path, cap=None)
        if args.max_val_samples and len(val_pairs_all) > args.max_val_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(val_pairs_all), size=args.max_val_samples, replace=False)
            idx.sort()
            val_pairs_all = [val_pairs_all[i] for i in idx]
        val_rows = val_pairs_all
        print(f"[data] val pairs ← {val_path}  ({len(val_rows)})")
    else:
        n_val = max(1, int(len(train_rows) * args.val_frac))
        n_val = min(n_val, args.max_val_samples or n_val)
        rng = random.Random(args.seed)
        idx = list(range(len(train_rows)))
        rng.shuffle(idx)
        val_idx = sorted(idx[:n_val])
        train_rows = [train_rows[i] for i in sorted(idx[n_val:])]
        val_rows = [train_rows[i] for i in val_idx]
        print(f"[data] no val split; held out {len(val_rows)} pairs from train")

    train_ds = PairDataset(train_rows)
    val_ds   = PairDataset(val_rows)

    tokenizer = get_tokenizer(args.backbone)
    collate = make_collate_fn(tokenizer, args.max_seq_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[model] BiEncoder({args.backbone!r}, pooling={args.pooling!r})")
    model = BiEncoder(
        model_name=args.backbone,
        pooling_strategy=args.pooling,
        anchor_prefix="",
        positive_prefix="",
    ).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    log_temperature = torch.nn.Parameter(torch.tensor(np.log(args.temperature_init), device=device))
    encoder_params = list(model.anchor_encoder.parameters()) + list(model.positive_encoder.parameters())
    optimizer = AdamW([
        {"params": encoder_params,      "lr": args.lr,             "weight_decay": args.weight_decay},
        {"params": [log_temperature],   "lr": args.temperature_lr, "weight_decay": 0.0},
    ])
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.temperature_lr],
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

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
        "loss":                 "pairwise_preference_dpo",
        "device":               str(device),
    }
    (results_dir / "config.json").write_text(json.dumps(cfg, indent=2))

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
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
        if improved:
            patience = 0
            torch.save({
                "epoch":            epoch_idx,
                "global_step":      step_idx,
                "model_state_dict": model.state_dict(),
                "log_temperature":  log_temperature.item(),
                "metrics":          val_metrics,
                "cfg":              cfg,
            }, best_path)
            tail = f"  -> new best MRR={best_mrr:.4f}  (saved)"
        else:
            patience += 1
            tail = (f"  -> no improvement (best={best_mrr:.4f}, "
                    f"patience {patience}/{args.early_stopping_patience})")
        history.append({"tag": tag, "epoch": epoch_idx + 1, "global_step": step_idx, "val_metrics": val_metrics})
        torch.save({
            "epoch":                epoch_idx,
            "global_step":          step_idx,
            "model_state_dict":     model.state_dict(),
            "log_temperature":      log_temperature.item(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics":              val_metrics,
            "cfg":                  cfg,
        }, latest_path)
        print(
            f"  [{tag}] step={step_idx} epoch={epoch_idx + 1}  "
            f"MRR={val_metrics['mrr']:.4f}  R@1={val_metrics['recall@1']:.3f}  "
            f"R@10={val_metrics['recall@10']:.3f}  med_rank={val_metrics['median_rank']:.0f}"
        )
        print(tail)
        return val_metrics

    print()
    print("=" * 78)
    print(f"training: {args.epochs} epochs · batch={args.batch_size} · pairwise DPO loss · lr={args.lr}")
    print(f"  loss = -log σ( (s(A,B) - s(B,A)) / τ )   τ_init={args.temperature_init}")
    print(f"steps/epoch={len(train_loader)} · total_steps={total_steps}")
    print(f"bf16={args.bf16}  early_stop_min_delta={args.early_stopping_min_delta}")
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

        for a_ids, a_mask, p_ids, p_mask in train_loader:
            a_ids  = a_ids.to(device, non_blocking=True)
            a_mask = a_mask.to(device, non_blocking=True)
            p_ids  = p_ids.to(device, non_blocking=True)
            p_mask = p_mask.to(device, non_blocking=True)

            with amp_ctx():
                # Four embeddings for the pairwise DPO loss:
                #   s(A,B) = enc_anchor(A) · enc_positive(B)  — forward direction
                #   s(B,A) = enc_anchor(B) · enc_positive(A)  — reverse direction
                a_cause  = model.encode_anchor(a_ids, a_mask)    # enc_anchor(A)
                b_effect = model.encode_positive(p_ids, p_mask)  # enc_positive(B)
                b_cause  = model.encode_anchor(p_ids, p_mask)    # enc_anchor(B)
                a_effect = model.encode_positive(a_ids, a_mask)  # enc_positive(A)

            tau   = torch.exp(log_temperature).clamp(0.01, 0.5)
            s_ab  = (a_cause.float()  * b_effect.float()).sum(dim=-1) / tau  # (B,)
            s_ba  = (b_cause.float()  * a_effect.float()).sum(dim=-1) / tau  # (B,)
            loss  = -F.logsigmoid(s_ab - s_ba).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [log_temperature], args.grad_clip
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            last_stats = {
                "temperature": tau.item(),
                "mean_s_ab":   s_ab.mean().item(),
                "mean_s_ba":   s_ba.mean().item(),
                "mean_margin": (s_ab - s_ba).mean().item(),
            }
            global_step += 1

            if args.val_every_n_steps and global_step % args.val_every_n_steps == 0:
                run_validation_and_maybe_save("intra_epoch", epoch, global_step)
                model.train()
                if patience >= args.early_stopping_patience:
                    print(f"\n[early stop] patience exhausted")
                    stop_training = True
                    break

        if stop_training:
            break

        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - t0
        print(
            f"\nepoch {epoch + 1}/{args.epochs}  ({epoch_time:.0f}s)"
            f"\n  train loss={train_loss:.4f}  τ={last_stats.get('temperature', 0):.4f}"
            f"  s(A→B)={last_stats.get('mean_s_ab', 0):.3f}"
            f"  s(B→A)={last_stats.get('mean_s_ba', 0):.3f}"
            f"  margin={last_stats.get('mean_margin', 0):.3f}"
        )

        run_validation_and_maybe_save("end_of_epoch", epoch, global_step)
        if patience >= args.early_stopping_patience:
            print(f"\n[early stop] patience exhausted after {patience} checks")
            break

    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print()
    print("=" * 78)
    print(f"[done] best val MRR={best_mrr:.4f}")
    print(f"  best:    {best_path}")
    print(f"  latest:  {latest_path}")


if __name__ == "__main__":
    main()
