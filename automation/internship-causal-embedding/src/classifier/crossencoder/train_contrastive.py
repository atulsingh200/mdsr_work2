"""Train a cross-encoder with InfoNCE (contrastive) loss.

Uses grouped training data produced by build_contrastive_data.py.
Each training batch contains B complete groups (1 positive + 5 negatives each).
The model scores all B*6 pairs in one forward pass; InfoNCE loss is applied
per group with the positive always at index 0.

Val / test evaluation reuses the ORIGINAL directional_{val,test}.jsonl
(with the original binary labels) and the standard evaluate() from train_ce.py.

Outputs (in --out-dir/<run-name>/):
  config.json, best.pt, final.pt, train.log,
  test_metrics.json, test_predictions.jsonl, val_predictions.jsonl

Invoke (from repo root):
  python3 -m src.classifier.crossencoder.train_contrastive \\
      --contrastive-data-dir data/aep_causal_allpos_34 \\
      --eval-data-dir        data/aep_causal_classification_34 \\
      --out-dir runs/crossencoder_contrastive --run-name ce_contrastive \\
      --base-model bert-base-uncased --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.classifier.crossencoder.data import CECollate, CEPairDataset
    from src.classifier.crossencoder.data_contrastive import (
        ContrastiveCollate, GroupedDataset,
    )
    from src.classifier.crossencoder.model import CrossEncoderClassifier
    from src.classifier.crossencoder.train_ce import (
        _per_tier_pair, _write_preds, evaluate, scan_max_seq_len,
    )
    from src.classifier.training.train import device_auto, make_logger, set_seed
else:
    from ..training.train import device_auto, make_logger, set_seed
    from .data import CECollate, CEPairDataset
    from .data_contrastive import ContrastiveCollate, GroupedDataset
    from .model import CrossEncoderClassifier
    from .train_ce import _per_tier_pair, _write_preds, evaluate, scan_max_seq_len


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def infonce_loss(
    logits: torch.Tensor,
    group_sizes: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss over VARIABLE-size grouped logits.

    Args:
        logits:      (sum(group_sizes),) flat logits from CrossEncoderClassifier.
        group_sizes: (B,) tensor of per-group sizes (may differ across groups).
        tau:         temperature.

    Positive is always at index 0 within each group. For a group with logits
    [s_0, s_1, ..., s_k] (s_0 = positive):
        L = -log( exp(s_0/τ) / Σ_j exp(s_j/τ) )
          = F.cross_entropy([s_0, ..., s_k]/τ, target=0)

    Groups of size 1 (positive only, no negatives) contribute 0 loss and are
    skipped. The total is averaged over groups that have at least one negative.
    """
    scaled = logits / tau
    chunks = torch.split(scaled, group_sizes.tolist())
    losses = []
    for g in chunks:
        if g.numel() < 2:
            continue  # no negatives -> nothing to contrast
        # cross_entropy on a single row, target index 0 (the positive)
        losses.append(F.cross_entropy(g.unsqueeze(0),
                                      g.new_zeros(1, dtype=torch.long)))
    if not losses:
        return logits.new_zeros((), requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a cross-encoder with InfoNCE contrastive loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contrastive-data-dir", required=True,
                    help="Directory with grouped contrastive JSONL (train+val).")
    ap.add_argument("--eval-data-dir", required=True,
                    help="Original _34 dir; untouched test/val for binary eval.")
    ap.add_argument("--out-dir", default="runs/crossencoder_contrastive")
    ap.add_argument("--base-model", default="bert-base-uncased")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Number of groups per training batch.")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07,
                    help="InfoNCE temperature τ.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = device_auto()
    use_bf16 = device.type == "cuda"

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}")
    log.info(f"device={device}  bf16={use_bf16}  tau={args.temperature}")

    log.info(f"tokenizer <- {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    contrastive_dir = Path(args.contrastive_data_dir)
    eval_dir = Path(args.eval_data_dir)

    # ---- Max sequence length: scan contrastive train + eval val ----
    if args.max_seq_len is None:
        scan_paths = [
            contrastive_dir / "directional_train.jsonl",
            eval_dir / "directional_val.jsonl",
        ]
        log.info("scanning max JOINT token length ...")
        t0 = time.time()
        mx = scan_max_seq_len(scan_paths, tokenizer)
        model_cap = getattr(tokenizer, "model_max_length", 512)
        if model_cap is None or model_cap > 100_000:
            model_cap = 512
        args.max_seq_len = min(model_cap, max(128, int(np.ceil(mx * 1.05)) + 2))
        log.info(f"  observed max={mx}  -> max_seq_len={args.max_seq_len}"
                 f"  (scan {time.time() - t0:.1f}s)")
    else:
        log.info(f"max_seq_len={args.max_seq_len} (user-provided)")

    # ---- Training data (grouped contrastive) ----
    log.info("loading grouped training dataset")
    ds_train = GroupedDataset(contrastive_dir / "directional_train.jsonl")
    log.info(f"  train groups: {len(ds_train)}")

    # Infer group size from first group
    first_group_size = len(ds_train[0])
    log.info(f"  group size (inferred): {first_group_size}")

    collate_train = ContrastiveCollate(tokenizer, args.max_seq_len)
    pin = device.type == "cuda"
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
        pin_memory=pin,
    )

    # ---- Val / test data (original binary labels) ----
    log.info("loading val/test datasets (original binary labels)")
    ds_val = CEPairDataset(eval_dir / "directional_val.jsonl")
    ds_test = CEPairDataset(eval_dir / "directional_test.jsonl")
    log.info(f"  val={len(ds_val)}  test={len(ds_test)}")

    collate_eval = CECollate(tokenizer, args.max_seq_len)
    dl_val = DataLoader(ds_val, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_eval,
                        pin_memory=pin)
    dl_test = DataLoader(ds_test, batch_size=args.eval_batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_eval,
                         pin_memory=pin)

    # ---- Model ----
    log.info(f"building CrossEncoderClassifier on {args.base_model}")
    model = CrossEncoderClassifier(args.base_model, dropout=args.dropout).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  hidden_size={model.hidden_size}  trainable_params={n_trainable:,}")

    # ---- Optimizer + scheduler ----
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.classifier.parameters())
    optim = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder,
         "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.0},
    ])
    total_steps = len(dl_train) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    log.info(f"  total_steps={total_steps}  warmup_steps={warmup_steps}")

    # BCE loss is only used for val/test evaluation (to compute val loss)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    cfg = {
        "base_model": args.base_model,
        "contrastive_data_dir": str(args.contrastive_data_dir),
        "eval_data_dir": str(args.eval_data_dir),
        "run_name": args.run_name,
        "model_type": "cross_encoder_contrastive",
        "loss": "InfoNCE",
        "temperature": args.temperature,
        "group_size": first_group_size,
        "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs, "warmup_frac": args.warmup_frac,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "dropout": args.dropout, "seed": args.seed,
        "max_seq_len": args.max_seq_len, "device": str(device), "bf16": use_bf16,
        "n_train_groups": len(ds_train),
        "n_val": len(ds_val), "n_test": len(ds_test),
        "pool": "cls", "input_form": "[CLS] text_1 [SEP] text_2 [SEP]",
        "head": "Linear(hidden, 1)",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16 else nullcontext
    )

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = running_pos_rank1 = running_total = 0.0

        for step, batch in enumerate(dl_train, 1):
            enc = {k: v.to(device, non_blocking=True)
                   for k, v in batch["enc"].items()}
            group_sizes = batch["group_sizes"].to(device)

            with ctx_factory():
                logits = model(enc)                           # (B*G,)
                loss = infonce_loss(logits, group_sizes, tau=args.temperature)

            optim.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            # Proxy metric: fraction of groups where positive (index 0) has the
            # highest score. Handles variable group sizes via per-group split.
            B = int(group_sizes.numel())
            with torch.no_grad():
                chunks = torch.split(logits.detach(), group_sizes.tolist())
                n_rank1 = sum(int(g.argmax().item() == 0)
                              for g in chunks if g.numel() >= 2)
                n_scored = sum(1 for g in chunks if g.numel() >= 2)
                pos_rank1 = n_rank1 / max(n_scored, 1)

            running_loss += loss.item() * B
            running_pos_rank1 += pos_rank1 * B
            running_total += B

            if step % args.log_every == 0:
                lrs = " ".join(
                    f"lr{i}={pg['lr']:.2e}"
                    for i, pg in enumerate(optim.param_groups)
                )
                log.info(
                    f"  e{epoch} step {step:>5}/{len(dl_train)}  "
                    f"loss={running_loss / running_total:.4f}  "
                    f"pos_rank1={running_pos_rank1 / running_total:.4f}  {lrs}"
                )

        # ---- Epoch-level validation (binary classification on original labels) ----
        val_metrics, _ = evaluate(model, dl_val, device, use_bf16, bce_loss_fn)
        log.info(
            f"[epoch {epoch}] "
            f"train_loss={running_loss / running_total:.4f}  "
            f"train_pos_rank1={running_pos_rank1 / running_total:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}  "
            f"elapsed={time.time() - t_epoch:.1f}s"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "val": val_metrics, "cfg": cfg},
                run_dir / "best.pt",
            )
            log.info(f"  saved best.pt  val_acc={val_metrics['acc']:.4f}")

    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
        run_dir / "final.pt",
    )

    # ---- Reload best checkpoint for final val + test eval ----
    log.info("loading best.pt for val/test eval")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    val_metrics, val_rows = evaluate(
        model, dl_val, device, use_bf16, bce_loss_fn,
        dataset=ds_val, return_predictions=True,
    )
    _write_preds(run_dir / "val_predictions.jsonl", val_rows)
    log.info(f"[val] acc={val_metrics['acc']:.4f}  "
             f"auc={val_metrics['auc']:.4f}  f1={val_metrics['f1']:.4f}")

    test_metrics, test_rows = evaluate(
        model, dl_test, device, use_bf16, bce_loss_fn,
        dataset=ds_test, return_predictions=True,
    )
    log.info(f"[test] loss={test_metrics['loss']:.4f}  "
             f"acc={test_metrics['acc']:.4f}  "
             f"auc={test_metrics['auc']:.4f}  f1={test_metrics['f1']:.4f}")

    per_pair = _per_tier_pair(test_rows)
    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_tier_pair": per_pair}, indent=2)
    )
    _write_preds(run_dir / "test_predictions.jsonl", test_rows)
    log.info(f"[done] results in {run_dir}")


if __name__ == "__main__":
    main()
