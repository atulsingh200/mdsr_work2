"""Train a directional classifier on tier-labeled segment pairs.

Outputs (in --out-dir/<run-name>/):
  config.json              hyperparameters and architecture spec
  best.pt                  checkpoint at best val accuracy
  final.pt                 checkpoint after last epoch
  train.log                per-step loss, per-epoch val metrics
  test_metrics.json        test loss/acc/AUC/F1 + per-tier-pair accuracy
  test_predictions.jsonl   one row per test pair with predicted probability

CLI example:
  uv run train-classifier \
      --data-dir data/aep_causal_classification \
      --out-dir runs/classifier \
      --head-type mlp --head-hidden-dims 512,128 \
      --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..heads import head_cfg_from_args, parse_hidden_dims
from ..model import DirectionalClassifier
from .data import Collate, PairDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scan_max_seq_len(jsonls: list[Path], tokenizer) -> int:
    """Longest tokenized length (with special tokens) across all splits."""
    mx = 0
    for p in jsonls:
        with open(p) as f:
            for ln in f:
                r = json.loads(ln)
                for t in (r["text_1"], r["text_2"]):
                    n = len(tokenizer.encode(t, add_special_tokens=True))
                    if n > mx:
                        mx = n
    return mx


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def _describe_head(cfg: dict) -> str:
    """Compact human-readable summary of a head config."""
    t = cfg.get("head_type", "linear")
    if t == "linear":
        return "Linear(4d, 1)"
    if t == "mlp":
        dims = "->".join(str(h) for h in cfg.get("hidden_dims", [512, 128]))
        return (
            f"MLP(4d->{dims}->1, act={cfg.get('activation', 'gelu')}, "
            f"norm={cfg.get('norm', 'layer')}, dropout={cfg.get('dropout', 0.1)})"
        )
    return t


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: DirectionalClassifier,
    dl: DataLoader,
    device: torch.device,
    use_bf16: bool,
    loss_fn: nn.Module,
    dataset: PairDataset | None = None,
    return_predictions: bool = False,
) -> tuple[dict, list[dict] | None]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    rows: list[dict] | None = [] if return_predictions else None

    ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16
        else nullcontext
    )

    for batch in dl:
        enc1 = {k: v.to(device, non_blocking=True) for k, v in batch["enc1"].items()}
        enc2 = {k: v.to(device, non_blocking=True) for k, v in batch["enc2"].items()}
        labels = batch["labels"].to(device, non_blocking=True)
        with ctx_factory():
            dir_logits, _, _ = model(enc1, enc2)
            loss = loss_fn(dir_logits, labels)
        probs = torch.sigmoid(dir_logits).float().cpu().numpy()
        lbls = labels.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(lbls)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        if return_predictions and dataset is not None:
            for j, i_ds in enumerate(batch["idx"]):
                r = dataset.rows[i_ds]
                rows.append({
                    "url_1": r.get("url_1", ""),
                    "url_2": r.get("url_2", ""),
                    "sub_1": r.get("sub_1", ""),
                    "sub_2": r.get("sub_2", ""),
                    "tier_1": r.get("tier_1"),
                    "tier_2": r.get("tier_2"),
                    "label": int(lbls[j]),
                    "prob": float(probs[j]),
                })

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    try:
        from sklearn.metrics import f1_score, roc_auc_score

        auc = float(roc_auc_score(labels, probs))
        f1 = float(f1_score(labels, preds))
    except Exception:
        auc = float("nan")
        f1 = float("nan")
    return (
        {
            "loss": total_loss / max(total_n, 1),
            "acc": acc,
            "auc": auc,
            "f1": f1,
            "n": total_n,
        },
        rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a directional classifier on tier-labeled segment pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", default="data/aep_causal_classification")
    ap.add_argument(
        "--split-prefix",
        default="directional_",
        help="Filename prefix for splits, i.e. <prefix>{train,val,test}.jsonl. "
        "Use --split-prefix '' for bare train.jsonl/val.jsonl/test.jsonl.",
    )
    ap.add_argument("--out-dir", default="runs/classifier")
    ap.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="LR for the entire classification head (linear or all MLP layers).",
    )
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override; default = auto-scan all splits then ceil(max*1.05)+2.",
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--freeze-encoders",
        action="store_true",
        help="Freeze both encoders; train only the head (linear probe).",
    )
    # ---- Pluggable head architecture ----
    ap.add_argument(
        "--head-type",
        choices=["linear", "mlp"],
        default="linear",
        help="Classification head over the [A;B;A-B;A*B] feature.",
    )
    ap.add_argument(
        "--head-hidden-dims",
        type=str,
        default="512,128",
        help="Comma-separated hidden dims for --head-type mlp.",
    )
    ap.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
        help="Dropout between MLP hidden layers.",
    )
    ap.add_argument(
        "--head-activation",
        choices=["gelu", "relu", "silu"],
        default="gelu",
    )
    ap.add_argument(
        "--head-norm",
        choices=["layer", "none"],
        default="layer",
    )
    # ---- Auxiliary tier-prediction heads ----
    ap.add_argument(
        "--tier-aux-weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary tier-prediction loss. 0 = disabled.",
    )
    args = ap.parse_args()
    args.head_hidden_dims = parse_hidden_dims(args.head_hidden_dims)

    set_seed(args.seed)
    device = device_auto()
    use_bf16 = device.type == "cuda"

    run_name = (
        args.run_name
        or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bge_small"
    )
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}")
    log.info(f"device={device}  bf16={use_bf16}")

    # -- tokenizer --
    log.info(f"tokenizer <- {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # -- max_seq_len --
    data_dir = Path(args.data_dir)
    jsonls = [
        data_dir / f"{args.split_prefix}{s}.jsonl" for s in ("train", "val", "test")
    ]
    # Cap at the encoder's positional-embedding limit (e.g. 512 for BERT/BGE);
    # exceeding it crashes in the embedding add (position_embeddings mismatch).
    model_cap = getattr(tokenizer, "model_max_length", 512)
    if model_cap is None or model_cap > 100000:
        model_cap = 512
    if args.max_seq_len is None:
        log.info("scanning max token length across all splits")
        t0 = time.time()
        mx = scan_max_seq_len(jsonls, tokenizer)
        args.max_seq_len = min(model_cap, max(128, int(np.ceil(mx * 1.05)) + 2))
        log.info(
            f"  observed max tokens={mx}  -> max_seq_len={args.max_seq_len}"
            f"  (cap {model_cap}, scan {time.time() - t0:.1f}s)"
        )
    else:
        if args.max_seq_len > model_cap:
            log.info(f"max_seq_len={args.max_seq_len} exceeds model cap "
                     f"{model_cap}; clamping to {model_cap}")
            args.max_seq_len = model_cap
        log.info(f"max_seq_len={args.max_seq_len} (user-provided)")

    # -- datasets --
    log.info("loading datasets")
    ds_train = PairDataset(jsonls[0])
    ds_val = PairDataset(jsonls[1])
    ds_test = PairDataset(jsonls[2])
    log.info(f"  train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    collate = Collate(tokenizer, args.max_seq_len)
    pin = device.type == "cuda"
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )

    # -- model --
    head_cfg = head_cfg_from_args(args)
    n_tiers = 15 if args.tier_aux_weight > 0 else 0
    log.info(
        f"building DirectionalClassifier on {args.base_model}"
        f"  head_cfg={head_cfg}  n_tiers={n_tiers}"
    )
    model = DirectionalClassifier(
        args.base_model, head_cfg=head_cfg, n_tiers=n_tiers
    ).to(device)
    if args.freeze_encoders:
        for p in model.src.parameters():
            p.requires_grad = False
        for p in model.tgt.parameters():
            p.requires_grad = False
        model.src.eval()
        model.tgt.eval()
        log.info("  encoders FROZEN (linear probe mode)")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  hidden_size={model.hidden_size}  trainable_params={n_trainable:,}")

    # -- optimizer with param groups --
    if n_tiers > 0:
        head_params = (
            list(model.head_backbone.parameters())
            + list(model.head_final.parameters())
            + list(model.tier_head_1.parameters())
            + list(model.tier_head_2.parameters())
        )
    else:
        head_params = list(model.head.parameters())
    param_groups: list[dict] = [
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.0},
    ]
    if not args.freeze_encoders:
        encoder_params = list(model.src.parameters()) + list(model.tgt.parameters())
        param_groups.insert(
            0,
            {
                "params": encoder_params,
                "lr": args.lr_encoder,
                "weight_decay": args.weight_decay,
            },
        )
    optim = torch.optim.AdamW(param_groups)
    total_steps = len(dl_train) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    log.info(f"  total_steps={total_steps}  warmup_steps={warmup_steps}")

    loss_fn = nn.BCEWithLogitsLoss()
    tier_loss_fn = nn.CrossEntropyLoss() if n_tiers > 0 else None
    tier_aux_weight = args.tier_aux_weight

    # -- config dump --
    cfg = {
        "base_model": args.base_model,
        "data_dir": str(args.data_dir),
        "run_name": run_name,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "warmup_frac": args.warmup_frac,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "device": str(device),
        "bf16": use_bf16,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "pool": "cls",
        "l2_normalize": True,
        "concat_form": "[A; B; A-B; A*B]",
        "head_cfg": head_cfg,
        "head": _describe_head(head_cfg),
        "freeze_encoders": args.freeze_encoders,
        "tier_aux_weight": args.tier_aux_weight,
        "n_tiers": n_tiers,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # -- train loop --
    ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16
        else nullcontext
    )

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoders:
            model.src.eval()
            model.tgt.eval()
        t_epoch = time.time()
        running_loss = 0.0
        running_dir_loss = 0.0
        running_aux_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, batch in enumerate(dl_train, 1):
            enc1 = {
                k: v.to(device, non_blocking=True)
                for k, v in batch["enc1"].items()
            }
            enc2 = {
                k: v.to(device, non_blocking=True)
                for k, v in batch["enc2"].items()
            }
            labels = batch["labels"].to(device, non_blocking=True)

            with ctx_factory():
                dir_logits, tier_logits_1, tier_logits_2 = model(enc1, enc2)
                dir_loss = loss_fn(dir_logits, labels)
                aux_loss_val = 0.0
                if tier_aux_weight > 0 and tier_logits_1 is not None:
                    tier_labels_1 = batch["tier_1"].to(device, non_blocking=True)
                    tier_labels_2 = batch["tier_2"].to(device, non_blocking=True)
                    aux_loss = tier_loss_fn(
                        tier_logits_1, tier_labels_1
                    ) + tier_loss_fn(tier_logits_2, tier_labels_2)
                    aux_loss_val = aux_loss.item()
                    loss = dir_loss + tier_aux_weight * aux_loss
                else:
                    loss = dir_loss

            optim.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_dir_loss += dir_loss.item() * bs
            running_aux_loss += aux_loss_val * bs
            preds = (torch.sigmoid(dir_logits.detach()) > 0.5).float()
            binary_labels = (labels > 0.5).float()
            running_correct += (preds == binary_labels).sum().item()
            running_total += bs

            if step % args.log_every == 0:
                lrs = " ".join(
                    f"lr{i}={pg['lr']:.2e}"
                    for i, pg in enumerate(optim.param_groups)
                )
                aux_str = (
                    f"  dir={running_dir_loss / running_total:.4f}"
                    f"  aux={running_aux_loss / running_total:.4f}"
                    if tier_aux_weight > 0
                    else ""
                )
                log.info(
                    f"  e{epoch} step {step:>5}/{len(dl_train)}  "
                    f"loss={running_loss / running_total:.4f}{aux_str}  "
                    f"acc={running_correct / running_total:.4f}  "
                    f"{lrs}"
                )

        val_metrics, _ = evaluate(model, dl_val, device, use_bf16, loss_fn)
        log.info(
            f"[epoch {epoch}] "
            f"train_loss={running_loss / running_total:.4f} "
            f"train_acc={running_correct / running_total:.4f}  "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}  "
            f"elapsed={time.time() - t_epoch:.1f}s"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val_metrics,
                    "cfg": cfg,
                },
                run_dir / "best.pt",
            )
            log.info(f"  saved best.pt  val_acc={val_metrics['acc']:.4f}")

    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
        run_dir / "final.pt",
    )

    # -- test with best checkpoint --
    log.info("loading best.pt for test eval")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, preds_rows = evaluate(
        model, dl_test, device, use_bf16, loss_fn,
        dataset=ds_test, return_predictions=True,
    )
    log.info(
        f"[test] loss={test_metrics['loss']:.4f}  "
        f"acc={test_metrics['acc']:.4f}  "
        f"auc={test_metrics['auc']:.4f}  "
        f"f1={test_metrics['f1']:.4f}"
    )

    # per-tier-pair accuracy
    pair_stats: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0}
    )
    for row in preds_rows:
        t1 = row["tier_1"] or 0
        t2 = row["tier_2"] or 0
        key = (min(t1, t2), max(t1, t2))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int(
            (row["prob"] > 0.5) == (row["label"] == 1)
        )
    per_pair = {
        f"T{a}-T{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
        for (a, b), s in sorted(pair_stats.items())
    }
    log.info("[test] per-tier-pair accuracy:")
    for k, v in per_pair.items():
        log.info(f"    {k:>9}  n={v['n']:>5}  acc={v['acc']:.4f}")

    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_tier_pair": per_pair}, indent=2)
    )
    with (run_dir / "test_predictions.jsonl").open("w") as f:
        for row in preds_rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"[done] results in {run_dir}")


if __name__ == "__main__":
    main()
