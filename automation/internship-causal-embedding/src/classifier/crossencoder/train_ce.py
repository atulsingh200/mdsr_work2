"""Train a CROSS-ENCODER directional classifier on tier-labeled segment pairs.

Adapted from src/classifier/training/train.py (dual-encoder), but the model is a
single encoder over the joint  text_1 [SEP] text_2  sequence. Loss is
BCEWithLogitsLoss. Reuses set_seed / make_logger / device_auto from the existing
training module so behavior matches the dual-encoder runs.

Outputs (in --out-dir/<run-name>/):
  config.json, best.pt, final.pt, train.log,
  test_metrics.json, test_predictions.jsonl
  (and val_predictions.jsonl, for the ensemble alpha sweep)

Invoke (from repo root):
  python3 -m src.classifier.crossencoder.train_ce \
      --data-dir data/aep_causal_classification_hard_neg_semantic \
      --out-dir runs/crossencoder --run-name ce_semantic \
      --base-model bert-base-uncased --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.classifier.crossencoder.data import CECollate, CEPairDataset
    from src.classifier.crossencoder.model import CrossEncoderClassifier
    from src.classifier.training.train import device_auto, make_logger, set_seed
else:
    from ..training.train import device_auto, make_logger, set_seed
    from .data import CECollate, CEPairDataset
    from .model import CrossEncoderClassifier


def scan_max_seq_len(jsonls: list[Path], tokenizer) -> int:
    """Longest JOINT tokenized length (text_1 [SEP] text_2) across splits."""
    mx = 0
    for p in jsonls:
        with open(p) as f:
            for ln in f:
                r = json.loads(ln)
                n = len(tokenizer.encode(r["text_1"], r["text_2"],
                                         add_special_tokens=True,
                                         max_length=512, truncation=True))
                if n > mx:
                    mx = n
    return mx


@torch.no_grad()
def evaluate(model, dl, device, use_bf16, loss_fn,
             dataset=None, return_predictions=False):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_n = 0.0, 0
    rows = [] if return_predictions else None
    ctx_factory = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                   if use_bf16 else nullcontext)
    for batch in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        labels = batch["labels"].to(device, non_blocking=True)
        with ctx_factory():
            logits = model(enc)
            loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        lbls = labels.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(lbls)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        if return_predictions and dataset is not None:
            for j, i_ds in enumerate(batch["idx"]):
                r = dataset.rows[i_ds]
                rows.append({
                    "url_1": r["url_1"], "url_2": r["url_2"],
                    "sub_1": r["sub_1"], "sub_2": r["sub_2"],
                    "tier_1": r["tier_1"], "tier_2": r["tier_2"],
                    "label": int(lbls[j]), "prob": float(probs[j]),
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
        auc = f1 = float("nan")
    return ({"loss": total_loss / max(total_n, 1), "acc": acc,
             "auc": auc, "f1": f1, "n": total_n}, rows)


def _per_tier_pair(preds_rows: list[dict]) -> dict:
    pair_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in preds_rows:
        key = (min(row["tier_1"], row["tier_2"]), max(row["tier_1"], row["tier_2"]))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int((row["prob"] > 0.5) == (row["label"] == 1))
    return {f"T{a}-T{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
            for (a, b), s in sorted(pair_stats.items())}


def _write_preds(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a cross-encoder directional classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="runs/crossencoder")
    ap.add_argument("--base-model", default="bert-base-uncased")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="Override; default = auto-scan then ceil(max*1.05)+2.")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = device_auto()
    use_bf16 = device.type == "cuda"

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}")
    log.info(f"device={device}  bf16={use_bf16}")

    log.info(f"tokenizer <- {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    data_dir = Path(args.data_dir)
    jsonls = [data_dir / f"directional_{s}.jsonl" for s in ("train", "val", "test")]
    if args.max_seq_len is None:
        log.info("scanning max JOINT token length across all splits")
        t0 = time.time()
        mx = scan_max_seq_len(jsonls, tokenizer)
        # Cross-encoder packs both texts; cap at the model's position limit.
        model_cap = getattr(tokenizer, "model_max_length", 512)
        if model_cap is None or model_cap > 100000:
            model_cap = 512
        args.max_seq_len = min(model_cap, max(128, int(np.ceil(mx * 1.05)) + 2))
        log.info(f"  observed max joint tokens={mx}  -> max_seq_len={args.max_seq_len}"
                 f"  (cap {model_cap}, scan {time.time() - t0:.1f}s)")
    else:
        log.info(f"max_seq_len={args.max_seq_len} (user-provided)")

    log.info("loading datasets")
    ds_train = CEPairDataset(jsonls[0])
    ds_val = CEPairDataset(jsonls[1])
    ds_test = CEPairDataset(jsonls[2])
    log.info(f"  train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    collate = CECollate(tokenizer, args.max_seq_len)
    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_val = DataLoader(ds_val, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_test = DataLoader(ds_test, batch_size=args.eval_batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)

    log.info(f"building CrossEncoderClassifier on {args.base_model}")
    model = CrossEncoderClassifier(args.base_model, dropout=args.dropout).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  hidden_size={model.hidden_size}  trainable_params={n_trainable:,}")

    encoder_params = list(model.encoder.parameters())
    head_params = list(model.classifier.parameters())
    optim = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.0},
    ])
    total_steps = len(dl_train) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    log.info(f"  total_steps={total_steps}  warmup_steps={warmup_steps}")

    loss_fn = nn.BCEWithLogitsLoss()

    cfg = {
        "base_model": args.base_model, "data_dir": str(args.data_dir),
        "run_name": args.run_name, "model_type": "cross_encoder",
        "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs, "warmup_frac": args.warmup_frac,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "dropout": args.dropout, "seed": args.seed,
        "max_seq_len": args.max_seq_len, "device": str(device), "bf16": use_bf16,
        "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "pool": "cls", "input_form": "[CLS] text_1 [SEP] text_2 [SEP]",
        "head": "Linear(hidden, 1)", "loss": "BCEWithLogitsLoss",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    ctx_factory = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                   if use_bf16 else nullcontext)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = running_correct = running_total = 0.0
        for step, batch in enumerate(dl_train, 1):
            enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device, non_blocking=True)
            with ctx_factory():
                logits = model(enc)
                loss = loss_fn(logits, labels)
            optim.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            running_correct += (preds == labels).sum().item()
            running_total += bs
            if step % args.log_every == 0:
                lrs = " ".join(f"lr{i}={pg['lr']:.2e}"
                               for i, pg in enumerate(optim.param_groups))
                log.info(f"  e{epoch} step {step:>5}/{len(dl_train)}  "
                         f"loss={running_loss / running_total:.4f}  "
                         f"acc={running_correct / running_total:.4f}  {lrs}")

        val_metrics, _ = evaluate(model, dl_val, device, use_bf16, loss_fn)
        log.info(f"[epoch {epoch}] train_loss={running_loss / running_total:.4f} "
                 f"train_acc={running_correct / running_total:.4f}  "
                 f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                 f"val_auc={val_metrics['auc']:.4f}  elapsed={time.time() - t_epoch:.1f}s")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": val_metrics, "cfg": cfg}, run_dir / "best.pt")
            log.info(f"  saved best.pt  val_acc={val_metrics['acc']:.4f}")

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
               run_dir / "final.pt")

    # -- reload best for eval on val + test --
    log.info("loading best.pt for val/test eval")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    val_metrics, val_rows = evaluate(model, dl_val, device, use_bf16, loss_fn,
                                     dataset=ds_val, return_predictions=True)
    _write_preds(run_dir / "val_predictions.jsonl", val_rows)
    log.info(f"[val] acc={val_metrics['acc']:.4f} auc={val_metrics['auc']:.4f} "
             f"f1={val_metrics['f1']:.4f}")

    test_metrics, test_rows = evaluate(model, dl_test, device, use_bf16, loss_fn,
                                       dataset=ds_test, return_predictions=True)
    log.info(f"[test] loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
             f"auc={test_metrics['auc']:.4f}  f1={test_metrics['f1']:.4f}")

    per_pair = _per_tier_pair(test_rows)
    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_tier_pair": per_pair}, indent=2))
    _write_preds(run_dir / "test_predictions.jsonl", test_rows)
    log.info(f"[done] results in {run_dir}")


if __name__ == "__main__":
    main()
