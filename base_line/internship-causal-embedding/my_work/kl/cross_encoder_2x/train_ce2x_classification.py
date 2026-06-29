"""Train CrossEncoder2x as a binary classifier on text_1/text_2/label data.

Reads JSONL files with {text_1, text_2, label} fields.
Concatenates text_1 and text_2 as [CLS] text_1 [SEP] text_2 [SEP],
passes through 24-layer BERT, and maps CLS -> Linear(1) -> logit.
Trained with BCE loss; label 0 or 1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_ce2x import CrossEncoder2x, get_tokenizer

DATA_ROOT = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification")


class TextPairDataset(Dataset):
    def __init__(self, path: Path):
        self.examples = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append((obj["text_1"], obj["text_2"], int(obj["label"])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts1 = [t1 for t1, t2, lbl in batch]
        texts2 = [t2 for t1, t2, lbl in batch]
        labels = torch.tensor([lbl for t1, t2, lbl in batch], dtype=torch.float32)
        enc = tokenizer(
            texts1, texts2,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"), labels
    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for ids, mask, tti, labels in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if tti is not None:
            tti = tti.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        all_logits.extend(logits.float().cpu().tolist())
        all_labels.extend(labels.tolist())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    preds = (logits_arr > 0).astype(int)
    auc = float(roc_auc_score(labels_arr, logits_arr))
    acc = float(accuracy_score(labels_arr, preds))
    return {"auc": auc, "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_ROOT))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased")
    ap.add_argument("--n-layers", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-patience", type=int, default=2)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(__file__).resolve().parent / "results" / "classification")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    tokenizer = get_tokenizer(args.backbone)
    collate = make_collate(tokenizer, args.max_length)

    train_ds = TextPairDataset(data_dir / "directional_train.jsonl")
    val_ds   = TextPairDataset(data_dir / "directional_val.jsonl")
    test_ds  = TextPairDataset(data_dir / "directional_test.jsonl")
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder2x(backbone=args.backbone, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}  layers={args.n_layers}  params={n_params:,}  device={device}")

    loss_fn = nn.BCEWithLogitsLoss()
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=args.lr)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log_path = out_dir / "train.log"
    log_fh = open(log_path, "w")

    def log(msg):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[config] {vars(args)}")
    log(f"[train] total_steps={total_steps}  warmup={warmup_steps}  "
        f"effective_batch={args.batch_size * args.grad_accum}")

    best_auc = -1.0
    bad_epochs = 0
    history = []
    opt_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, (ids, mask, tti, labels) in enumerate(train_loader):
            ids    = ids.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if tti is not None:
                tti = tti.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask, tti)
                loss = loss_fn(logits, labels) / args.grad_accum

            loss.backward()
            epoch_loss += float(loss.item()) * args.grad_accum

            if (micro_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    log(f"  step={opt_step}/{total_steps}  loss={loss.item() * args.grad_accum:.4f}"
                        f"  lr={scheduler.get_last_lr()[0]:.2e}  epoch={epoch}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_m = evaluate(model, val_loader, device)
        log(f"[epoch {epoch}/{args.epochs}]  train_loss={avg_loss:.4f}  "
            f"val_AUC={val_m['auc']:.4f}  val_Acc={val_m['acc']:.4f}  "
            f"({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in val_m.items()}})

        ckpt = {"model_state_dict": model.state_dict(), "cfg": vars(args),
                "epoch": epoch, **{f"val_{k}": v for k, v in val_m.items()}}
        torch.save(ckpt, out_dir / "checkpoint_latest.pt")

        if val_m["auc"] > best_auc + 1e-4:
            best_auc = val_m["auc"]
            bad_epochs = 0
            torch.save(ckpt, out_dir / "checkpoint_best.pt")
            log(f"  * new best val_AUC={best_auc:.4f}")
        else:
            bad_epochs += 1
            log(f"  no improvement (best={best_auc:.4f}, patience {bad_epochs}/{args.early_stop_patience})")
            if bad_epochs >= args.early_stop_patience:
                log(f"[early stop] triggered at epoch {epoch}")
                break

    # Evaluate best checkpoint on test set
    log("\n[test] loading checkpoint_best.pt ...")
    ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, device)
    log(f"[test]  AUC={test_m['auc']:.4f}  Acc={test_m['acc']:.4f}")
    history.append({"test_auc": test_m["auc"], "test_acc": test_m["acc"]})

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log(f"\n[done]  best_val_AUC={best_auc:.4f}  test_AUC={test_m['auc']:.4f}  test_Acc={test_m['acc']:.4f}")
    log_fh.close()


if __name__ == "__main__":
    main()
