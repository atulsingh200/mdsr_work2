"""Train CrossEncoder2x (24-layer stacked BERT, ~194 M params) as a binary classifier.

Comparable to the ensemble of two 1x cross-encoders — same total parameter budget,
single model.  Works on any pre-built classification dataset with label 0/1 rows.

Typical usage
-------------
  # semantic hard-negative dataset
  python3 -m src.classifier.crossencoder.train_ce2x \\
      --data-dir data/aep_causal_classification_hard_neg_semantic \\
      --out-dir runs/crossencoder2x --run-name ce2x_semantic \\
      --epochs 3 --batch-size 32

  # in-domain hard-negative dataset
  python3 -m src.classifier.crossencoder.train_ce2x \\
      --data-dir data/aep_causal_classification_hard_neg_indomain \\
      --out-dir runs/crossencoder2x --run-name ce2x_indomain \\
      --epochs 3 --batch-size 32
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
from transformers import get_linear_schedule_with_warmup

# ── path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]          # src/classifier/crossencoder -> repo root
# All helper modules are in the same directory as this script (training/)
_TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TRAINING_DIR))

from data import CECollate, CEPairDataset, _parse_step          # noqa: E402
from train_utils import device_auto, make_logger, set_seed      # noqa: E402

from model_ce2x import CrossEncoder2x, get_tokenizer  # noqa: E402  (after sys.path)


# ── helpers (mirrors train_ce.py) ─────────────────────────────────────────────

def scan_max_seq_len(jsonls: list[Path], tokenizer) -> int:
    """Longest joint-tokenised length ([CLS] text_1 [SEP] text_2 [SEP]) across splits."""
    mx = 0
    for p in jsonls:
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
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
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    for batch in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        labels = batch["labels"].to(device, non_blocking=True)
        with ctx():
            logits = model(enc["input_ids"], enc.get("attention_mask"),
                           enc.get("token_type_ids"))
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
                k1 = f"{dataset.step_key}_1"
                k2 = f"{dataset.step_key}_2"
                rows.append({
                    "url_1": r.get("url_1", ""), "url_2": r.get("url_2", ""),
                    "sub_1": r.get("sub_1", ""), "sub_2": r.get("sub_2", ""),
                    "step_1": r.get(k1), "step_2": r.get(k2),
                    "label": int(lbls[j]), "prob": float(probs[j]),
                })

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels).astype(int)
    preds = (probs_arr > 0.5).astype(int)
    acc = float((preds == labels_arr).mean())
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        auc = float(roc_auc_score(labels_arr, probs_arr))
        f1 = float(f1_score(labels_arr, preds))
    except Exception:
        auc = f1 = float("nan")
    return ({"loss": total_loss / max(total_n, 1), "acc": acc,
             "auc": auc, "f1": f1, "n": total_n}, rows)


def _per_step_pair(preds_rows: list[dict]) -> dict:
    pair_stats: dict = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in preds_rows:
        s1 = _parse_step(row.get("step_1"))
        s2 = _parse_step(row.get("step_2"))
        if s1 is None or s2 is None:
            continue
        key = (min(s1, s2), max(s1, s2))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int((row["prob"] > 0.5) == (row["label"] == 1))
    return {f"S{a}-S{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
            for (a, b), s in sorted(pair_stats.items())}


def _write_preds(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train CrossEncoder2x (24-layer BERT, ~194 M params) on a binary classification dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True,
                    help="Directory with <split-prefix>{train,val,test}.jsonl")
    ap.add_argument("--split-prefix", default="directional_",
                    help="Filename prefix for split files: <prefix>{train,val,test}.jsonl. "
                         "Default 'directional_' matches new_aep_workflow_scrap; pass '' for "
                         "datasets named plain train/val/test.jsonl (e.g. ajo_newstyle).")
    ap.add_argument("--out-dir", default="runs/crossencoder2x")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased")
    ap.add_argument("--n-layers", type=int, default=24,
                    help="Number of transformer layers (24 = 2× BERT-base, matches BiEncoder param budget).")
    ap.add_argument("--native-backbone", action="store_true",
                    help="Use AutoModel.from_pretrained directly instead of the BERT-stacking logic. "
                         "Required for non-BERT backbones such as DeBERTa-v3-large.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="Override joint sequence length cap; default = auto-scan.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--model-ce2x-dir", default=str(_CE2X_MODEL_DIR),
                    help="Directory that contains model_ce2x.py.")
    ap.add_argument("--step-key", default="tier",
                    help="Field-name prefix for step/tier columns. "
                         "Use 'tier' for datasets with tier_1/tier_2 (int), "
                         "'step' for datasets with step_1/step_2 (e.g. 't2' strings).")
    args = ap.parse_args()

    # allow overriding the model dir at runtime
    if args.model_ce2x_dir != str(_CE2X_MODEL_DIR):
        sys.path.insert(0, args.model_ce2x_dir)

    set_seed(args.seed)
    device = device_auto()
    use_bf16 = device.type == "cuda"

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}  device={device}  bf16={use_bf16}")

    tokenizer = get_tokenizer(args.backbone)
    log.info(f"tokenizer <- {args.backbone}")

    data_dir = Path(args.data_dir)
    jsonls = [data_dir / f"{args.split_prefix}{s}.jsonl" for s in ("train", "val", "test")]

    if args.max_seq_len is None:
        log.info("scanning max joint token length across all splits …")
        t0 = time.time()
        mx = scan_max_seq_len(jsonls, tokenizer)
        model_cap = getattr(tokenizer, "model_max_length", 512)
        if model_cap is None or model_cap > 100_000:
            model_cap = 512
        args.max_seq_len = min(model_cap, max(128, int(np.ceil(mx * 1.05)) + 2))
        log.info(f"  observed max={mx}  -> max_seq_len={args.max_seq_len}  (scan {time.time()-t0:.1f}s)")
    else:
        log.info(f"max_seq_len={args.max_seq_len} (user-provided)")

    log.info("loading datasets …")
    ds_train = CEPairDataset(jsonls[0], step_key=args.step_key)
    ds_val   = CEPairDataset(jsonls[1], step_key=args.step_key)
    ds_test  = CEPairDataset(jsonls[2], step_key=args.step_key)
    log.info(f"  train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    collate = CECollate(tokenizer, args.max_seq_len)
    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)

    log.info(f"building CrossEncoder2x  backbone={args.backbone}  n_layers={args.n_layers}  native={args.native_backbone} …")
    model = CrossEncoder2x(backbone=args.backbone, n_layers=args.n_layers,
                           dropout=args.dropout, native_backbone=args.native_backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  total params={n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss()

    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=args.lr)

    total_steps  = (len(dl_train) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(f"  total_steps={total_steps}  warmup_steps={warmup_steps}")

    cfg = {
        "backbone": args.backbone, "n_layers": args.n_layers,
        "native_backbone": args.native_backbone,
        "data_dir": str(args.data_dir), "split_prefix": args.split_prefix,
        "run_name": args.run_name,
        "model_type": "cross_encoder_2x",
        "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum, "eff_batch_size": args.batch_size * args.grad_accum,
        "epochs": args.epochs, "warmup_frac": args.warmup_frac,
        "grad_clip": args.grad_clip, "dropout": args.dropout, "seed": args.seed,
        "max_seq_len": args.max_seq_len, "device": str(device), "bf16": use_bf16,
        "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "n_params": n_params,
        "pool": "cls", "input_form": "[CLS] text_1 [SEP] text_2 [SEP]",
        "head": "Linear(hidden, 1)", "loss": "BCEWithLogitsLoss",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    best_val_acc = -1.0
    opt_step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = running_correct = running_total = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(dl_train, 1):
            enc    = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device, non_blocking=True)

            with ctx():
                logits = model(enc["input_ids"], enc.get("attention_mask"),
                               enc.get("token_type_ids"))
                loss = loss_fn(logits, labels) / args.grad_accum

            loss.backward()
            bs = labels.size(0)
            running_loss    += float(loss.item()) * args.grad_accum * bs
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            running_correct += (preds == labels).sum().item()
            running_total   += bs

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    log.info(f"  e{epoch} opt_step={opt_step}/{total_steps}  "
                             f"loss={running_loss / running_total:.4f}  "
                             f"acc={running_correct / running_total:.4f}  "
                             f"lr={sched.get_last_lr()[0]:.2e}")

        val_metrics, _ = evaluate(model, dl_val, device, use_bf16, loss_fn)
        log.info(f"[epoch {epoch}/{args.epochs}]  "
                 f"train_loss={running_loss / running_total:.4f}  "
                 f"train_acc={running_correct / running_total:.4f}  "
                 f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
                 f"val_auc={val_metrics['auc']:.4f}  elapsed={time.time()-t_epoch:.1f}s")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "cfg": cfg}
        torch.save(ckpt, run_dir / "checkpoint_latest.pt")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(ckpt, run_dir / "best.pt")
            log.info(f"  * saved best.pt  val_acc={best_val_acc:.4f}")

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
               run_dir / "final.pt")
    log.info(f"saved final.pt  total_time={time.time()-t_start:.0f}s")

    # ── reload best for val + test eval ───────────────────────────────────────
    log.info("loading best.pt for final val/test evaluation …")
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    val_metrics, val_rows = evaluate(model, dl_val, device, use_bf16, loss_fn,
                                     dataset=ds_val, return_predictions=True)
    _write_preds(run_dir / "val_predictions.jsonl", val_rows)
    log.info(f"[val]  acc={val_metrics['acc']:.4f}  auc={val_metrics['auc']:.4f}  "
             f"f1={val_metrics['f1']:.4f}")

    test_metrics, test_rows = evaluate(model, dl_test, device, use_bf16, loss_fn,
                                       dataset=ds_test, return_predictions=True)
    log.info(f"[test] loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
             f"auc={test_metrics['auc']:.4f}  f1={test_metrics['f1']:.4f}")

    per_pair = _per_step_pair(test_rows)
    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_step_pair": per_pair}, indent=2))
    _write_preds(run_dir / "test_predictions.jsonl", test_rows)

    log.info(f"[done] best_val_acc={best_val_acc:.4f}  "
             f"test_acc={test_metrics['acc']:.4f}  test_auc={test_metrics['auc']:.4f}  "
             f"results in {run_dir}")


if __name__ == "__main__":
    main()
