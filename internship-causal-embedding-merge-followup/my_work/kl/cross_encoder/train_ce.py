"""Train the BERT cross-encoder on a specified dataset.

Setup mirrors finetune_eval's BiEncoder training so the comparison is
apples-to-apples:

  * Source pairs:        finetune_eval/results/<dataset>/train_pairs.jsonl
  * Hard negatives:      finetune_eval/results/<dataset>/hard_negatives.npy   (K=4)
  * Validation:          finetune_eval/results/<dataset>/val_pairs.jsonl
                         (mined separately). qrecc has no val split — falls
                         back to a 5 % slice of train.
  * Backbone:            google-bert/bert-base-uncased
  * Loss:                BCE-with-logits over (1 positive + 4 hard-neg) per anchor
  * Optim:               AdamW, lr 2e-5, linear warmup → linear decay

Early stopping: stop if val AUC fails to improve by `--early-stop-threshold`
for `--early-stop-patience` consecutive epochs.

Use --no-cap to train on the FULL train split (override any TRAIN_CAPS).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_ce import CrossEncoder, get_tokenizer
from dataset_ce import PairLabelDataset, load_pairs, make_collate_fn


DEFAULT_DATA_ROOT = Path("/mnt/localssd/base_line/my_work/kl/cross_encoder/data")


def auto_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def validate(model, val_loader, device) -> dict:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    for batch in val_loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        tti = batch["token_type_ids"]
        if tti is not None:
            tti = tti.to(device, non_blocking=True)
        labels = batch["labels"]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        all_logits.extend(logits.float().cpu().tolist())
        all_labels.extend(labels.tolist())

    logits = np.array(all_logits)
    labels = np.array(all_labels)
    auc = float(roc_auc_score(labels, logits))
    group = 5  # 1 positive + 4 hard negatives per anchor
    n_anchors = len(labels) // group
    pos_logits = logits.reshape(n_anchors, group)[:, 0]
    neg_logits = logits.reshape(n_anchors, group)[:, 1:]
    p_at_1 = float(np.mean(pos_logits > neg_logits.max(axis=1)))
    return {"auc": auc, "p_at_1": p_at_1, "n_anchors": n_anchors}


def split_train_val_from_train(pairs, hn, frac: float, seed: int):
    """Hold out a fraction of train as val (for datasets without their own val)."""
    rng = np.random.default_rng(seed)
    N = len(pairs)
    idx = rng.permutation(N)
    n_val = max(int(N * frac), 1)
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    train_hn = hn[train_idx] if hn is not None else None
    val_hn = hn[val_idx] if hn is not None else None
    # Remap HN indices so they point into the *subset* positives list.
    # But this is wrong — hard_negatives.npy indexes into the original positives.
    # For the val split we need negatives whose indices live in the val subset.
    # Simpler & still meaningful: synthesize val hard negs by sampling 4 random
    # other positives within the val split.
    if val_hn is not None:
        N_val = len(val_pairs)
        new_val_hn = np.empty((N_val, val_hn.shape[1]), dtype=np.int64)
        for i in range(N_val):
            picks = rng.choice(N_val - 1, size=val_hn.shape[1], replace=False)
            picks[picks >= i] += 1
            new_val_hn[i] = picks
        val_hn = new_val_hn
    if train_hn is not None:
        # Same remapping problem for train: hn[i] is an index into the original
        # positives list, but we've taken a subset. Need to remap or re-mine.
        # For simplicity, drop hardneg-mined indices that aren't in train_idx
        # and refill with random samples from train_idx.
        N_train = len(train_pairs)
        original_to_new = {orig: new for new, orig in enumerate(train_idx)}
        K = train_hn.shape[1]
        new_train_hn = np.empty((N_train, K), dtype=np.int64)
        for i in range(N_train):
            orig_neg = [int(j) for j in train_hn[i]]
            mapped = [original_to_new[j] for j in orig_neg if j in original_to_new]
            while len(mapped) < K:
                pick = int(rng.integers(0, N_train))
                if pick != i:
                    mapped.append(pick)
            new_train_hn[i] = mapped[:K]
        train_hn = new_train_hn
    return train_pairs, train_hn, val_pairs, val_hn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="Dataset name (aep_causal, followupqg, multiwoz_v24, qrecc, workflow).")
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-threshold", type=float, default=0.001,
                    help="Min val-AUC improvement over current best to count as progress.")
    ap.add_argument("--early-stop-patience", type=int, default=2,
                    help="Stop if this many consecutive epochs fail to improve.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir. Default: results/<dataset>/")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_ROOT),
                    help="Directory with <dataset>/{train,val,test}_pairs.jsonl + *hard_negatives.npy")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(HERE / "results" / args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data -----------------------------------------------------------------
    ds_dir = Path(args.data_dir) / args.dataset
    train_pairs_path = ds_dir / "train_pairs.jsonl"
    train_hn_path    = ds_dir / "hard_negatives.npy"
    val_pairs_path   = ds_dir / "val_pairs.jsonl"
    val_hn_path      = ds_dir / "val_hard_negatives.npy"

    print(f"[data] train from {train_pairs_path}")
    if not train_pairs_path.exists():
        raise FileNotFoundError(
            f"missing train pairs at {train_pairs_path}.\n"
            f"Re-mine with: mine_hard_negatives.py --dataset {args.dataset} --split train --k 4 --no-cap"
        )
    train_pairs_full = load_pairs(train_pairs_path)
    print(f"        {len(train_pairs_full)} train pairs")
    train_hn_full = np.load(train_hn_path) if train_hn_path.exists() else None
    if train_hn_full is None:
        raise FileNotFoundError(f"missing hard_negatives.npy at {train_hn_path}")
    print(f"        hard negatives {train_hn_full.shape}")
    assert train_hn_full.shape[0] == len(train_pairs_full)

    if val_pairs_path.exists() and val_hn_path.exists():
        val_pairs = load_pairs(val_pairs_path)
        val_hn = np.load(val_hn_path)
        train_pairs, train_hn = train_pairs_full, train_hn_full
        print(f"[data] val from   {val_pairs_path}    {len(val_pairs)} pairs")
    else:
        print(f"[data] no mined val split for {args.dataset}; holding out 5% of train as val")
        train_pairs, train_hn, val_pairs, val_hn = split_train_val_from_train(
            train_pairs_full, train_hn_full, frac=0.05, seed=args.seed
        )
        print(f"        train={len(train_pairs)}  val={len(val_pairs)}")

    tokenizer = get_tokenizer(args.backbone)
    collate = make_collate_fn(tokenizer, args.max_length)
    train_ds = PairLabelDataset(train_pairs, train_hn)
    val_ds   = PairLabelDataset(val_pairs,   val_hn)
    print(f"        flattened: train rows={len(train_ds)}  val rows={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=False,
    )

    # ---- Model ----------------------------------------------------------------
    device = auto_device()
    model = CrossEncoder(backbone=args.backbone).to(device)
    print(f"[model] backbone={args.backbone}  params={sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.BCEWithLogitsLoss()
    no_decay = ("bias", "LayerNorm.weight")
    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = torch.optim.AdamW(
        [{"params": params_decay,   "weight_decay": args.weight_decay},
         {"params": params_nodecay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Train ----------------------------------------------------------------
    log_path = out_dir / "train.log"
    log_fh = open(log_path, "w")
    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[train] dataset={args.dataset} total_steps={total_steps} "
        f"warmup={warmup_steps} batch_size={args.batch_size} epochs={args.epochs} "
        f"max_length={args.max_length}  early_stop=(thresh {args.early_stop_threshold}, "
        f"patience {args.early_stop_patience})")

    best_auc = -1.0
    bad_epochs = 0
    history: list[dict] = []
    step = 0
    t0 = time.time()
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            tti = batch["token_type_ids"]
            if tti is not None:
                tti = tti.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask, tti)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                log(f"  step={step:6d}/{total_steps}  loss={loss.item():.4f}  "
                    f"lr={lr_now:.2e}  epoch={epoch}")

        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        val_metrics = validate(model, val_loader, device)
        improved = val_metrics["auc"] > best_auc + args.early_stop_threshold
        log(f"[epoch {epoch}/{args.epochs}]  train_loss={avg_loss:.4f}  "
            f"val_AUC={val_metrics['auc']:.4f}  val_P@1={val_metrics['p_at_1']:.4f}  "
            f"({time.time() - t0:.0f}s)")
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auc": val_metrics["auc"],
            "val_p_at_1": val_metrics["p_at_1"],
            "elapsed_s": time.time() - t0,
        })

        torch.save({
            "model_state_dict": model.state_dict(),
            "cfg": vars(args),
            "epoch": epoch,
            "val_auc": val_metrics["auc"],
            "val_p_at_1": val_metrics["p_at_1"],
        }, out_dir / "checkpoint_latest.pt")

        if improved:
            best_auc = val_metrics["auc"]
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": vars(args),
                "epoch": epoch,
                "val_auc": val_metrics["auc"],
                "val_p_at_1": val_metrics["p_at_1"],
            }, out_dir / "checkpoint_best.pt")
            log(f"  * new best val_AUC={best_auc:.4f}, saved checkpoint_best.pt")
        else:
            bad_epochs += 1
            log(f"  no improvement (best={best_auc:.4f}, bad_epochs={bad_epochs}/{args.early_stop_patience})")
            if bad_epochs >= args.early_stop_patience:
                log(f"[early stop] no val-AUC improvement for {bad_epochs} epochs.")
                stopped_early = True
                break

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    log(f"\n[done]  best val_AUC={best_auc:.4f}  stopped_early={stopped_early}")
    log(f"  best:   {out_dir / 'checkpoint_best.pt'}")
    log(f"  latest: {out_dir / 'checkpoint_latest.pt'}")
    log_fh.close()


if __name__ == "__main__":
    main()
