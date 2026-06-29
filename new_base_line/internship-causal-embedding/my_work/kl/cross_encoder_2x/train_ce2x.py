"""Train the 2x CrossEncoder (24-layer BERT) on a specified dataset.

  * Source pairs / hard negatives:  same data layout as cross_encoder/
  * Backbone init:  bert-base-uncased stacked to 24 layers (~194 M params)
  * Loss:           Listwise softmax (cross-entropy over 1 pos + K hard-neg per anchor)
                    — directly optimises ranking, calibrates scores across anchors
  * Optim:          AdamW, lr 2e-5, linear warmup → linear decay
  * Early stopping: stop if val P@1 fails to improve by threshold for patience epochs

Parameter count comparison
--------------------------
  BiEncoder (2× bert-base, untied):   ~218.96 M
  CrossEncoder-2x (24-layer bert):    ~194.54 M   (shared embeddings explain the gap)
  CrossEncoder-1x (12-layer bert):    ~109.48 M
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CE_DIR = HERE.parent / "cross_encoder"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CE_DIR))

from model_ce2x import CrossEncoder2x, get_tokenizer
from dataset_ce import load_pairs, make_collate_fn  # reuse 1x helpers


DEFAULT_DATA_ROOT = Path("/mnt/localssd/base_line/my_work/kl/cross_encoder/data")


def auto_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AnchorGroupDataset(torch.utils.data.Dataset):
    """Each item is one anchor's full group: (anchor, positive) + negatives.

    Supports mixed negatives: n_hard hard negatives (from hard_negatives.npy) +
    n_random random negatives (sampled from other positives in the dataset).
    Total group size = 1 + n_hard + n_random.

    Returns a list of (anchor_text, candidate_text, label) tuples of length (1+n_hard+n_random).
    The DataLoader collate stacks these into a flat batch of size batch_size*(1+n_hard+n_random),
    preserving the positive-first order required by listwise_loss.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        hard_negatives: np.ndarray,
        n_hard: int | None = None,
        n_random: int = 0,
        seed: int = 42,
    ) -> None:
        self.anchors   = [a for a, _ in pairs]
        self.positives = [p for _, p in pairs]
        self.hn        = hard_negatives  # (N, K)
        self.n_hard    = self.hn.shape[1] if n_hard is None else min(n_hard, self.hn.shape[1])
        self.n_random  = n_random
        self.rng       = np.random.default_rng(seed)
        self.N         = len(self.anchors)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> list[tuple[str, str, int]]:
        group = [(self.anchors[i], self.positives[i], 1)]
        for j in self.hn[i, :self.n_hard]:
            group.append((self.anchors[i], self.positives[int(j)], 0))
        if self.n_random > 0:
            # sample random negatives from any other index
            candidates = self.rng.choice(self.N - 1, size=self.n_random, replace=False)
            candidates[candidates >= i] += 1
            for j in candidates:
                group.append((self.anchors[i], self.positives[int(j)], 0))
        return group


def make_group_collate_fn(tokenizer, max_length: int):
    """Collate a list-of-groups into a flat tokenized batch.

    Input:  list of N groups, each group = list of (a, b, label) of length G.
    Output: flat batch of N*G rows, groups are contiguous (pos first in each group).
    """
    def collate(batch: list[list[tuple[str, str, int]]]):
        flat       = [item for group in batch for item in group]
        anchors    = [x[0] for x in flat]
        candidates = [x[1] for x in flat]
        labels     = torch.tensor([x[2] for x in flat], dtype=torch.float)
        enc = tokenizer(
            anchors, candidates,
            padding=True, truncation="longest_first",
            max_length=max_length, return_tensors="pt",
            return_token_type_ids=True,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids"),
            "labels":         labels,
        }
    return collate


def listwise_loss(logits: torch.Tensor, group: int) -> torch.Tensor:
    """Listwise softmax loss over (1 positive + K hard-negatives) per anchor.

    logits: flat (B * group,) tensor; every group rows starting at 0 is the positive.
    target is always index 0 in each group.
    """
    scores = logits.view(-1, group)           # (B, group)
    target = torch.zeros(scores.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(scores, target)


@torch.no_grad()
def validate(model, val_loader, device, group: int = 5) -> dict:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    for batch in val_loader:
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        tti  = batch["token_type_ids"]
        if tti is not None:
            tti = tti.to(device, non_blocking=True)
        labels = batch["labels"]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        all_logits.extend(logits.float().cpu().tolist())
        all_labels.extend(labels.tolist())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    auc    = float(roc_auc_score(labels_arr, logits_arr))
    n_anchors  = len(labels_arr) // group
    pos_logits = logits_arr.reshape(n_anchors, group)[:, 0]
    neg_logits = logits_arr.reshape(n_anchors, group)[:, 1:]
    p_at_1     = float(np.mean(pos_logits > neg_logits.max(axis=1)))
    return {"auc": auc, "p_at_1": p_at_1, "n_anchors": n_anchors}


def split_train_val_from_train(pairs, hn, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    N   = len(pairs)
    idx = rng.permutation(N)
    n_val     = max(int(N * frac), 1)
    val_idx   = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    train_hn = hn[train_idx] if hn is not None else None
    val_hn   = hn[val_idx]   if hn is not None else None

    if val_hn is not None:
        N_val = len(val_pairs)
        new_val_hn = np.empty((N_val, val_hn.shape[1]), dtype=np.int64)
        for i in range(N_val):
            picks = rng.choice(N_val - 1, size=val_hn.shape[1], replace=False)
            picks[picks >= i] += 1
            new_val_hn[i] = picks
        val_hn = new_val_hn

    if train_hn is not None:
        N_train = len(train_pairs)
        original_to_new = {orig: new for new, orig in enumerate(train_idx)}
        K = train_hn.shape[1]
        new_train_hn = np.empty((N_train, K), dtype=np.int64)
        for i in range(N_train):
            orig_neg = [int(j) for j in train_hn[i]]
            mapped   = [original_to_new[j] for j in orig_neg if j in original_to_new]
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
    ap.add_argument("--n-layers", type=int, default=24,
                    help="Number of transformer layers in the stacked backbone (default 24).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Anchors per step. Actual GPU rows = batch_size * group (default 32*5=160).")
    ap.add_argument("--val-batch-size", type=int, default=64,
                    help="Flat rows per val step (no grad, so can be 2-4x train batch_size*group).")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-threshold", type=float, default=0.002,
                    help="Min val-P@1 improvement to count as progress.")
    ap.add_argument("--early-stop-patience", type=int, default=3,
                    help="Stop after this many non-improving epochs.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir. Default: results/<dataset>/")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_ROOT),
                    help="Directory with <dataset>/{train,val,test}_pairs.jsonl + *hard_negatives.npy")
    ap.add_argument("--n-hard", type=int, default=None,
                    help="Hard negatives per anchor. Default: use all available in hard_negatives.npy.")
    ap.add_argument("--n-random", type=int, default=0,
                    help="Random negatives per anchor sampled from other positives (default 0).")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(HERE / "results" / args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_dir           = Path(args.data_dir) / args.dataset
    train_pairs_path = ds_dir / "train_pairs.jsonl"
    train_hn_path    = ds_dir / "hard_negatives.npy"
    val_pairs_path   = ds_dir / "val_pairs.jsonl"
    val_hn_path      = ds_dir / "val_hard_negatives.npy"

    print(f"[data] train from {train_pairs_path}")
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"missing train pairs at {train_pairs_path}")
    train_pairs_full = load_pairs(train_pairs_path)
    print(f"        {len(train_pairs_full)} train pairs")
    train_hn_full = np.load(train_hn_path) if train_hn_path.exists() else None
    if train_hn_full is None:
        raise FileNotFoundError(f"missing hard_negatives.npy at {train_hn_path}")
    print(f"        hard negatives {train_hn_full.shape}")
    assert train_hn_full.shape[0] == len(train_pairs_full)

    if val_pairs_path.exists() and val_hn_path.exists():
        val_pairs = load_pairs(val_pairs_path)
        val_hn    = np.load(val_hn_path)
        train_pairs, train_hn = train_pairs_full, train_hn_full
        print(f"[data] val from   {val_pairs_path}    {len(val_pairs)} pairs")
    else:
        print(f"[data] no mined val split for {args.dataset}; holding out 5% of train as val")
        train_pairs, train_hn, val_pairs, val_hn = split_train_val_from_train(
            train_pairs_full, train_hn_full, frac=0.05, seed=args.seed
        )
        print(f"        train={len(train_pairs)}  val={len(val_pairs)}")

    tokenizer  = get_tokenizer(args.backbone)
    n_hard_eff = train_hn.shape[1] if args.n_hard is None else min(args.n_hard, train_hn.shape[1])
    group      = 1 + n_hard_eff + args.n_random  # 1 pos + hard + random
    val_group  = 1 + val_hn.shape[1]              # val stays flat: 1 pos + all K hard-neg

    train_ds      = AnchorGroupDataset(train_pairs, train_hn,
                                       n_hard=args.n_hard, n_random=args.n_random,
                                       seed=args.seed)
    group_collate = make_group_collate_fn(tokenizer, args.max_length)

    from dataset_ce import PairLabelDataset
    val_ds       = PairLabelDataset(val_pairs, val_hn)
    flat_collate = make_collate_fn(tokenizer, args.max_length)

    print(f"        train anchors={len(train_ds)}  val rows={len(val_ds)}  "
          f"train_group={group}  val_group={val_group}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=group_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=flat_collate, drop_last=False,
    )

    device   = auto_device()
    model    = CrossEncoder2x(backbone=args.backbone, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] backbone={args.backbone}  layers={args.n_layers}  params={n_params:,}")

    no_decay       = ("bias", "LayerNorm.weight")
    params_decay   = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)]
    optimizer = torch.optim.AdamW(
        [{"params": params_decay,   "weight_decay": args.weight_decay},
         {"params": params_nodecay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log_path = out_dir / "train.log"
    log_fh   = open(log_path, "w")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[train] dataset={args.dataset} total_steps={total_steps} "
        f"warmup={warmup_steps} batch_size={args.batch_size} "
        f"epochs={args.epochs} max_length={args.max_length} "
        f"params={n_params:,}  loss=listwise_softmax(group={group})  "
        f"early_stop=(thresh {args.early_stop_threshold}, patience {args.early_stop_patience})")

    best_p1    = -1.0
    bad_epochs = 0
    history: list[dict] = []
    step = 0
    t0   = time.time()
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps    = 0

        for batch in train_loader:
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            tti  = batch["token_type_ids"]
            if tti is not None:
                tti = tti.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask, tti)
                loss   = listwise_loss(logits, group)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step           += 1
            epoch_loss_sum += float(loss.item())
            epoch_steps    += 1

            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                log(f"  step={step:6d}/{total_steps}  loss={loss.item():.4f}  "
                    f"lr={lr_now:.2e}  epoch={epoch}")

        avg_loss    = epoch_loss_sum / max(epoch_steps, 1)
        val_metrics = validate(model, val_loader, device, group=val_group)
        improved    = val_metrics["p_at_1"] > best_p1 + args.early_stop_threshold
        log(f"[epoch {epoch}/{args.epochs}]  train_loss={avg_loss:.4f}  "
            f"val_AUC={val_metrics['auc']:.4f}  val_P@1={val_metrics['p_at_1']:.4f}  "
            f"({time.time() - t0:.0f}s)")
        history.append({
            "epoch":      epoch,
            "train_loss": avg_loss,
            "val_auc":    val_metrics["auc"],
            "val_p_at_1": val_metrics["p_at_1"],
            "elapsed_s":  time.time() - t0,
        })

        torch.save({
            "model_state_dict": model.state_dict(),
            "cfg":        vars(args),
            "epoch":      epoch,
            "val_auc":    val_metrics["auc"],
            "val_p_at_1": val_metrics["p_at_1"],
        }, out_dir / "checkpoint_latest.pt")

        if improved:
            best_p1    = val_metrics["p_at_1"]
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg":        vars(args),
                "epoch":      epoch,
                "val_auc":    val_metrics["auc"],
                "val_p_at_1": val_metrics["p_at_1"],
            }, out_dir / "checkpoint_best.pt")
            log(f"  * new best val_P@1={best_p1:.4f}, saved checkpoint_best.pt")
        else:
            bad_epochs += 1
            log(f"  no improvement (best_P@1={best_p1:.4f}, bad_epochs={bad_epochs}/{args.early_stop_patience})")
            if bad_epochs >= args.early_stop_patience:
                log(f"[early stop] no val-P@1 improvement for {bad_epochs} epochs.")
                stopped_early = True
                break

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    log(f"\n[done]  best val_P@1={best_p1:.4f}  stopped_early={stopped_early}")
    log(f"  best:   {out_dir / 'checkpoint_best.pt'}")
    log(f"  latest: {out_dir / 'checkpoint_latest.pt'}")
    log_fh.close()


if __name__ == "__main__":
    main()
