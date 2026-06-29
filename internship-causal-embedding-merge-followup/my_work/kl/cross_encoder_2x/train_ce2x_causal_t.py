"""Train CrossEncoder2x (ranking) on aep_dataset_t.

Pipeline (all in one script):
  1. Load aep_dataset_t JSONL (text_1 / text_2 / label); keep only label=1
     rows as anchor/positive pairs.
  2. Mine hard negatives OR use random negatives (--neg-type hard|random).
  3. Train CrossEncoder2x with BCE loss, early stopping on val AUC.
  4. Evaluate best checkpoint on the test split (AUC + P@1).

aep_dataset_t is derived from aep_dataset with every label=0 row having
text_1 and text_2 swapped, testing sensitivity to pair order.

Usage examples
--------------
  # hard negatives (default, K=4)
  python train_ce2x_causal_t.py --neg-type hard

  # random negatives
  python train_ce2x_causal_t.py --neg-type random

  # change K, miner model, etc.
  python train_ce2x_causal_t.py --neg-type hard --k 8 \\
      --miner-model sentence-transformers/all-MiniLM-L6-v2
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
CE_DIR = HERE.parent / "cross_encoder"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CE_DIR))

from model_ce2x import CrossEncoder2x, get_tokenizer
from dataset_ce import PairLabelDataset, make_collate_fn

RAW_DATA_ROOT = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_dataset_t")

# ── 1. data preparation ────────────────────────────────────────────────────────

def load_raw_pairs(path: Path) -> list[tuple[str, str]]:
    """Load all rows as (anchor, positive) pairs — aep_dataset_t is all label=1."""
    pairs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            anchor   = (row.get("text_1") or "").strip()
            positive = (row.get("text_2") or "").strip()
            if anchor and positive:
                pairs.append((anchor, positive))
    return pairs


def mine_hard_negatives(pairs: list[tuple[str, str]], k: int, args) -> np.ndarray:
    """Encode positives with a frozen sentence encoder and return top-K indices."""
    import torch as _torch

    # lazy import — only needed for hard-neg mode
    BIENCODER_ROOT = HERE.parent.parent.parent.parent / "automation" / "internship-causal-embedding"
    sys.path.insert(0, str(BIENCODER_ROOT))
    sys.path.insert(0, str(BIENCODER_ROOT / "src"))
    from evaluation.retrievers import PretrainedRetriever

    print(f"[mine] loading miner: {args.miner_model}")
    miner = PretrainedRetriever(
        model_name=args.miner_model,
        pooling=args.miner_pooling,
        max_seq_length=args.miner_max_seq_length,
        batch_size=args.miner_batch_size,
    )

    positives = [p for _, p in pairs]
    N = len(positives)
    print(f"[mine] encoding {N} positives …")
    t0 = time.time()
    pos_emb = miner.encode_candidates(positives)   # (N, D)
    print(f"[mine] encoded in {time.time()-t0:.1f}s  dim={pos_emb.shape[1]}")

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    pos_t = _torch.from_numpy(pos_emb).to(device)
    chunk = args.miner_chunk_size
    hard_idx = np.empty((N, k), dtype=np.int32)

    t1 = time.time()
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sims = pos_t[start:end] @ pos_t.T
        rows = _torch.arange(start, end, device=device)
        sims[_torch.arange(end - start, device=device), rows] = float("-inf")
        _, top_idx = sims.topk(k, dim=1, largest=True, sorted=True)
        hard_idx[start:end] = top_idx.cpu().numpy()
    print(f"[mine] k-NN done in {time.time()-t1:.1f}s")
    return hard_idx


def make_random_negatives(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        picks = rng.choice(n - 1, size=k, replace=False)
        picks[picks >= i] += 1
        idx[i] = picks
    return idx


# ── 2. validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, group: int) -> dict:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    for batch in val_loader:
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        tti  = batch["token_type_ids"]
        if tti is not None:
            tti = tti.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        all_logits.extend(logits.float().cpu().tolist())
        all_labels.extend(batch["labels"].tolist())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    auc = float(roc_auc_score(labels_arr, logits_arr))
    n_anchors = len(labels_arr) // group
    pos_logits = logits_arr.reshape(n_anchors, group)[:, 0]
    neg_logits = logits_arr.reshape(n_anchors, group)[:, 1:]
    p_at_1 = float(np.mean(pos_logits > neg_logits.max(axis=1)))
    return {"auc": auc, "p_at_1": p_at_1, "n_anchors": n_anchors}


# ── 3. main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # negative strategy
    ap.add_argument("--neg-type", choices=["hard", "random"], default="hard",
                    help="Use hard negatives (mined by sentence encoder) or random negatives.")
    ap.add_argument("--k", type=int, default=4,
                    help="Number of negatives per positive (default 4).")
    # data
    ap.add_argument("--raw-data-dir", default=str(RAW_DATA_ROOT))
    ap.add_argument("--out-dir", default=None,
                    help="Output dir. Default: results/aep_causal_t/<neg-type>/")
    # miner (only used when --neg-type hard)
    ap.add_argument("--miner-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--miner-pooling", default="mean")
    ap.add_argument("--miner-max-seq-length", type=int, default=256)
    ap.add_argument("--miner-batch-size", type=int, default=256)
    ap.add_argument("--miner-chunk-size", type=int, default=1024)
    # model
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased")
    ap.add_argument("--n-layers", type=int, default=24)
    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-threshold", type=float, default=0.001)
    ap.add_argument("--early-stop-patience", type=int, default=2)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(HERE / "results" / "aep_dataset_t" / args.neg_type)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw_dir = Path(args.raw_data_dir)

    # ── load pairs ──
    print("[data] loading raw pairs …")
    train_pairs = load_raw_pairs(raw_dir / "directional_train.jsonl")
    val_pairs   = load_raw_pairs(raw_dir / "directional_val.jsonl")
    test_pairs  = load_raw_pairs(raw_dir / "directional_test.jsonl")
    print(f"[data] train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    # ── negatives ──
    group = args.k + 1  # 1 positive + K negatives

    if args.neg_type == "hard":
        print("[neg] mining hard negatives …")
        train_hn = mine_hard_negatives(train_pairs, args.k, args)
        val_hn   = mine_hard_negatives(val_pairs,   args.k, args)
        test_hn  = mine_hard_negatives(test_pairs,  args.k, args)
        np.save(out_dir / "train_hard_negatives.npy", train_hn)
        np.save(out_dir / "val_hard_negatives.npy",   val_hn)
        np.save(out_dir / "test_hard_negatives.npy",  test_hn)
    else:
        print("[neg] generating random negatives …")
        train_hn = make_random_negatives(len(train_pairs), args.k, args.seed)
        val_hn   = make_random_negatives(len(val_pairs),   args.k, args.seed + 1)
        test_hn  = make_random_negatives(len(test_pairs),  args.k, args.seed + 2)

    # ── datasets & loaders ──
    tokenizer = get_tokenizer(args.backbone)
    collate   = make_collate_fn(tokenizer, args.max_length)

    train_ds = PairLabelDataset(train_pairs, train_hn)
    val_ds   = PairLabelDataset(val_pairs,   val_hn)
    test_ds  = PairLabelDataset(test_pairs,  test_hn)
    print(f"[data] flattened rows — train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)

    # ── model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CrossEncoder2x(backbone=args.backbone, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}  layers={args.n_layers}  params={n_params:,}  device={device}")

    # ── optimizer & scheduler ──
    loss_fn  = nn.BCEWithLogitsLoss()
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=args.lr)

    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── logging ──
    log_fh = open(out_dir / "train.log", "w")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[config] dataset=aep_dataset_t  neg_type={args.neg_type}  k={args.k}  "
        f"epochs={args.epochs}  batch={args.batch_size}  grad_accum={args.grad_accum}  "
        f"eff_batch={args.batch_size * args.grad_accum}  max_len={args.max_length}  "
        f"total_steps={total_steps}  warmup={warmup_steps}  params={n_params:,}")

    # ── training loop ──
    best_auc   = -1.0
    bad_epochs = 0
    history: list[dict] = []
    opt_step = 0
    t0 = time.time()
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device, non_blocking=True)
            mask   = batch["attention_mask"].to(device, non_blocking=True)
            tti    = batch["token_type_ids"]
            if tti is not None:
                tti = tti.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask, tti)
                loss   = loss_fn(logits, labels) / args.grad_accum

            loss.backward()
            epoch_loss  += float(loss.item()) * args.grad_accum
            epoch_steps += 1

            if (micro_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    log(f"  opt_step={opt_step:6d}/{total_steps}  "
                        f"loss={loss.item() * args.grad_accum:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  epoch={epoch}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        val_m = validate(model, val_loader, device, group)
        improved = val_m["auc"] > best_auc + args.early_stop_threshold

        log(f"[epoch {epoch}/{args.epochs}]  train_loss={avg_loss:.4f}  "
            f"val_AUC={val_m['auc']:.4f}  val_P@1={val_m['p_at_1']:.4f}  "
            f"({time.time()-t0:.0f}s)")
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auc": val_m["auc"],
            "val_p_at_1": val_m["p_at_1"],
            "elapsed_s": time.time() - t0,
        })

        ckpt = {"model_state_dict": model.state_dict(), "cfg": vars(args),
                "epoch": epoch, "val_auc": val_m["auc"], "val_p_at_1": val_m["p_at_1"]}
        torch.save(ckpt, out_dir / "checkpoint_latest.pt")

        if improved:
            best_auc   = val_m["auc"]
            bad_epochs = 0
            torch.save(ckpt, out_dir / "checkpoint_best.pt")
            log(f"  * new best val_AUC={best_auc:.4f}, saved checkpoint_best.pt")
        else:
            bad_epochs += 1
            log(f"  no improvement (best={best_auc:.4f}, bad_epochs={bad_epochs}/{args.early_stop_patience})")
            if bad_epochs >= args.early_stop_patience:
                log(f"[early stop] triggered at epoch {epoch}")
                stopped_early = True
                break

    # ── test evaluation on best checkpoint ──
    log("\n[test] loading checkpoint_best.pt …")
    best_ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_m = validate(model, test_loader, device, group)
    log(f"[test]  AUC={test_m['auc']:.4f}  P@1={test_m['p_at_1']:.4f}  "
        f"n_anchors={test_m['n_anchors']}")
    history.append({"test_auc": test_m["auc"], "test_p_at_1": test_m["p_at_1"]})

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    log(f"\n[done]  neg_type={args.neg_type}  k={args.k}  "
        f"best_val_AUC={best_auc:.4f}  "
        f"test_AUC={test_m['auc']:.4f}  test_P@1={test_m['p_at_1']:.4f}  "
        f"stopped_early={stopped_early}")
    log_fh.close()


if __name__ == "__main__":
    main()
