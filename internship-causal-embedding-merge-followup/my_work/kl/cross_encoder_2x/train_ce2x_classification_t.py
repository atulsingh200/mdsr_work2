"""Train CrossEncoder2x as a binary classifier on a swapped-pair dataset.

Supports two datasets via --data-dir:
  - aep_dataset_t            (default) — derived from aep_dataset
  - aep_causal_classification_t        — derived from aep_causal_classification

Both datasets have every label=0 row with text_1 and text_2 swapped, testing
whether the model is sensitive to pair order and can still classify direction
correctly.

This script works directly on the {text_1, text_2, label} format — no
conversion needed.

Hard-negative mode: for each positive (label=1) row, mines K semantically
similar text_2 values from OTHER rows and creates K synthetic label=0 rows
that are harder to discriminate than the original random negatives.

Random-negative mode: uses the dataset's existing label=0 rows as-is (no
extra mining) — the dataset is already balanced with ~50% negatives.

Usage examples
--------------
  # aep_dataset_t — random negatives (default)
  python train_ce2x_classification_t.py --neg-type random

  # aep_dataset_t — hard negatives
  python train_ce2x_classification_t.py --neg-type hard

  # aep_causal_classification_t — random negatives
  python train_ce2x_classification_t.py --neg-type random \\
      --data-dir /mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_t \\
      --out-dir results/aep_causal_classification_t/random

  # aep_causal_classification_t — hard negatives
  python train_ce2x_classification_t.py --neg-type hard \\
      --data-dir /mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_t \\
      --out-dir results/aep_causal_classification_t/hard
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
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_ce2x import CrossEncoder2x, get_tokenizer

DATA_ROOT = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_dataset_t")

# ── 1. datasets ────────────────────────────────────────────────────────────────

class TextPairDataset(Dataset):
    """Direct {text_1, text_2, label} dataset — used in random-neg mode."""

    def __init__(self, path: Path):
        self.examples: list[tuple[str, str, int]] = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append((obj["text_1"], obj["text_2"], int(obj["label"])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class HardNegDataset(Dataset):
    """Positives + mined hard negatives dataset.

    For each label=1 row (text_1, text_2), appends K hard negatives by
    replacing text_2 with text_2 of the K most similar OTHER positives
    (similarity measured on text_2 by a frozen sentence encoder).
    """

    def __init__(
        self,
        pos_examples: list[tuple[str, str]],   # all (text_1, text_2) with label=1
        hard_idx: np.ndarray,                   # (N_pos, K)
    ):
        self.rows: list[tuple[str, str, int]] = []
        pos_texts2 = [p2 for _, p2 in pos_examples]
        for i, (t1, t2) in enumerate(pos_examples):
            self.rows.append((t1, t2, 1))
            for j in hard_idx[i]:
                self.rows.append((t1, pos_texts2[int(j)], 0))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate(tokenizer, max_length: int):
    def collate(batch: list[tuple[str, str, int]]):
        texts1 = [b[0] for b in batch]
        texts2 = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
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


# ── 2. hard negative mining ────────────────────────────────────────────────────

def mine_hard_negatives_cls(
    pos_examples: list[tuple[str, str]],
    k: int,
    args,
) -> np.ndarray:
    """Encode text_2 of positives with a frozen encoder; return top-K indices."""
    import torch as _torch

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

    texts2 = [p2 for _, p2 in pos_examples]
    N = len(texts2)
    print(f"[mine] encoding {N} text_2 values …")
    t0 = time.time()
    emb = miner.encode_candidates(texts2)
    print(f"[mine] encoded in {time.time()-t0:.1f}s  dim={emb.shape[1]}")

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    emb_t  = _torch.from_numpy(emb).to(device)
    chunk  = args.miner_chunk_size
    hard_idx = np.empty((N, k), dtype=np.int32)

    t1 = time.time()
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        sims = emb_t[start:end] @ emb_t.T
        rows = _torch.arange(start, end, device=device)
        sims[_torch.arange(end - start, device=device), rows] = float("-inf")
        _, top_idx = sims.topk(k, dim=1, largest=True, sorted=True)
        hard_idx[start:end] = top_idx.cpu().numpy()
    print(f"[mine] k-NN done in {time.time()-t1:.1f}s")
    return hard_idx


# ── 3. validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for ids, mask, tti, labels in loader:
        ids  = ids.to(device, non_blocking=True)
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
    return {
        "auc": float(roc_auc_score(labels_arr, logits_arr)),
        "acc": float(accuracy_score(labels_arr, preds)),
    }


# ── 4. main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # negative strategy
    ap.add_argument("--neg-type", choices=["hard", "random"], default="random",
                    help="random: use existing label=0 rows as-is. "
                         "hard: mine K hard negatives from label=1 positives only.")
    ap.add_argument("--k", type=int, default=4,
                    help="Hard negatives per positive (only used when --neg-type hard).")
    # data
    ap.add_argument("--data-dir", default=str(DATA_ROOT))
    ap.add_argument("--out-dir", default=None)
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
        # derive dataset name from the last component of data_dir for the output path
        ds_name = Path(args.data_dir).name
        args.out_dir = str(HERE / "results" / ds_name / args.neg_type)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    tokenizer = get_tokenizer(args.backbone)
    collate   = make_collate(tokenizer, args.max_length)

    if args.neg_type == "random":
        # Use the existing balanced label=0/1 dataset directly
        train_ds = TextPairDataset(data_dir / "directional_train.jsonl")
        val_ds   = TextPairDataset(data_dir / "directional_val.jsonl")
        test_ds  = TextPairDataset(data_dir / "directional_test.jsonl")
        print(f"[data] random mode — train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    else:
        # Hard-neg mode: collect only positives, mine K negatives per positive
        def load_positives(path: Path) -> list[tuple[str, str]]:
            pos = []
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if int(r["label"]) == 1:
                        pos.append((r["text_1"], r["text_2"]))
            return pos

        print("[data] hard mode — collecting positives …")
        train_pos = load_positives(data_dir / "directional_train.jsonl")
        val_pos   = load_positives(data_dir / "directional_val.jsonl")
        test_pos  = load_positives(data_dir / "directional_test.jsonl")
        print(f"[data] positives — train={len(train_pos)}  val={len(val_pos)}  test={len(test_pos)}")

        print("[neg] mining hard negatives for train …")
        train_hn = mine_hard_negatives_cls(train_pos, args.k, args)
        print("[neg] mining hard negatives for val …")
        val_hn   = mine_hard_negatives_cls(val_pos,   args.k, args)
        print("[neg] mining hard negatives for test …")
        test_hn  = mine_hard_negatives_cls(test_pos,  args.k, args)

        np.save(out_dir / "train_hard_negatives.npy", train_hn)
        np.save(out_dir / "val_hard_negatives.npy",   val_hn)
        np.save(out_dir / "test_hard_negatives.npy",  test_hn)

        train_ds = HardNegDataset(train_pos, train_hn)
        val_ds   = HardNegDataset(val_pos,   val_hn)
        test_ds  = HardNegDataset(test_pos,  test_hn)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CrossEncoder2x(backbone=args.backbone, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}  layers={args.n_layers}  params={n_params:,}  device={device}")

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

    log_fh = open(out_dir / "train.log", "w")

    def log(msg: str):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[config] dataset={Path(args.data_dir).name}  neg_type={args.neg_type}  k={args.k}  "
        f"epochs={args.epochs}  batch={args.batch_size}  grad_accum={args.grad_accum}  "
        f"eff_batch={args.batch_size * args.grad_accum}  max_len={args.max_length}  "
        f"total_steps={total_steps}  warmup={warmup_steps}  params={n_params:,}")

    best_auc   = -1.0
    bad_epochs = 0
    history    = []
    opt_step   = 0
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
                loss   = loss_fn(logits, labels) / args.grad_accum

            loss.backward()
            epoch_loss += float(loss.item()) * args.grad_accum

            if (micro_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    log(f"  step={opt_step}/{total_steps}  "
                        f"loss={loss.item() * args.grad_accum:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  epoch={epoch}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_m = evaluate(model, val_loader, device)
        log(f"[epoch {epoch}/{args.epochs}]  train_loss={avg_loss:.4f}  "
            f"val_AUC={val_m['auc']:.4f}  val_Acc={val_m['acc']:.4f}  "
            f"({time.time()-t0:.0f}s)")
        history.append({"epoch": epoch, "train_loss": avg_loss,
                        **{f"val_{k}": v for k, v in val_m.items()}})

        ckpt = {"model_state_dict": model.state_dict(), "cfg": vars(args),
                "epoch": epoch, **{f"val_{k}": v for k, v in val_m.items()}}
        torch.save(ckpt, out_dir / "checkpoint_latest.pt")

        if val_m["auc"] > best_auc + 1e-4:
            best_auc   = val_m["auc"]
            bad_epochs = 0
            torch.save(ckpt, out_dir / "checkpoint_best.pt")
            log(f"  * new best val_AUC={best_auc:.4f}")
        else:
            bad_epochs += 1
            log(f"  no improvement (best={best_auc:.4f}, patience {bad_epochs}/{args.early_stop_patience})")
            if bad_epochs >= args.early_stop_patience:
                log(f"[early stop] triggered at epoch {epoch}")
                break

    # ── test evaluation ──
    log("\n[test] loading checkpoint_best.pt …")
    best_ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, device)
    log(f"[test]  AUC={test_m['auc']:.4f}  Acc={test_m['acc']:.4f}")
    history.append({"test_auc": test_m["auc"], "test_acc": test_m["acc"]})

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log(f"\n[done]  neg_type={args.neg_type}  best_val_AUC={best_auc:.4f}  "
        f"test_AUC={test_m['auc']:.4f}  test_Acc={test_m['acc']:.4f}")
    log_fh.close()


if __name__ == "__main__":
    main()
