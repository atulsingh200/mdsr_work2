"""Cross-encoder (monoBERT) for cause→effect ranking.

ARCHITECTURE EXPLORATION — same base model, different architecture.

The baseline is a two-tower BiEncoder: anchor and candidate are encoded
SEPARATELY into vectors, then dotted. The towers never see each other's text.

This is a CROSS-ENCODER (Nogueira & Cho, 2019, "Passage Re-ranking with BERT"):
the anchor and candidate are concatenated into ONE sequence

        [CLS] anchor [SEP] candidate [SEP]

and run through a SINGLE bert-base-uncased. Every anchor token attends to every
candidate token (full cross-attention), and the [CLS] vector → linear → scalar
relevance score. This joint attention is strictly more expressive than two
independent towers, and is the standard way to get large ranking gains while
keeping the *same* backbone.

  * Base model: google-bert/bert-base-uncased   (IDENTICAL to the baseline)
  * Change: bi-encoder (2 towers, dot product) → cross-encoder (1 tower, joint)

Training: listwise softmax / InfoNCE over the cross-encoder scores. For each
anchor we score its positive + K mined hard negatives and apply cross-entropy
with the positive as the target (monoBERT-style localized contrastive loss).

Hard negatives: semantic kNN (MiniLM, same as finetune_eval) + anchor-similarity
filter (drop a negative whose anchor ≈ the query anchor — likely a false neg).

Memory guard: check_memory() before heavy steps; abort if <4 GB CPU RAM free.

Usage:
  PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY arch_improve/train_crossencoder.py --dataset aep_causal
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finetune_eval.data import load_pairs                      # noqa: E402
from finetune_eval.datasets import SPLITS, split_path          # noqa: E402


def check_memory(label: str = "") -> None:
    import psutil
    avail = psutil.virtual_memory().available / 1024 ** 3
    print(f"  [mem {label}] avail={avail:.1f}GB")
    if avail < 4.0:
        raise MemoryError(f"OOM risk: {avail:.1f}GB available")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrossEncoder(nn.Module):
    """bert-base-uncased + linear scoring head over the [CLS] token."""

    def __init__(self, model_name="google-bert/bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]          # [N, H]
        return self.score_head(self.dropout(cls)).squeeze(-1)  # [N]


# ---------------------------------------------------------------------------
# Dataset: each item = (anchor, [positive, hardneg_1..hardneg_K])
# ---------------------------------------------------------------------------

class CEDataset(Dataset):
    def __init__(self, pairs, hard_negatives, n_random=0, seed=0):
        self.anchors = [a for a, _ in pairs]
        self.positives = [p for _, p in pairs]
        self.hard_negatives = hard_negatives        # [N, K] int idx into positives
        self.n_random = n_random
        self.N = len(pairs)
        self.rng = random.Random(seed)

    @property
    def k_hard(self):
        return 0 if self.hard_negatives is None else int(self.hard_negatives.shape[1])

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        anchor = self.anchors[i]
        cand_texts = [self.positives[i]]            # index 0 = positive (target)
        if self.k_hard > 0:
            for j in self.hard_negatives[i]:
                cand_texts.append(self.positives[int(j)])
        for _ in range(self.n_random):              # optional extra random negs
            r = self.rng.randrange(self.N)
            while r == i:
                r = self.rng.randrange(self.N)
            cand_texts.append(self.positives[r])
        return anchor, cand_texts


def make_collate(tokenizer, max_length):
    def collate(batch):
        anchors, cands_per = zip(*batch)
        C = len(cands_per[0])                       # candidates per anchor (1 + K [+ R])
        a_rep, c_flat = [], []
        for a, cands in zip(anchors, cands_per):
            for c in cands:
                a_rep.append(a); c_flat.append(c)
        enc = tokenizer(a_rep, c_flat, truncation="longest_first",
                        max_length=max_length, padding=True, return_tensors="pt")
        return enc, len(batch), C
    return collate


# ---------------------------------------------------------------------------
# Validation: build score matrix over the val positives, rank the diagonal
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_matrix(model, tokenizer, anchors, candidates, max_length, device,
                 batch_pairs=256, amp_ctx=nullcontext):
    """Return [n_anchor, n_cand] cross-encoder score matrix (row=anchor)."""
    model.eval()
    nA, nC = len(anchors), len(candidates)
    S = np.zeros((nA, nC), dtype=np.float32)
    for i in range(nA):
        a = anchors[i]
        row = np.zeros(nC, dtype=np.float32)
        for s in range(0, nC, batch_pairs):
            chunk = candidates[s:s + batch_pairs]
            enc = tokenizer([a] * len(chunk), chunk, truncation="longest_first",
                            max_length=max_length, padding=True, return_tensors="pt").to(device)
            with amp_ctx():
                sc = model(**enc)
            row[s:s + len(chunk)] = sc.float().cpu().numpy()
        S[i] = row
    return S


def retrieval_from_scores(S, k_values=(1, 3, 5, 10)):
    """Diagonal-gold retrieval metrics from a score matrix (higher=better)."""
    nA = S.shape[0]
    diag = np.diag(S)[:, None]
    ranks = (S > diag).sum(axis=1) + 1               # 1-based rank of the true positive
    out = {"mrr": float((1.0 / ranks).mean()),
           "mean_rank": float(ranks.mean()),
           "median_rank": float(np.median(ranks))}
    for k in k_values:
        out[f"recall@{k}"] = float((ranks <= k).mean())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal")
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased",
                    help="MUST match the baseline base model.")
    ap.add_argument("--results-dir", default=str(ROOT / "arch_improve/results"))
    ap.add_argument("--max-seq-length", type=int, default=384)
    ap.add_argument("--k-neg", type=int, default=7)
    ap.add_argument("--n-random", type=int, default=0)
    ap.add_argument("--anchor-threshold", type=float, default=0.9)
    ap.add_argument("--mine-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16, help="Anchors per batch.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-max", type=int, default=562, help="Cap val anchors for the N×N matrix.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force-mine", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir) / f"{args.dataset}_crossencoder"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"[cross-encoder]  dataset={args.dataset}  backbone={args.backbone}")
    print(f"  device={device}  k_neg={args.k_neg}  max_len={args.max_seq_length}")
    print("=" * 74)
    check_memory("start")

    # ---- Data ----
    train_pairs = load_pairs(split_path(args.dataset, "train"), cap=None)
    val_path = split_path(args.dataset, "val")
    val_pairs = load_pairs(val_path, cap=None) if (val_path and val_path.exists()) else []
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}")
    if args.val_max and len(val_pairs) > args.val_max:
        val_pairs = val_pairs[:args.val_max]

    # ---- Hard negatives (MiniLM + anchor filter) ----
    mine_cache = results_dir / "hard_negatives.npy"
    if args.force_mine and mine_cache.exists():
        mine_cache.unlink()
    if mine_cache.exists():
        print("  loading cached hard negatives")
        hard_neg = np.load(mine_cache)
    else:
        from gnn.negatives import mine_hard_negatives
        hard_neg = mine_hard_negatives(
            [a for a, _ in train_pairs], [p for _, p in train_pairs],
            k=args.k_neg, anchor_sim_threshold=args.anchor_threshold,
            model_name=args.mine_encoder,
        )
        np.save(mine_cache, hard_neg)
    print(f"  hard_neg: {hard_neg.shape}")
    check_memory("mined")

    # ---- Model ----
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = CrossEncoder(args.backbone).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    check_memory("model loaded")

    train_ds = CEDataset(train_pairs, hard_neg, n_random=args.n_random, seed=args.seed)
    C = 1 + train_ds.k_hard + args.n_random
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=make_collate(tokenizer, args.max_seq_length), num_workers=2)
    print(f"  candidates/anchor C={C}  steps/epoch={len(loader)}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    sched = OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                       pct_start=args.warmup_ratio, anneal_strategy="cos")
    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
          if (args.bf16 and device.type == "cuda") else (lambda: nullcontext())

    cfg = {"dataset": args.dataset, "backbone": args.backbone, "arch": "cross_encoder",
           "max_seq_length": args.max_seq_length, "k_neg": args.k_neg, "n_random": args.n_random,
           "anchor_threshold": args.anchor_threshold, "epochs": args.epochs,
           "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed}
    (results_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    val_anchors = [a for a, _ in val_pairs]
    val_cands = [p for _, p in val_pairs]

    best_mrr = -1.0; history = []
    best_path = results_dir / "best.pt"
    print()
    print("=" * 74)
    print(f"Training {args.epochs} epochs · batch={args.batch_size} · C={C} candidates/anchor")
    print("=" * 74)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time(); rl = 0.0; nb = 0
        for enc, B, C_ in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            opt.zero_grad()
            with amp():
                scores = model(**enc)               # [B*C]
            scores = scores.float().view(B, C_)      # [B, C]
            labels = torch.zeros(B, dtype=torch.long, device=device)  # idx 0 = positive
            loss = F.cross_entropy(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            rl += loss.item(); nb += 1

        # ---- validation (N×N score matrix) ----
        vm = {"mrr": 0.0}
        if val_anchors:
            S = score_matrix(model, tokenizer, val_anchors, val_cands,
                             args.max_seq_length, device, amp_ctx=amp)
            vm = retrieval_from_scores(S)
        print(f"  ep {epoch+1}/{args.epochs}  {int(time.time()-t0)}s  loss={rl/max(nb,1):.4f}  "
              f"|| val MRR={vm['mrr']:.4f}  R@1={vm.get('recall@1',0):.3f}  "
              f"R@5={vm.get('recall@5',0):.3f}  R@10={vm.get('recall@10',0):.3f}")

        if vm["mrr"] > best_mrr:
            best_mrr = vm["mrr"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": vm, "cfg": cfg}, best_path)
            print(f"    → best val MRR={best_mrr:.4f}  saved")
        history.append({"epoch": epoch + 1, "train_loss": rl/max(nb,1), "val": vm})
        check_memory(f"ep{epoch+1}")

    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done]  best val MRR={best_mrr:.4f}  → {best_path}")


if __name__ == "__main__":
    main()
