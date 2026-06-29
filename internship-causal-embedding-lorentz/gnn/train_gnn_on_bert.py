"""GNN on top of fine-tuned BERT embeddings.

Uses an existing BiEncoder checkpoint (from finetune_eval) as the frozen
text encoder — no new fine-tuning of BERT.  Trains only the GNN layers and
asymmetric projection heads to add causal-graph context on top of already
task-adapted text representations.

This is the fairest test of whether a directional GNN adds value beyond
a fine-tuned two-tower model: we start from identical BERT representations
and only add the GNN structural signal.

Hard negative mining (same as finetune_eval/mine_hard_negatives.py):
  kNN of positives in positive-embedding space, filtered for anchor similarity.

Usage:
  PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  CKPT=/mnt/localssd/internship-causal-embedding-merge-followup/finetune_eval/results/aep_causal/checkpoint_best.pt
  CUDA_VISIBLE_DEVICES=0 $PYTHON gnn/train_gnn_on_bert.py \\
      --bert-checkpoint $CKPT --dataset aep_causal
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gnn.model import HybridCauseEffectGNN          # noqa: E402
from gnn.train_gnn import (                          # noqa: E402
    load_pairs_jsonl, build_doc_registry, build_graph,
    SPLITS_CFG, DATA_DIR, check_memory,
)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def guard_mem(label=""):
    import psutil
    avail = psutil.virtual_memory().available / 1024**3
    print(f"  [mem {label}] avail={avail:.1f}GB")
    if avail < 4.0:
        raise MemoryError(f"OOM risk: {avail:.1f}GB available")


# ---------------------------------------------------------------------------
# BiEncoder text encoding (frozen)
# ---------------------------------------------------------------------------

def load_biencoder_and_encode(
    ckpt_path: Path,
    texts_anchor: list[str],  # for anchor encoder
    texts_positive: list[str],  # for positive encoder
    all_doc_texts: list[str],  # for GNN nodes (use positive encoder)
    max_length: int = 256,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode texts with a frozen BiEncoder checkpoint.

    Returns:
        anchor_embs:   [N, D] — pair-specific anchor embeddings
        positive_embs: [N, D] — pair-specific positive embeddings
        doc_embs:      [M, D] — document embeddings for GNN nodes
    """
    from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts  # noqa: E402

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    print(f"  loaded BiEncoder: backbone={cfg['backbone']}  "
          f"pooling={cfg['pooling']}  val_mrr={ckpt.get('metrics',{}).get('mrr',0):.4f}")

    guard_mem("before biencoder load")
    tokenizer = get_tokenizer(cfg["backbone"])
    model = BiEncoder(
        model_name=cfg["backbone"],
        pooling_strategy=cfg["pooling"],
        anchor_prefix="",
        positive_prefix="",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def encode_batch(texts, which):
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            ids, mask = tokenize_texts(tokenizer, chunk, max_length)
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.no_grad():
                emb = model.encode_anchor(ids, mask) if which == "anchor" \
                      else model.encode_positive(ids, mask)
            out.append(emb.cpu().numpy())
        return np.concatenate(out, 0)

    print(f"  encoding {len(texts_anchor)} anchor pairs …")
    a_emb = encode_batch(texts_anchor, "anchor")
    guard_mem("after anchor encode")

    print(f"  encoding {len(texts_positive)} positive pairs …")
    p_emb = encode_batch(texts_positive, "positive")
    guard_mem("after positive encode")

    print(f"  encoding {len(all_doc_texts)} docs (positive encoder) for GNN …")
    d_emb = encode_batch(all_doc_texts, "positive")
    guard_mem("after doc encode")

    del model
    torch.cuda.empty_cache()
    return a_emb, p_emb, d_emb


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class BertGNNDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str, dict]],
        anchor_embs: torch.Tensor,    # [N, D]
        positive_embs: torch.Tensor,  # [N, D]
        hard_negatives: np.ndarray,   # [N, K]
        id2idx: dict[str, int],
    ):
        self.anchor_embs   = anchor_embs
        self.positive_embs = positive_embs
        self.hard_negatives = hard_negatives
        self.k_neg = hard_negatives.shape[1] if hard_negatives is not None else 0

        self.items: list[tuple[int, int, int]] = []
        self.pair_to_tgt: dict[int, int] = {}
        for i, (_, _, meta) in enumerate(pairs):
            sid, tid = meta["source_doc_id"], meta["target_doc_id"]
            if sid in id2idx and tid in id2idx:
                self.items.append((i, id2idx[sid], id2idx[tid]))
                self.pair_to_tgt[i] = id2idx[tid]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pair_i, src, tgt = self.items[idx]
        a = self.anchor_embs[pair_i]
        p = self.positive_embs[pair_i]
        if self.k_neg > 0:
            hn = self.hard_negatives[pair_i].tolist()
            neg_t = self.positive_embs[hn]
            neg_d = torch.tensor([self.pair_to_tgt.get(int(j), tgt) for j in hn], dtype=torch.long)
        else:
            neg_t = torch.zeros(0, a.shape[0])
            neg_d = torch.zeros(0, dtype=torch.long)
        return a, p, torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long), neg_t, neg_d


def collate_fn(batch):
    a  = torch.stack([b[0] for b in batch])
    p  = torch.stack([b[1] for b in batch])
    s  = torch.stack([b[2] for b in batch])
    t  = torch.stack([b[3] for b in batch])
    K  = batch[0][4].shape[0]
    if K > 0:
        nt = torch.stack([b[4] for b in batch])
        nd = torch.stack([b[5] for b in batch])
    else:
        D  = a.shape[1]
        nt = torch.zeros(len(batch), 0, D)
        nd = torch.zeros(len(batch), 0, dtype=torch.long)
    return a, p, s, t, nt, nd


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class HardNegInfoNCE(nn.Module):
    def __init__(self, tau_init=0.05, tau_min=0.02):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(tau_init).log())
        self.tau_min = tau_min

    @property
    def temperature(self): return self.log_tau.exp().clamp(self.tau_min, 0.5)

    def forward(self, c, e, neg=None):
        B = c.size(0); tau = self.temperature; dev = c.device
        sim = c @ e.T
        if neg is not None and neg.numel() > 0:
            hn = torch.einsum("bd,bkd->bk", c, neg)
            logits = torch.cat([sim, hn], 1) / tau
        else:
            logits = sim / tau
        loss = F.cross_entropy(logits, torch.arange(B, device=dev))
        with torch.no_grad():
            off = ~torch.eye(B, dtype=torch.bool, device=dev)
            st = {"tau": float(tau), "pos": float(sim.diag().mean()),
                  "neg": float(sim[off].mean()),
                  "hn":  float(hn.mean()) if (neg is not None and neg.numel() > 0) else 0.0}
        return loss, st


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_pairs, val_a, val_p, id2idx, x_text, edge_index, device, k=(1,3,5,10)):
    model.eval()
    guard_mem("validate")
    all_h = model.encode_graph(x_text.to(device), edge_index.to(device))
    cause_l, effect_l = [], []
    for i, (_, _, meta) in enumerate(val_pairs):
        sid, tid = meta["source_doc_id"], meta["target_doc_id"]
        if sid not in id2idx or tid not in id2idx: continue
        h_c = all_h[id2idx[sid]].unsqueeze(0)
        h_e = all_h[id2idx[tid]].unsqueeze(0)
        cause_l.append(model.fuse_cause(val_a[i:i+1].to(device), h_c))
        effect_l.append(model.fuse_effect(val_p[i:i+1].to(device), h_e))
    if not cause_l: return {"mrr": 0.0}
    A = torch.cat(cause_l);  P = torch.cat(effect_l)
    sim = A @ P.T
    ranks = (sim > sim.diag().unsqueeze(1)).sum(1).float() + 1
    out = {"mrr": float((1/ranks).mean())}
    for kk in k:
        out[f"recall@{kk}"] = float((ranks <= kk).float().mean())
    out["mean_rank"] = float(ranks.mean())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal", choices=list(SPLITS_CFG.keys()))
    ap.add_argument("--bert-checkpoint", required=True,
                    help="Path to BiEncoder checkpoint from finetune_eval.")
    ap.add_argument("--results-dir", default=str(ROOT / "gnn/results"))
    ap.add_argument("--gnn-dim", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=768,
                    help="Output dim. Default=768 to preserve full BERT representations.")
    ap.add_argument("--use-text-projection", action="store_true",
                    help="Project text (768→out_dim) before GNN addition. "
                         "OFF by default (out_dim=768 residual).")
    ap.add_argument("--num-gnn-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--k-neg", type=int, default=8)
    ap.add_argument("--anchor-threshold", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gnn-lr", type=float, default=3e-4)
    ap.add_argument("--temperature-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-encode", action="store_true")
    ap.add_argument("--force-mine", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) / f"{args.dataset}_bert_gnn"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[train_gnn_on_bert]  dataset={args.dataset}  device={device}")
    print(f"  bert_ckpt={Path(args.bert_checkpoint).name}")
    print("=" * 72)
    guard_mem("start")

    # ---- Load pairs ----
    cfg = SPLITS_CFG[args.dataset]
    data_dir = DATA_DIR / args.dataset
    train_pairs = load_pairs_jsonl(data_dir / cfg["train"])
    val_pairs   = load_pairs_jsonl(data_dir / cfg["val"])
    test_pairs  = load_pairs_jsonl(data_dir / cfg["test"])
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    # ---- Doc registry + graph ----
    id2idx, idx2id, doc_texts = build_doc_registry(train_pairs, val_pairs + test_pairs)
    N_docs = len(doc_texts)
    edge_index, _ = build_graph(train_pairs, id2idx)
    print(f"  unique docs: {N_docs}  edges: {edge_index.size(1)}")

    # ---- Encode with frozen BERT ----
    train_a_cache = results_dir / "bert_train_anchor_embs.npy"
    train_p_cache = results_dir / "bert_train_positive_embs.npy"
    doc_cache     = results_dir / "bert_doc_embs.npy"
    val_a_cache   = results_dir / "bert_val_anchor_embs.npy"
    val_p_cache   = results_dir / "bert_val_positive_embs.npy"

    need_encode = args.force_encode or not all(
        c.exists() for c in [train_a_cache, train_p_cache, doc_cache, val_a_cache, val_p_cache]
    )

    if need_encode:
        train_a_raw = [a for a, _, _ in train_pairs]
        train_p_raw = [p for _, p, _ in train_pairs]
        a_emb, p_emb, d_emb = load_biencoder_and_encode(
            Path(args.bert_checkpoint),
            train_a_raw, train_p_raw, doc_texts,
            device=device,
        )
        np.save(train_a_cache, a_emb)
        np.save(train_p_cache, p_emb)
        np.save(doc_cache, d_emb)

        val_a_raw = [a for a, _, _ in val_pairs]
        val_p_raw = [p for _, p, _ in val_pairs]
        # Re-load model just for val (it was del-ed above)
        va_emb, vp_emb, _ = load_biencoder_and_encode(
            Path(args.bert_checkpoint),
            val_a_raw, val_p_raw, doc_texts[:1],  # dummy doc
            device=device,
        )
        np.save(val_a_cache, va_emb)
        np.save(val_p_cache, vp_emb)
        print("  all embeddings cached")
    else:
        print("  loading cached BERT embeddings")

    guard_mem("load cached embs")
    a_emb  = torch.from_numpy(np.load(train_a_cache)).float()
    p_emb  = torch.from_numpy(np.load(train_p_cache)).float()
    d_emb  = torch.from_numpy(np.load(doc_cache)).float()
    va_emb = torch.from_numpy(np.load(val_a_cache)).float()
    vp_emb = torch.from_numpy(np.load(val_p_cache)).float()
    text_dim = a_emb.shape[1]
    print(f"  text_dim={text_dim}  train: a={a_emb.shape}  docs: {d_emb.shape}")
    guard_mem("after load embs")

    # ---- Hard negative mining (on BERT positive embeddings) ----
    mine_cache = results_dir / "gnn_hard_negatives.npy"
    if args.force_mine and mine_cache.exists():
        mine_cache.unlink()

    if mine_cache.exists():
        print("  loading cached hard negatives")
        hard_neg = np.load(mine_cache)
    else:
        from gnn.negatives import mine_hard_negatives
        # Use BERT anchor/positive embeddings for mining (higher quality)
        train_anchors   = [a for a, _, _ in train_pairs]
        train_positives = [p for _, p, _ in train_pairs]
        # For mining, provide already-computed embeddings
        # Override the encode_texts call with precomputed BERT embeddings
        hard_neg = _mine_with_precomputed(a_emb.numpy(), p_emb.numpy(),
                                          k=args.k_neg,
                                          anchor_sim_threshold=args.anchor_threshold)
        np.save(mine_cache, hard_neg)
    print(f"  hard_neg: {hard_neg.shape}")
    guard_mem("after mining")

    # ---- Dataset ----
    train_ds = BertGNNDataset(train_pairs, a_emb, p_emb, hard_neg, id2idx)
    K = train_ds.k_neg
    print(f"  train dataset: {len(train_ds)} items  K={K}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)

    # ---- Model ----
    model = HybridCauseEffectGNN(
        text_dim=text_dim, gnn_dim=args.gnn_dim, out_dim=args.out_dim,
        num_gnn_layers=args.num_gnn_layers, heads=args.heads, dropout=args.dropout,
        use_text_projection=args.use_text_projection,
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  GNN params: {n_p:,}")

    loss_fn = HardNegInfoNCE(tau_init=0.05, tau_min=0.02).to(device)

    text_params = (list(model.W_text_cause.parameters()) if model.W_text_cause is not None else []) + \
                  (list(model.W_text_effect.parameters()) if model.W_text_effect is not None else [])
    gnn_params  = (list(model.gnn_input_proj.parameters()) +
                   list(model.gnn_layers.parameters()) +
                   list(model.W_gnn_cause.parameters()) +
                   list(model.W_gnn_effect.parameters()))
    all_opt_params = []
    if text_params:
        all_opt_params.append({"params": text_params, "lr": args.lr, "weight_decay": 1e-4})
    all_opt_params.append({"params": gnn_params, "lr": args.gnn_lr, "weight_decay": 1e-4})
    all_opt_params.append({"params": [loss_fn.log_tau], "lr": args.temperature_lr, "weight_decay": 0.0})
    opt = AdamW(all_opt_params)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    # Staged training: freeze GNN during warmup
    def freeze_gnn(frozen):
        for p in gnn_params: p.requires_grad = not frozen
    if args.warmup_epochs > 0:
        freeze_gnn(True)
        print(f"  staged: GNN frozen for {args.warmup_epochs} warmup epochs")

    # ---- Config ----
    cfg_out = {
        "dataset": args.dataset, "bert_checkpoint": args.bert_checkpoint,
        "gnn_dim": args.gnn_dim, "out_dim": args.out_dim,
        "num_gnn_layers": args.num_gnn_layers, "heads": args.heads,
        "k_neg": args.k_neg, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size, "lr": args.lr, "gnn_lr": args.gnn_lr,
        "text_dim": text_dim, "n_docs": N_docs, "device": str(device),
    }
    (results_dir / "gnn_bert_config.json").write_text(json.dumps(cfg_out, indent=2))

    # ---- Training ----
    best_mrr = -1.0; history = []
    best_path = results_dir / "gnn_bert_best.pt"
    x_dev = d_emb.to(device)
    ei_dev = edge_index.to(device)

    print()
    print("=" * 72)
    print(f"Training  {args.epochs} epochs · batch={args.batch_size} · K={K}")
    print("=" * 72)

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            freeze_gnn(False)
            print(f"\n  [epoch {epoch+1}] GNN unfrozen")

        model.train()
        t0 = time.time(); rl = 0.0; nb = 0; last_st = {}
        for a, p, src, tgt, neg_t, neg_d in loader:
            a = a.to(device); p = p.to(device)
            src = src.to(device); tgt = tgt.to(device)
            opt.zero_grad()
            all_h = model.encode_graph(x_dev, ei_dev)
            cause  = model.fuse_cause(a, all_h[src])
            effect = model.fuse_effect(p, all_h[tgt])
            K_ = neg_t.shape[1]
            if K_ > 0:
                neg_t = neg_t.to(device); neg_d = neg_d.to(device)
                neg_emb = model.fuse_effect(neg_t, all_h[neg_d])
            else:
                neg_emb = None
            loss, st = loss_fn(cause, effect, neg_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); rl += loss.item(); nb += 1; last_st = st
        sched.step()

        vm = validate(model, val_pairs, va_emb, vp_emb, id2idx, x_dev, ei_dev, device)
        print(f"  ep {epoch+1:2d}/{args.epochs}  {int(time.time()-t0)}s  "
              f"loss={rl/max(nb,1):.4f}  τ={last_st.get('tau',0):.4f}  "
              f"pos={last_st.get('pos',0):.3f}  hn={last_st.get('hn',0):.3f}  "
              f"|| val MRR={vm['mrr']:.4f}  R@1={vm.get('recall@1',0):.3f}  "
              f"R@5={vm.get('recall@5',0):.3f}")
        if vm["mrr"] > best_mrr:
            best_mrr = vm["mrr"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "loss_fn_state": loss_fn.state_dict(),
                        "val_metrics": vm, "cfg": cfg_out,
                        "id2idx": id2idx, "idx2id": idx2id}, best_path)
            print(f"    → best MRR={best_mrr:.4f}  saved")
        history.append({"epoch": epoch+1, "val": vm, **last_st})
        if (epoch + 1) % 5 == 0: guard_mem(f"ep{epoch+1}")

    (results_dir / "gnn_bert_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done]  best val MRR={best_mrr:.4f}  → {best_path}")


def _mine_with_precomputed(
    anc_emb: np.ndarray,  # [N, D] BERT anchor embeddings
    pos_emb: np.ndarray,  # [N, D] BERT positive embeddings
    k: int,
    anchor_sim_threshold: float,
) -> np.ndarray:
    """Mine hard negatives using precomputed BERT embeddings."""
    import torch
    N = len(pos_emb)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_t = torch.from_numpy(pos_emb.astype("float32")).to(device)
    anc_t = torch.from_numpy(anc_emb.astype("float32")).to(device)
    hard_idx = np.empty((N, k), dtype=np.int32)
    chunk = 1024
    n_filtered = 0
    print(f"  mining hard negatives: N={N}  k={k}  threshold={anchor_sim_threshold}")
    for start in range(0, N, chunk):
        end = min(start + chunk, N); b = end - start
        ps = pos_t[start:end] @ pos_t.T                    # [b, N]
        as_ = anc_t[start:end] @ anc_t.T                   # [b, N]
        rows = torch.arange(b, device=device)
        ps[rows, torch.arange(start, end, device=device)] = float("-inf")  # mask self
        n_filtered += int((as_ > anchor_sim_threshold).sum().item()) - b
        ps[as_ > anchor_sim_threshold] = float("-inf")
        _, top = ps.topk(k, dim=1, largest=True, sorted=True)
        hard_idx[start:end] = top.cpu().numpy()
    print(f"  anchor-filtered: {n_filtered}")
    return hard_idx


if __name__ == "__main__":
    main()
