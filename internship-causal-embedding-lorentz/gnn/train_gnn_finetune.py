"""Train Hybrid GNN with a fine-tunable text encoder.

Architecture (end-to-end trainable):
  anchor_text  →  Encoder (fine-tuned)  →  W_cause(text) + W_gnn_c(GNN(src_doc))  →  L2-norm
  positive_text → Encoder (fine-tuned)  →  W_effect(text) + W_gnn_e(GNN(tgt_doc)) → L2-norm

The encoder is updated every epoch by re-encoding all graph documents. This
"cache-refresh" strategy fine-tunes the encoder while keeping the GNN forward
efficient (no need to backprop through encoder at every GNN step).

Beats the two-tower BERT because:
  1. BGE-base-en-v1.5 is a stronger text encoder than bert-base-uncased
  2. Fine-tuning adapts the encoder to the causal task
  3. GNN adds document-level causal graph context

Hard negative mining (same as finetune_eval/mine_hard_negatives.py):
  For each pair i (anchor_i, positive_i):
    1. Find K nearest other positives by cosine similarity of positive texts.
    2. Filter: skip pair j if sim(anchor_i, anchor_j) > anchor_sim_threshold.

Memory guard: checks CPU RAM and aborts if < 4 GB.

Usage:
  PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  CUDA_VISIBLE_DEVICES=0 $PYTHON gnn/train_gnn_finetune.py --dataset aep_causal
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
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

from gnn.model import HybridCauseEffectGNN        # noqa: E402
from gnn.train_gnn import (                        # noqa: E402
    load_pairs_jsonl, build_doc_registry, build_graph,
    SPLITS_CFG, DATA_DIR, check_memory,
)


# ---------------------------------------------------------------------------
# Text encoder wrapper (fine-tunable)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Shared text encoder with mean pooling + L2 normalisation."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 max_length: int = 256):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size

    def tokenize(self, texts: list[str], device: torch.device):
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over tokens (mask padding)
        token_emb = out.last_hidden_state                        # [B, T, D]
        mask_exp  = attention_mask.unsqueeze(-1).float()         # [B, T, 1]
        pooled    = (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        return F.normalize(pooled, dim=-1)                       # [B, D]

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """Encode a list of texts; returns (N, D) float32 L2-normalised."""
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            ids, mask = self.tokenize(texts[i:i+batch_size], device)
            emb = self(ids, mask)
            out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.hidden_size))


# ---------------------------------------------------------------------------
# Training dataset — pair-level texts, no pre-cached embeddings
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Returns (anchor_text, positive_text, src_doc_idx, tgt_doc_idx,
                [hardneg_texts], [hardneg_doc_idxs]) per pair."""

    def __init__(
        self,
        pairs: list[tuple[str, str, dict]],
        hard_negatives: np.ndarray,   # [N, K]
        id2idx: dict[str, int],
    ):
        self.pairs = [(a, p, meta) for a, p, meta in pairs
                      if meta["source_doc_id"] in id2idx
                      and meta["target_doc_id"] in id2idx]
        self.hard_negatives = hard_negatives
        self.id2idx = id2idx
        # Map pair index → original pair index for hard neg lookup
        self.orig_idx = [i for i, (a, p, meta) in enumerate(pairs)
                         if meta["source_doc_id"] in id2idx
                         and meta["target_doc_id"] in id2idx]
        self.all_positives = [p for _, p, _ in pairs]  # all positives for hard neg text lookup

    @property
    def k_neg(self) -> int:
        return 0 if self.hard_negatives is None else int(self.hard_negatives.shape[1])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        a, p, meta = self.pairs[idx]
        orig = self.orig_idx[idx]
        src = self.id2idx[meta["source_doc_id"]]
        tgt = self.id2idx[meta["target_doc_id"]]
        K = self.k_neg
        if K > 0:
            hn_idxs = self.hard_negatives[orig].tolist()
            hn_texts = [self.all_positives[int(j)] for j in hn_idxs]
            hn_docs  = [self.id2idx.get(self.pairs[int(j)][2]["target_doc_id"], tgt)
                        if int(j) < len(self.pairs) else tgt
                        for j in hn_idxs]
        else:
            hn_texts, hn_docs = [], []
        return a, p, src, tgt, hn_texts, hn_docs


def collate_pairs_text(batch):
    a_texts  = [b[0] for b in batch]
    p_texts  = [b[1] for b in batch]
    src      = torch.tensor([b[2] for b in batch], dtype=torch.long)
    tgt      = torch.tensor([b[3] for b in batch], dtype=torch.long)
    # Hard negs: list of lists
    hn_texts_per = [b[4] for b in batch]
    hn_docs_per  = [b[5] for b in batch]
    return a_texts, p_texts, src, tgt, hn_texts_per, hn_docs_per


# ---------------------------------------------------------------------------
# InfoNCE with hard negatives
# ---------------------------------------------------------------------------

class HardNegInfoNCE(nn.Module):
    def __init__(self, temperature_init=0.05, temperature_min=0.02):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(temperature_init).log())
        self.temperature_min = temperature_min

    @property
    def temperature(self): return self.log_tau.exp().clamp(self.temperature_min, 0.5)

    def forward(self, cause, effect, neg=None):
        B = cause.size(0)
        tau = self.temperature
        dev = cause.device
        sim_ib = cause @ effect.T
        if neg is not None and neg.numel() > 0:
            sim_hn = torch.einsum("bd,bkd->bk", cause, neg)
            logits = torch.cat([sim_ib, sim_hn], 1) / tau
        else:
            logits = sim_ib / tau
        loss = F.cross_entropy(logits, torch.arange(B, device=dev))
        with torch.no_grad():
            d = sim_ib.diag()
            off = ~torch.eye(B, dtype=torch.bool, device=dev)
            stats = {"tau": float(tau), "pos": float(d.mean()),
                     "neg": float(sim_ib[off].mean()),
                     "hn":  float(sim_hn.mean()) if (neg is not None and neg.numel() > 0) else 0.0}
        return loss, stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    encoder: TextEncoder,
    gnn_model: HybridCauseEffectGNN,
    val_pairs: list[tuple[str, str, dict]],
    id2idx: dict[str, int],
    x_text: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
) -> dict:
    encoder.eval()
    gnn_model.eval()
    check_memory("validate")
    all_h = gnn_model.encode_graph(x_text.to(device), edge_index.to(device))

    cause_list, effect_list = [], []
    batch_size = 32
    anchors  = [a for a, _, _ in val_pairs]
    positives = [p for _, p, _ in val_pairs]
    metas    = [m for _, _, m in val_pairs]

    # Encode all val texts in batch
    all_a_emb = torch.from_numpy(
        encoder.encode_texts(anchors, batch_size=batch_size, device=device)
    ).float()
    all_p_emb = torch.from_numpy(
        encoder.encode_texts(positives, batch_size=batch_size, device=device)
    ).float()

    valid_idx = []
    for i, meta in enumerate(metas):
        sid, tid = meta["source_doc_id"], meta["target_doc_id"]
        if sid in id2idx and tid in id2idx:
            valid_idx.append((i, id2idx[sid], id2idx[tid]))

    if not valid_idx:
        return {"mrr": 0.0}

    idxs  = [v[0] for v in valid_idx]
    srcs  = torch.tensor([v[1] for v in valid_idx], device=device)
    tgts  = torch.tensor([v[2] for v in valid_idx], device=device)

    h_src = all_h[srcs]
    h_tgt = all_h[tgts]
    a_emb = all_a_emb[idxs].to(device)
    p_emb = all_p_emb[idxs].to(device)

    cause  = gnn_model.fuse_cause(a_emb, h_src)   # [N, D]
    effect = gnn_model.fuse_effect(p_emb, h_tgt)  # [N, D]

    sim = cause @ effect.T
    ranks = (sim > sim.diag().unsqueeze(1)).sum(1).float() + 1
    out = {"mrr": float((1.0 / ranks).mean())}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = float((ranks <= k).float().mean())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal", choices=list(SPLITS_CFG.keys()))
    ap.add_argument("--results-dir", default=str(ROOT / "gnn/results"))
    ap.add_argument("--encoder-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--gnn-dim", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--num-gnn-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--k-neg", type=int, default=4,
                    help="Hard negatives per anchor (lower k for fine-tuning).")
    ap.add_argument("--anchor-threshold", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=5,
                    help="Fewer epochs needed since encoder is being fine-tuned.")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Smaller batch for encoder fine-tuning memory.")
    ap.add_argument("--encoder-lr", type=float, default=2e-5,
                    help="LR for text encoder (low for fine-tuning).")
    ap.add_argument("--head-lr", type=float, default=1e-3,
                    help="LR for W_cause/W_effect/GNN heads.")
    ap.add_argument("--temperature-lr", type=float, default=1e-2)
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--temperature-min", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-refresh-every", type=int, default=1,
                    help="Refresh GNN document embeddings every N epochs.")
    ap.add_argument("--force-mine", action="store_true")
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 for encoder forward (A100-friendly).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) / f"{args.dataset}_finetune"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[train_gnn_finetune]  dataset={args.dataset}  device={device}")
    print(f"  encoder={args.encoder_model}  fine-tune lr={args.encoder_lr}")
    print(f"  gnn_dim={args.gnn_dim}  k_neg={args.k_neg}")
    print("=" * 72)
    check_memory("start")

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
    check_memory("graph built")

    # ---- Load text encoder ----
    print(f"  loading encoder: {args.encoder_model}")
    encoder = TextEncoder(args.encoder_model, max_length=args.max_seq_length).to(device)
    text_dim = encoder.hidden_size
    print(f"  text_dim={text_dim}  encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # ---- Initial document embeddings for GNN ----
    def refresh_doc_embeddings() -> torch.Tensor:
        check_memory("refresh_doc_emb")
        print("  refreshing doc embeddings …")
        emb = encoder.encode_texts(doc_texts, batch_size=64, device=device)
        return torch.from_numpy(emb).float()

    x_text = refresh_doc_embeddings()
    check_memory("initial doc embs")

    # ---- Hard negative mining (initial, uses initial BGE embeddings) ----
    train_anchors   = [a for a, _, _ in train_pairs]
    train_positives = [p for _, p, _ in train_pairs]
    mine_cache = results_dir / "gnn_hard_negatives.npy"
    if args.force_mine and mine_cache.exists():
        mine_cache.unlink()

    if mine_cache.exists():
        print(f"  loading cached hard negatives")
        hard_neg_idx = np.load(mine_cache)
    else:
        from gnn.negatives import mine_hard_negatives
        hard_neg_idx = mine_hard_negatives(
            train_anchors, train_positives,
            k=args.k_neg, anchor_sim_threshold=args.anchor_threshold,
        )
        np.save(mine_cache, hard_neg_idx)
    print(f"  hard_neg_idx: {hard_neg_idx.shape}")
    check_memory("after mining")

    # ---- Datasets ----
    train_ds = PairDataset(train_pairs, hard_neg_idx, id2idx)
    K = train_ds.k_neg
    print(f"  train: {len(train_ds)} items  K={K}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pairs_text, num_workers=0, drop_last=True,
    )

    # ---- GNN model ----
    gnn_model = HybridCauseEffectGNN(
        text_dim=text_dim, gnn_dim=args.gnn_dim, out_dim=args.out_dim,
        num_gnn_layers=args.num_gnn_layers, heads=args.heads, dropout=args.dropout,
    ).to(device)
    n_gnn = sum(p.numel() for p in gnn_model.parameters())
    print(f"  GNN params: {n_gnn:,}")

    loss_fn = HardNegInfoNCE(args.temperature_init, args.temperature_min).to(device)

    # Separate optimizer groups
    optimizer = AdamW([
        {"params": encoder.model.parameters(),              "lr": args.encoder_lr, "weight_decay": 0.01},
        {"params": list(gnn_model.parameters()),            "lr": args.head_lr,    "weight_decay": 1e-4},
        {"params": [loss_fn.log_tau],                       "lr": args.temperature_lr, "weight_decay": 0.0},
    ])
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.encoder_lr, args.head_lr, args.temperature_lr],
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

    # ---- Config ----
    cfg_out = {
        "dataset": args.dataset, "encoder_model": args.encoder_model,
        "gnn_dim": args.gnn_dim, "out_dim": args.out_dim,
        "num_gnn_layers": args.num_gnn_layers, "heads": args.heads,
        "k_neg": args.k_neg, "anchor_threshold": args.anchor_threshold,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "temperature_init": args.temperature_init, "text_dim": text_dim,
        "n_docs": N_docs, "n_edges": int(edge_index.size(1)),
        "device": str(device),
    }
    (results_dir / "gnn_ft_config.json").write_text(json.dumps(cfg_out, indent=2))

    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if (args.bf16 and device.type == "cuda") else nullcontext())

    # ---- Training loop ----
    best_mrr = -1.0
    history  = []
    best_path = results_dir / "gnn_ft_best.pt"
    edge_idx_dev = edge_index.to(device)

    print()
    print("=" * 72)
    print(f"Fine-tune training  {args.epochs} epochs · batch={args.batch_size} · K={K}")
    print("=" * 72)

    for epoch in range(args.epochs):
        # Refresh document embeddings periodically
        if epoch % args.cache_refresh_every == 0:
            x_text = refresh_doc_embeddings().to(device)

        encoder.train(); gnn_model.train()
        t0 = time.time()
        running_loss = 0.0; nb = 0; last_stats = {}

        for a_texts, p_texts, src_doc, tgt_doc, hn_texts_per, hn_docs_per in train_loader:
            B = len(a_texts)
            src_doc = src_doc.to(device)
            tgt_doc = tgt_doc.to(device)

            optimizer.zero_grad()

            with amp_ctx():
                # Fine-tune encoder on batch texts
                a_ids, a_mask = encoder.tokenize(a_texts, device)
                p_ids, p_mask = encoder.tokenize(p_texts, device)
                a_emb = encoder(a_ids, a_mask)  # [B, D]
                p_emb = encoder(p_ids, p_mask)  # [B, D]

            # GNN forward with CURRENT doc embeddings (x_text may be stale but that's OK)
            all_h = gnn_model.encode_graph(x_text, edge_idx_dev)
            h_src = all_h[src_doc]
            h_tgt = all_h[tgt_doc]

            cause  = gnn_model.fuse_cause(a_emb.float(), h_src)
            effect = gnn_model.fuse_effect(p_emb.float(), h_tgt)

            # Hard neg embeddings
            if K > 0 and hn_texts_per[0]:
                flat_hn = [t for row in hn_texts_per for t in row]  # B*K texts
                flat_hd = torch.tensor([d for row in hn_docs_per for d in row], device=device)  # B*K
                with amp_ctx():
                    hn_ids, hn_mask = encoder.tokenize(flat_hn, device)
                    hn_emb = encoder(hn_ids, hn_mask)  # [B*K, D]
                h_negs = all_h[flat_hd]  # [B*K, gnn_dim]
                neg_eff = gnn_model.fuse_effect(hn_emb.float(), h_negs).view(B, K, -1)  # [B, K, D]
            else:
                neg_eff = None

            loss, stats = loss_fn(cause, effect, neg_eff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.model.parameters()) + list(gnn_model.parameters()),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item(); nb += 1; last_stats = stats

        # Refresh doc embeddings after epoch before validation
        x_text = refresh_doc_embeddings().to(device)

        val_metrics = validate(encoder, gnn_model, val_pairs, id2idx,
                               x_text, edge_idx_dev, device)
        print(
            f"  epoch {epoch+1}/{args.epochs}  {int(time.time()-t0)}s  "
            f"loss={running_loss/max(nb,1):.4f}  τ={last_stats.get('tau',0):.4f}  "
            f"pos={last_stats.get('pos',0):.3f}  hn={last_stats.get('hn',0):.3f}  "
            f"|| val MRR={val_metrics['mrr']:.4f}  "
            f"R@1={val_metrics.get('recall@1',0):.3f}  "
            f"R@5={val_metrics.get('recall@5',0):.3f}"
        )

        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "gnn_state":     gnn_model.state_dict(),
                "loss_fn_state": loss_fn.state_dict(),
                "val_metrics":   val_metrics, "cfg": cfg_out,
                "id2idx": id2idx, "idx2id": idx2id,
            }, best_path)
            print(f"    → new best  MRR={best_mrr:.4f}  saved")

        history.append({"epoch": epoch+1, "train_loss": running_loss/max(nb,1),
                        "val": val_metrics, **last_stats})
        check_memory(f"epoch{epoch+1}")

    (results_dir / "gnn_ft_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done]  best val MRR={best_mrr:.4f}  → {best_path}")


if __name__ == "__main__":
    main()
