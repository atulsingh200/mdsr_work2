"""Train the Hybrid Cause→Effect GNN on the aep_causal dataset.

Architecture:
  For each (anchor_text, positive_text) training pair:
    cause_emb  = W_cause(concat(text_proj(anchor_text), GNN(source_doc)))
    effect_emb = W_effect(concat(text_proj(positive_text), GNN(target_doc)))

  This combines:
    - pair-specific text signal (same semantic quality as fine-tuned BERT)
    - document-level graph context (directional causal structure)

  Both outputs are L2-normalised → cosine-space InfoNCE loss.

Hard negative mining (mirrors finetune_eval/mine_hard_negatives.py):
  For each pair i (anchor_i, positive_i):
    1. Encode all positives with a frozen sentence encoder.
    2. Find K nearest other positives by cosine similarity.
    3. Filter: skip pair j if sim(anchor_i, anchor_j) > anchor_sim_threshold.
       (Too-similar anchors → positive_j may be a true effect of anchor_i.)

Memory guard:
  Checks CPU RAM before heavy steps. Aborts if available < 4 GB.

Usage:
  PYTHON=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  CUDA_VISIBLE_DEVICES=1 \\
  $PYTHON gnn/train_gnn.py --dataset aep_causal
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

from gnn.model import HybridCauseEffectGNN    # noqa: E402
from gnn.negatives import load_and_mine       # noqa: E402


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def check_memory(label: str = "") -> None:
    """Print CPU memory; abort if available < 4 GB."""
    import psutil
    vm = psutil.virtual_memory()
    avail_gb = vm.available / 1024 ** 3
    total_gb = vm.total / 1024 ** 3
    tag = f" [{label}]" if label else ""
    print(f"  [mem{tag}] avail={avail_gb:.1f}GB / {total_gb:.0f}GB total")
    if avail_gb < 4.0:
        raise MemoryError(
            f"CPU RAM critically low ({avail_gb:.1f}GB available). Aborting to prevent OOM."
        )


# ---------------------------------------------------------------------------
# Data layout config
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "data_6"
SPLITS_CFG = {
    "aep_causal":   {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "followupqg":   {"train": "train.jsonl", "val": "valid.jsonl", "test": "test.jsonl"},
    "multiwoz_v24": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
}


def load_pairs_jsonl(path: Path) -> list[tuple[str, str, dict]]:
    """Load (anchor, positive, meta) from JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = (row.get("anchor") or "").strip()
            p = (row.get("positive") or "").strip()
            if a and p:
                meta = {
                    "source_doc_id": row.get("source_doc_id", ""),
                    "target_doc_id": row.get("target_doc_id", ""),
                    "causal_type":   row.get("causal_type", "default"),
                }
                pairs.append((a, p, meta))
    return pairs


def build_doc_registry(
    train_pairs: list[tuple[str, str, dict]],
    extra_pairs: list[tuple[str, str, dict]] | None = None,
) -> tuple[dict[str, int], dict[int, str], list[str]]:
    """Map doc_id → int index; canonical text = first anchor or first positive."""
    texts_by_id: dict[str, str] = {}
    for a, p, meta in (train_pairs + (extra_pairs or [])):
        sid, tid = meta["source_doc_id"], meta["target_doc_id"]
        if sid and sid not in texts_by_id:
            texts_by_id[sid] = a
        if tid and tid not in texts_by_id:
            texts_by_id[tid] = p

    doc_ids = sorted(texts_by_id.keys())
    id2idx = {d: i for i, d in enumerate(doc_ids)}
    idx2id = {i: d for d, i in id2idx.items()}
    doc_texts = [texts_by_id[d] for d in doc_ids]
    return id2idx, idx2id, doc_texts


def build_graph(
    train_pairs: list[tuple[str, str, dict]],
    id2idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge_index (training pairs only) with transition-weight normalisation."""
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for _, _, meta in train_pairs:
        sid, tid = meta["source_doc_id"], meta["target_doc_id"]
        if sid in id2idx and tid in id2idx:
            edge_counts[(id2idx[sid], id2idx[tid])] += 1

    out_sum: dict[int, int] = defaultdict(int)
    for (s, _), cnt in edge_counts.items():
        out_sum[s] += cnt

    edges, weights = [], []
    for (s, t), cnt in edge_counts.items():
        edges.append([s, t])
        weights.append(cnt / out_sum[s])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(weights, dtype=torch.float),
    )


# ---------------------------------------------------------------------------
# Document embedding (cached) — used for GNN graph nodes
# ---------------------------------------------------------------------------

def encode_documents(
    doc_texts: list[str],
    cache_path: Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    force: bool = False,
) -> torch.Tensor:
    """Encode doc_texts with a frozen sentence encoder; cache as .npy."""
    if cache_path.exists() and not force:
        print(f"  loading cached doc embeddings: {cache_path}")
        check_memory("load_doc_emb")
        return torch.from_numpy(np.load(cache_path)).float()

    print(f"  encoding {len(doc_texts)} documents with {model_name} …")
    check_memory("pre_encode_docs")
    from evaluation.retrievers import PretrainedRetriever
    retriever = PretrainedRetriever(model_name=model_name, pooling="mean",
                                   max_seq_length=256, batch_size=batch_size)
    t0 = time.time()
    emb = retriever.encode_candidates(doc_texts)
    print(f"  encoded in {time.time()-t0:.1f}s  dim={emb.shape[1]}")
    check_memory("post_encode_docs")
    np.save(cache_path, emb)
    print(f"  saved → {cache_path}")
    return torch.from_numpy(emb).float()


# ---------------------------------------------------------------------------
# Pair-level text embedding (cached) — pair-specific, different from doc emb
# ---------------------------------------------------------------------------

def encode_pair_texts(
    anchors: list[str],
    positives: list[str],
    cache_path_a: Path,
    cache_path_p: Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode pair-specific texts; returns (anchor_embs, positive_embs)."""
    if cache_path_a.exists() and cache_path_p.exists() and not force:
        print(f"  loading cached pair embeddings")
        check_memory("load_pair_emb")
        return (torch.from_numpy(np.load(cache_path_a)).float(),
                torch.from_numpy(np.load(cache_path_p)).float())

    print(f"  encoding {len(anchors)} anchor+positive pairs with {model_name} …")
    check_memory("pre_encode_pairs")
    from evaluation.retrievers import PretrainedRetriever
    retriever = PretrainedRetriever(model_name=model_name, pooling="mean",
                                   max_seq_length=256, batch_size=batch_size)
    t0 = time.time()
    a_emb = retriever.encode_candidates(anchors)
    p_emb = retriever.encode_candidates(positives)
    print(f"  encoded in {time.time()-t0:.1f}s")
    check_memory("post_encode_pairs")
    np.save(cache_path_a, a_emb)
    np.save(cache_path_p, p_emb)
    return torch.from_numpy(a_emb).float(), torch.from_numpy(p_emb).float()


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class HybridPairDataset(Dataset):
    """(anchor_text_emb, positive_text_emb, cause_doc_idx, effect_doc_idx,
        [hard_neg_text_embs], [hard_neg_doc_idxs]) per pair."""

    def __init__(
        self,
        pairs: list[tuple[str, str, dict]],
        anchor_embs: torch.Tensor,    # [N, D]  — pair-level anchor embeddings
        positive_embs: torch.Tensor,  # [N, D]  — pair-level positive embeddings
        hard_negatives: np.ndarray,   # [N, K]  — indices into pairs
        id2idx: dict[str, int],
    ):
        self.anchor_embs = anchor_embs
        self.positive_embs = positive_embs
        self.hard_negatives = hard_negatives  # indices into pairs list
        self.k_neg = hard_negatives.shape[1] if hard_negatives is not None else 0

        # Build items: (anchor_pair_idx, source_doc_idx, target_doc_idx)
        self.items: list[tuple[int, int, int]] = []
        self.pair_to_tgt_doc: dict[int, int] = {}  # pair_idx → target_doc_idx

        for i, (_, _, meta) in enumerate(pairs):
            sid, tid = meta["source_doc_id"], meta["target_doc_id"]
            if sid in id2idx and tid in id2idx:
                self.items.append((i, id2idx[sid], id2idx[tid]))
                self.pair_to_tgt_doc[i] = id2idx[tid]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pair_idx, src_doc, tgt_doc = self.items[idx]
        a_emb = self.anchor_embs[pair_idx]    # [D]
        p_emb = self.positive_embs[pair_idx]  # [D]

        # Hard neg: text emb + doc idx for each hard negative
        if self.k_neg > 0:
            neg_pair_idxs = self.hard_negatives[pair_idx].tolist()  # list of k ints
            neg_text_embs = self.positive_embs[neg_pair_idxs]  # [K, D]
            neg_doc_idxs = torch.tensor(
                [self.pair_to_tgt_doc.get(int(j), tgt_doc) for j in neg_pair_idxs],
                dtype=torch.long,
            )
        else:
            neg_text_embs = torch.zeros(0, a_emb.size(0))
            neg_doc_idxs = torch.zeros(0, dtype=torch.long)

        return (
            a_emb,                              # anchor text emb [D]
            p_emb,                              # positive text emb [D]
            torch.tensor(src_doc, dtype=torch.long),  # cause doc idx
            torch.tensor(tgt_doc, dtype=torch.long),  # effect doc idx
            neg_text_embs,                      # hard neg text embs [K, D]
            neg_doc_idxs,                       # hard neg doc idxs [K]
        )


def collate_hybrid(batch):
    a_emb  = torch.stack([b[0] for b in batch])            # [B, D]
    p_emb  = torch.stack([b[1] for b in batch])            # [B, D]
    src    = torch.stack([b[2] for b in batch])            # [B]
    tgt    = torch.stack([b[3] for b in batch])            # [B]
    K = batch[0][4].shape[0]
    if K > 0:
        neg_t = torch.stack([b[4] for b in batch])         # [B, K, D]
        neg_d = torch.stack([b[5] for b in batch])         # [B, K]
    else:
        D = a_emb.shape[1]
        neg_t = torch.zeros(len(batch), 0, D)
        neg_d = torch.zeros(len(batch), 0, dtype=torch.long)
    return a_emb, p_emb, src, tgt, neg_t, neg_d


# ---------------------------------------------------------------------------
# InfoNCE with hard negatives
# ---------------------------------------------------------------------------

class HardNegInfoNCE(nn.Module):
    def __init__(self, temperature_init: float = 0.07, learnable: bool = True,
                 temperature_min: float = 0.02):
        super().__init__()
        log_tau = torch.tensor(temperature_init).log()
        if learnable:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)
        self.temperature_min = temperature_min

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(self.temperature_min, 0.5)

    def forward(
        self,
        cause_emb: torch.Tensor,   # [B, D] — L2-normalised
        effect_emb: torch.Tensor,  # [B, D] — L2-normalised
        neg_emb: torch.Tensor | None = None,  # [B, K, D]
    ) -> tuple[torch.Tensor, dict]:
        B = cause_emb.size(0)
        tau = self.temperature
        device = cause_emb.device

        # In-batch similarity [B, B]
        sim_ib = cause_emb @ effect_emb.T

        if neg_emb is not None and neg_emb.numel() > 0:
            # [B, K] — anchor-specific hard neg scores
            sim_hn = torch.einsum("bd,bkd->bk", cause_emb, neg_emb)
            logits = torch.cat([sim_ib, sim_hn], dim=1) / tau  # [B, B+K]
        else:
            logits = sim_ib / tau

        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            diag = sim_ib.diag()
            off = ~torch.eye(B, dtype=torch.bool, device=device)
            stats = {
                "temperature": float(tau),
                "pos_sim":     float(diag.mean()),
                "neg_sim":     float(sim_ib[off].mean()),
                "hn_sim":      float(sim_hn.mean()) if (neg_emb is not None and neg_emb.numel() > 0) else 0.0,
            }
        return loss, stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: HybridCauseEffectGNN,
    val_pairs: list[tuple[str, str, dict]],
    val_anchor_embs: torch.Tensor,    # [N_val, D] — pair-level
    val_positive_embs: torch.Tensor,  # [N_val, D]
    id2idx: dict[str, int],
    x_text: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    model.eval()
    check_memory("validate")
    all_h = model.encode_graph(x_text.to(device), edge_index.to(device))

    cause_embs, effect_embs = [], []
    for i, (_, _, meta) in enumerate(val_pairs):
        sid, tid = meta["source_doc_id"], meta["target_doc_id"]
        if sid not in id2idx or tid not in id2idx:
            continue
        src_doc = id2idx[sid]
        tgt_doc = id2idx[tid]

        h_c = all_h[src_doc].unsqueeze(0)
        h_e = all_h[tgt_doc].unsqueeze(0)
        a_t = val_anchor_embs[i].unsqueeze(0).to(device)
        p_t = val_positive_embs[i].unsqueeze(0).to(device)

        cause_embs.append(model.fuse_cause(a_t, h_c))
        effect_embs.append(model.fuse_effect(p_t, h_e))

    if not cause_embs:
        return {"mrr": 0.0}

    A = torch.cat(cause_embs, dim=0)   # [N_val, D]
    P = torch.cat(effect_embs, dim=0)  # [N_val, D]

    sim = A @ P.T  # [N, N]
    diag = sim.diag().unsqueeze(1)
    ranks = (sim > diag).sum(dim=1).float() + 1

    out: dict = {"mrr": float((1.0 / ranks).mean())}
    for k in k_values:
        out[f"recall@{k}"] = float((ranks <= k).float().mean())
    out["mean_rank"] = float(ranks.mean())
    out["median_rank"] = float(ranks.median())
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: HybridCauseEffectGNN,
    loss_fn: HardNegInfoNCE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    x_text: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    model.train()
    running_loss = 0.0
    n_batches = 0
    last_stats: dict = {}

    x_dev = x_text.to(device)
    ei_dev = edge_index.to(device)

    for a_emb, p_emb, src_doc, tgt_doc, neg_t, neg_d in loader:
        a_emb   = a_emb.to(device)
        p_emb   = p_emb.to(device)
        src_doc = src_doc.to(device)
        tgt_doc = tgt_doc.to(device)

        optimizer.zero_grad()

        # Full-graph GNN forward (fixed graph, only GNN weights change per batch)
        all_h = model.encode_graph(x_dev, ei_dev)

        # Lookup graph embeddings for batch nodes
        h_src = all_h[src_doc]   # [B, gnn_dim]
        h_tgt = all_h[tgt_doc]   # [B, gnn_dim]

        # Fuse text + graph → L2-normalised embeddings
        cause_emb  = model.fuse_cause(a_emb, h_src)   # [B, out_dim]
        effect_emb = model.fuse_effect(p_emb, h_tgt)  # [B, out_dim]

        K = neg_t.shape[1]
        if K > 0:
            neg_t = neg_t.to(device)   # [B, K, D]
            neg_d = neg_d.to(device)   # [B, K]
            h_negs = all_h[neg_d]      # [B, K, gnn_dim]
            neg_emb = model.fuse_effect(neg_t, h_negs)  # [B, K, out_dim]
        else:
            neg_emb = None

        loss, stats = loss_fn(cause_emb, effect_emb, neg_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        last_stats = stats

    return running_loss / max(n_batches, 1), last_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal", choices=list(SPLITS_CFG.keys()))
    ap.add_argument("--results-dir", default=str(ROOT / "gnn/results"))
    ap.add_argument("--encoder-model", default="BAAI/bge-base-en-v1.5",
                    help="Frozen sentence encoder for all text embeddings + hard neg mining.")
    ap.add_argument("--gnn-dim", type=int, default=256,
                    help="GNN hidden dimension.")
    ap.add_argument("--out-dim", type=int, default=256,
                    help="Output embedding dimension for scoring.")
    ap.add_argument("--num-gnn-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--k-neg", type=int, default=8,
                    help="Hard negatives per anchor.")
    ap.add_argument("--anchor-threshold", type=float, default=0.8,
                    help="Cosine sim threshold: skip hard negatives whose anchors "
                         "are too similar to the query anchor.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup-epochs", type=int, default=8,
                    help="Epochs to train text heads only before unfreezing GNN. "
                         "0 = no staged training.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="LR for text projection heads.")
    ap.add_argument("--gnn-lr", type=float, default=3e-4,
                    help="LR for GNN components (lower than text lr).")
    ap.add_argument("--temperature-lr", type=float, default=1e-2,
                    help="LR for log-temperature (needs to be ~10x higher than lr).")
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--temperature-min", type=float, default=0.02,
                    help="Minimum temperature (prevents logits from becoming too sharp).")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-encode", action="store_true")
    ap.add_argument("--force-mine", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. 0 works safely with pre-loaded tensors.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[train_gnn]  dataset={args.dataset}  device={device}")
    print(f"  encoder={args.encoder_model}")
    print(f"  gnn_dim={args.gnn_dim}  layers={args.num_gnn_layers}  "
          f"heads={args.heads}  k_neg={args.k_neg}")
    print("=" * 72)
    check_memory("start")

    # ---- Load pairs ----
    cfg = SPLITS_CFG[args.dataset]
    data_dir = DATA_DIR / args.dataset
    train_pairs = load_pairs_jsonl(data_dir / cfg["train"])
    val_pairs   = load_pairs_jsonl(data_dir / cfg["val"])
    test_pairs  = load_pairs_jsonl(data_dir / cfg["test"])
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")
    check_memory("after load_pairs")

    # ---- Document registry + graph ----
    id2idx, idx2id, doc_texts = build_doc_registry(train_pairs, val_pairs + test_pairs)
    N_docs = len(doc_texts)
    print(f"  unique docs: {N_docs}")
    edge_index, edge_weight = build_graph(train_pairs, id2idx)
    print(f"  edges: {edge_index.size(1)}")
    check_memory("after build_graph")

    # ---- Document embeddings for GNN nodes (cached) ----
    doc_cache = results_dir / "doc_embeddings.npy"
    if args.force_encode and doc_cache.exists():
        doc_cache.unlink()
    x_text = encode_documents(doc_texts, doc_cache, model_name=args.encoder_model)
    text_dim = x_text.size(1)
    print(f"  text_dim={text_dim}  GNN node emb: {x_text.shape}")
    check_memory("after encode_docs")

    # ---- Pair-level text embeddings for training (cached) ----
    train_anchors   = [a for a, _, _ in train_pairs]
    train_positives = [p for _, p, _ in train_pairs]
    val_anchors     = [a for a, _, _ in val_pairs]
    val_positives   = [p for _, p, _ in val_pairs]

    train_a_cache = results_dir / "train_anchor_embs.npy"
    train_p_cache = results_dir / "train_positive_embs.npy"
    val_a_cache   = results_dir / "val_anchor_embs.npy"
    val_p_cache   = results_dir / "val_positive_embs.npy"

    if args.force_encode:
        for c in [train_a_cache, train_p_cache, val_a_cache, val_p_cache]:
            c.unlink(missing_ok=True)

    print("  encoding training pairs …")
    train_a_embs, train_p_embs = encode_pair_texts(
        train_anchors, train_positives,
        train_a_cache, train_p_cache, model_name=args.encoder_model,
    )
    print("  encoding val pairs …")
    val_a_embs, val_p_embs = encode_pair_texts(
        val_anchors, val_positives,
        val_a_cache, val_p_cache, model_name=args.encoder_model,
    )
    print(f"  train embs: a={train_a_embs.shape}  p={train_p_embs.shape}")
    check_memory("after encode_pairs")

    # ---- Hard negative mining ----
    mine_cache = results_dir / "gnn_hard_negatives.npy"
    if args.force_mine and mine_cache.exists():
        mine_cache.unlink()

    if mine_cache.exists():
        print(f"  loading cached hard negatives: {mine_cache}")
        hard_neg_idx = np.load(mine_cache)
    else:
        hard_neg_idx, _ = load_and_mine(
            list(zip(train_anchors, train_positives)),
            k=args.k_neg,
            anchor_sim_threshold=args.anchor_threshold,
            out_dir=results_dir,
            dataset_name="",
            model_name=args.encoder_model,
        )
    print(f"  hard_neg_idx shape: {hard_neg_idx.shape}")
    check_memory("after mining")

    # ---- Dataset & DataLoader ----
    train_ds = HybridPairDataset(
        train_pairs, train_a_embs, train_p_embs,
        hard_neg_idx, id2idx,
    )
    print(f"  train dataset: {len(train_ds)} items  k_neg={train_ds.k_neg}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_hybrid, num_workers=args.num_workers,
        pin_memory=False, drop_last=True,
    )

    # ---- Model ----
    model = HybridCauseEffectGNN(
        text_dim=text_dim,
        gnn_dim=args.gnn_dim,
        out_dim=args.out_dim,
        num_gnn_layers=args.num_gnn_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    loss_fn = HardNegInfoNCE(
        temperature_init=args.temperature_init,
        learnable=True,
        temperature_min=args.temperature_min,
    ).to(device)

    # Separate parameter groups: text heads (high lr), GNN (lower lr), temperature (10x lr)
    text_params = list(model.W_text_cause.parameters()) + list(model.W_text_effect.parameters())
    gnn_params  = (list(model.gnn_input_proj.parameters()) +
                   list(model.gnn_layers.parameters()) +
                   list(model.W_gnn_cause.parameters()) +
                   list(model.W_gnn_effect.parameters()))
    optimizer = AdamW([
        {"params": text_params,       "lr": args.lr,             "weight_decay": 1e-4},
        {"params": gnn_params,        "lr": args.gnn_lr,         "weight_decay": 1e-4},
        {"params": [loss_fn.log_tau], "lr": args.temperature_lr, "weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Config ----
    cfg_out = {
        "dataset": args.dataset, "encoder_model": args.encoder_model,
        "gnn_dim": args.gnn_dim, "out_dim": args.out_dim,
        "num_gnn_layers": args.num_gnn_layers, "heads": args.heads,
        "dropout": args.dropout, "k_neg": args.k_neg,
        "anchor_threshold": args.anchor_threshold,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr, "gnn_lr": args.gnn_lr, "temperature_lr": args.temperature_lr,
        "temperature_init": args.temperature_init, "temperature_min": args.temperature_min,
        "n_docs": N_docs, "n_train": len(train_ds), "n_val": len(val_pairs),
        "n_edges": int(edge_index.size(1)), "text_dim": text_dim,
        "device": str(device),
    }
    (results_dir / "gnn_config.json").write_text(json.dumps(cfg_out, indent=2))

    # ---- Training ----
    best_mrr = -1.0
    history = []
    best_path = results_dir / "gnn_best.pt"
    x_text_dev  = x_text.to(device)
    edge_idx_dev = edge_index.to(device)

    # ----- Staged training: freeze GNN in warmup phase -----
    def freeze_gnn(model, frozen: bool):
        for p in model.gnn_input_proj.parameters(): p.requires_grad = not frozen
        for p in model.gnn_layers.parameters():     p.requires_grad = not frozen
        for p in model.W_gnn_cause.parameters():    p.requires_grad = not frozen
        for p in model.W_gnn_effect.parameters():   p.requires_grad = not frozen

    if args.warmup_epochs > 0:
        freeze_gnn(model, frozen=True)
        print(f"  staged training: GNN frozen for first {args.warmup_epochs} epochs")

    print()
    print("=" * 72)
    print(f"Training {args.epochs} epochs · batch={args.batch_size} · K={train_ds.k_neg}")
    print(f"  lr={args.lr}  gnn_lr={args.gnn_lr}  tau_lr={args.temperature_lr}")
    print(f"  warmup_epochs={args.warmup_epochs} (GNN frozen until then)")
    print("=" * 72)

    for epoch in range(args.epochs):
        # Unfreeze GNN after warmup
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            freeze_gnn(model, frozen=False)
            print(f"\n  [epoch {epoch+1}] GNN unfrozen — full model now training")
        t0 = time.time()
        train_loss, stats = train_epoch(
            model, loss_fn, train_loader, optimizer,
            x_text_dev, edge_idx_dev, device,
            grad_clip=args.grad_clip,
        )
        scheduler.step()

        val_metrics = validate(
            model, val_pairs,
            val_a_embs, val_p_embs,
            id2idx, x_text_dev, edge_idx_dev, device,
        )
        epoch_time = time.time() - t0

        print(
            f"  epoch {epoch+1:3d}/{args.epochs}  {epoch_time:.0f}s  "
            f"loss={train_loss:.4f}  τ={stats.get('temperature',0):.4f}  "
            f"pos={stats.get('pos_sim',0):.3f}  hn={stats.get('hn_sim',0):.3f}  "
            f"|| val MRR={val_metrics['mrr']:.4f}  "
            f"R@1={val_metrics.get('recall@1',0):.3f}  "
            f"R@5={val_metrics.get('recall@5',0):.3f}"
        )

        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "loss_fn_state": loss_fn.state_dict(),
                "val_metrics": val_metrics, "cfg": cfg_out,
                "id2idx": id2idx, "idx2id": idx2id,
            }, best_path)
            print(f"    → new best  MRR={best_mrr:.4f}  saved")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val": val_metrics, **stats,
        })

        if (epoch + 1) % 5 == 0:
            check_memory(f"epoch{epoch+1}")

    (results_dir / "gnn_training_history.json").write_text(json.dumps(history, indent=2))
    print()
    print(f"[done]  best val MRR={best_mrr:.4f}  checkpoint → {best_path}")


if __name__ == "__main__":
    main()
