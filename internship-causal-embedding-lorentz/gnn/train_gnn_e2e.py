"""End-to-end Cause→Effect GNN with a fine-tunable strong encoder.

The goal: BEAT the fine-tuned bert-base-uncased two-tower BiEncoder baseline
by >10% on retrieval metrics.

KEY ARCHITECTURAL CHANGES vs the earlier frozen-feature GNN:
  1. STRONGER BACKBONE: BGE-base-en-v1.5 (a real retrieval encoder) instead of
     bert-base-uncased. Fine-tuned end-to-end. This is the main lever.
  2. UNTIED TWO-TOWER: separate anchor / positive encoders (matches baseline).
  3. GNN STRUCTURAL RESIDUAL on a STABLE graph: the GNN runs on FROZEN initial
     BGE document embeddings (so graph node features never drift), and produces
     a small additive correction to the live text embedding. A learnable gate
     controls how much structure is mixed in. At init the model == pure
     fine-tuned two-tower; the GNN only helps if it reduces the loss.
  4. Hard negatives: semantic kNN (MiniLM, same as finetune_eval) + anchor
     similarity filter (drop negatives whose anchor ≈ the query anchor).

WHY the stable-graph design (vs re-encoding the graph each step):
  Re-encoding 2.5K docs through the live encoder every step is slow AND makes
  the GNN input drift, which destabilises training (the earlier
  cache-refresh approach collapsed to val MRR≈0.08). Freezing the graph node
  features decouples the GNN (stable structural signal) from the encoder
  (fine-tuned semantic signal); W_gnn learns the mapping between the two spaces.

Memory guard: check_memory() before every heavy step; abort if <4 GB free.

Usage:
  PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY gnn/train_gnn_e2e.py --dataset aep_causal \\
      --encoder BAAI/bge-base-en-v1.5 --epochs 8 --batch-size 32
  # encoder-only ablation (no GNN):
  CUDA_VISIBLE_DEVICES=0 $PY gnn/train_gnn_e2e.py --no-gnn ...
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

from gnn.model import DirectionalGATLayer                  # noqa: E402
from gnn.train_gnn import (                                 # noqa: E402
    load_pairs_jsonl, build_doc_registry, build_graph,
    SPLITS_CFG, DATA_DIR, check_memory,
)


# ---------------------------------------------------------------------------
# Encoder tower (fine-tunable, mean-pooled, L2-normalised)
# ---------------------------------------------------------------------------

class EncoderTower(nn.Module):
    def __init__(self, model_name: str, max_length: int = 256):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (tok * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, dim=-1)


def make_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize(tokenizer, texts, max_length, device):
    enc = tokenizer(texts, max_length=max_length, padding=True,
                    truncation=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def encode_all(tower, tokenizer, texts, max_length, device, batch_size=128):
    tower.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        ids, mask = tokenize(tokenizer, texts[i:i + batch_size], max_length, device)
        out.append(tower(ids, mask).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, tower.hidden_size), dtype=np.float32)


# ---------------------------------------------------------------------------
# GNN structural module (operates on a fixed graph; produces a correction)
# ---------------------------------------------------------------------------

class StructuralGNN(nn.Module):
    """Directional GNN that maps fixed doc node features → asymmetric
    cause/effect correction vectors in the encoder's output space."""

    def __init__(self, text_dim, gnn_dim=256, out_dim=768, num_layers=2,
                 heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(text_dim, gnn_dim, bias=False)
        self.layers = nn.ModuleList([
            DirectionalGATLayer(gnn_dim, gnn_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.W_cause = nn.Linear(gnn_dim, out_dim, bias=False)
        self.W_effect = nn.Linear(gnn_dim, out_dim, bias=False)
        # small init so the correction starts tiny
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.W_cause.weight, std=0.02)
        nn.init.normal_(self.W_effect.weight, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x_text, edge_index):
        h = self.input_proj(x_text)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h                                   # [N, gnn_dim]

    def cause_correction(self, h_nodes):
        return self.W_cause(self.dropout(h_nodes))   # [.., out_dim]

    def effect_correction(self, h_nodes):
        return self.W_effect(self.dropout(h_nodes))  # [.., out_dim]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class E2EDataset(Dataset):
    def __init__(self, pairs, hard_negatives, id2idx):
        self.pairs = [(a, p, m) for a, p, m in pairs
                      if m["source_doc_id"] in id2idx and m["target_doc_id"] in id2idx]
        self.orig_idx = [i for i, (a, p, m) in enumerate(pairs)
                         if m["source_doc_id"] in id2idx and m["target_doc_id"] in id2idx]
        self.all_positives = [p for _, p, _ in pairs]
        self.all_meta = [m for _, _, m in pairs]
        self.hard_negatives = hard_negatives
        self.id2idx = id2idx

    @property
    def k_neg(self):
        return 0 if self.hard_negatives is None else int(self.hard_negatives.shape[1])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, p, m = self.pairs[idx]
        orig = self.orig_idx[idx]
        src = self.id2idx[m["source_doc_id"]]
        tgt = self.id2idx[m["target_doc_id"]]
        K = self.k_neg
        if K > 0:
            hn = self.hard_negatives[orig].tolist()
            hn_texts = [self.all_positives[int(j)] for j in hn]
            hn_docs = [self.id2idx.get(self.all_meta[int(j)]["target_doc_id"], tgt) for j in hn]
        else:
            hn_texts, hn_docs = [], []
        return a, p, src, tgt, hn_texts, hn_docs


def collate(batch):
    a = [b[0] for b in batch]
    p = [b[1] for b in batch]
    src = torch.tensor([b[2] for b in batch], dtype=torch.long)
    tgt = torch.tensor([b[3] for b in batch], dtype=torch.long)
    hn_texts = [b[4] for b in batch]
    hn_docs = [b[5] for b in batch]
    return a, p, src, tgt, hn_texts, hn_docs


# ---------------------------------------------------------------------------
# Full model: two towers + GNN + learnable gates
# ---------------------------------------------------------------------------

class E2EModel(nn.Module):
    def __init__(self, encoder_name, max_length, use_gnn,
                 gnn_dim=256, num_gnn_layers=2, heads=4, dropout=0.1,
                 gate_init=0.3):
        super().__init__()
        self.anchor_tower = EncoderTower(encoder_name, max_length)
        self.positive_tower = EncoderTower(encoder_name, max_length)
        self.hidden = self.anchor_tower.hidden_size
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = StructuralGNN(self.hidden, gnn_dim=gnn_dim,
                                     out_dim=self.hidden, num_layers=num_gnn_layers,
                                     heads=heads, dropout=dropout)
            # learnable mixing gates (start small so encoder dominates)
            self.gate_cause = nn.Parameter(torch.tensor(float(gate_init)))
            self.gate_effect = nn.Parameter(torch.tensor(float(gate_init)))

    def encode_graph(self, x_text, edge_index):
        return self.gnn.encode(x_text, edge_index) if self.use_gnn else None

    def cause_from(self, a_emb, h_src):
        if not self.use_gnn:
            return a_emb
        corr = self.gnn.cause_correction(h_src)
        return F.normalize(a_emb + self.gate_cause * corr, dim=-1)

    def effect_from(self, p_emb, h_tgt):
        if not self.use_gnn:
            return p_emb
        corr = self.gnn.effect_correction(h_tgt)
        return F.normalize(p_emb + self.gate_effect * corr, dim=-1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class HardNegInfoNCE(nn.Module):
    def __init__(self, tau_init=0.05, tau_min=0.01):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(tau_init).log())
        self.tau_min = tau_min

    @property
    def temperature(self):
        return self.log_tau.exp().clamp(self.tau_min, 0.5)

    def forward(self, cause, effect, neg=None):
        B = cause.size(0); tau = self.temperature; dev = cause.device
        sim = cause @ effect.T                                   # [B, B]
        if neg is not None and neg.numel() > 0:
            hn = torch.einsum("bd,bkd->bk", cause, neg)          # [B, K]
            logits = torch.cat([sim, hn], 1) / tau
        else:
            logits = sim / tau
        loss = F.cross_entropy(logits, torch.arange(B, device=dev))
        with torch.no_grad():
            off = ~torch.eye(B, dtype=torch.bool, device=dev)
            st = {"tau": float(tau), "pos": float(sim.diag().mean()),
                  "neg": float(sim[off].mean()),
                  "hn": float(hn.mean()) if (neg is not None and neg.numel() > 0) else 0.0}
        return loss, st


# ---------------------------------------------------------------------------
# Validation (full N×N MRR + Recall@K)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, tok_a, tok_p, val_pairs, id2idx, x_text, edge_index,
             device, max_length, k=(1, 3, 5, 10)):
    model.eval()
    anchors = [a for a, _, _ in val_pairs]
    positives = [p for _, p, _ in val_pairs]
    a_all = torch.from_numpy(encode_all(model.anchor_tower, tok_a, anchors, max_length, device)).float()
    p_all = torch.from_numpy(encode_all(model.positive_tower, tok_p, positives, max_length, device)).float()

    h_nodes = model.encode_graph(x_text.to(device), edge_index.to(device)) if model.use_gnn else None

    cs, es = [], []
    for i, (_, _, m) in enumerate(val_pairs):
        sid, tid = m["source_doc_id"], m["target_doc_id"]
        if sid not in id2idx or tid not in id2idx:
            continue
        a_e = a_all[i:i + 1].to(device)
        p_e = p_all[i:i + 1].to(device)
        if model.use_gnn:
            cs.append(model.cause_from(a_e, h_nodes[id2idx[sid]].unsqueeze(0)).cpu())
            es.append(model.effect_from(p_e, h_nodes[id2idx[tid]].unsqueeze(0)).cpu())
        else:
            cs.append(a_e.cpu()); es.append(p_e.cpu())
    if not cs:
        return {"mrr": 0.0}
    A = torch.cat(cs); P = torch.cat(es)
    sim = A @ P.T
    ranks = (sim > sim.diag().unsqueeze(1)).sum(1).float() + 1
    out = {"mrr": float((1 / ranks).mean()), "mean_rank": float(ranks.mean())}
    for kk in k:
        out[f"recall@{kk}"] = float((ranks <= kk).float().mean())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal", choices=list(SPLITS_CFG.keys()))
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--results-dir", default=str(ROOT / "gnn/results"))
    ap.add_argument("--run-name", default=None, help="Subdir name; default derived from flags.")
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--no-gnn", action="store_true", help="Encoder-only ablation.")
    ap.add_argument("--gnn-dim", type=int, default=256)
    ap.add_argument("--num-gnn-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--gate-init", type=float, default=0.3)
    ap.add_argument("--k-neg", type=int, default=4)
    ap.add_argument("--anchor-threshold", type=float, default=0.9)
    ap.add_argument("--mine-encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Frozen encoder for hard-neg mining (matches finetune_eval).")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--temperature-lr", type=float, default=1e-2)
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force-mine", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="Run dir of an existing checkpoint to init the encoder towers from "
                         "(e.g. the encoder-only run). Enables two-stage GNN training.")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Freeze the encoder towers (train only GNN + heads). Use with "
                         "--init-from to add a GNN on top of a fully fine-tuned encoder "
                         "with MATCHED graph features.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or (
        f"{args.dataset}_e2e_{'nognn' if args.no_gnn else 'gnn'}_"
        f"{args.encoder.split('/')[-1]}"
    )
    results_dir = Path(args.results_dir) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"[train_gnn_e2e]  dataset={args.dataset}  device={device}")
    print(f"  encoder={args.encoder}  use_gnn={not args.no_gnn}  k_neg={args.k_neg}")
    print(f"  run_dir={results_dir}")
    print("=" * 74)
    check_memory("start")

    # ---- Data ----
    cfg = SPLITS_CFG[args.dataset]
    dd = DATA_DIR / args.dataset
    train_pairs = load_pairs_jsonl(dd / cfg["train"])
    val_pairs = load_pairs_jsonl(dd / cfg["val"])
    test_pairs = load_pairs_jsonl(dd / cfg["test"])
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    id2idx, idx2id, doc_texts = build_doc_registry(train_pairs, val_pairs + test_pairs)
    N_docs = len(doc_texts)
    edge_index, _ = build_graph(train_pairs, id2idx)
    print(f"  docs={N_docs}  edges={edge_index.size(1)}")
    check_memory("graph built")

    # ---- Model ----
    print(f"  loading encoder towers: {args.encoder}")
    model = E2EModel(
        args.encoder, args.max_seq_length, use_gnn=not args.no_gnn,
        gnn_dim=args.gnn_dim, num_gnn_layers=args.num_gnn_layers,
        heads=args.heads, dropout=args.dropout, gate_init=args.gate_init,
    ).to(device)
    tok_a = make_tokenizer(args.encoder)
    tok_p = tok_a  # same tokenizer for both towers

    # ---- Optional: init encoder towers from an existing checkpoint ----
    if args.init_from:
        init_ckpt = Path(args.init_from)
        if init_ckpt.is_dir():
            init_ckpt = init_ckpt / "best.pt"
        print(f"  init-from: loading tower weights from {init_ckpt}")
        prev = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        # load only the encoder-tower weights (ignore GNN keys from either side)
        prev_state = prev["model_state"]
        tower_state = {k: v for k, v in prev_state.items()
                       if k.startswith("anchor_tower.") or k.startswith("positive_tower.")}
        missing, unexpected = model.load_state_dict(tower_state, strict=False)
        print(f"    loaded {len(tower_state)} tower tensors  "
              f"(missing={len(missing)} unexpected={len(unexpected)})")

    # ---- Optional: freeze encoder (train only GNN/heads) ----
    if args.freeze_encoder:
        for p in model.anchor_tower.parameters():
            p.requires_grad = False
        for p in model.positive_tower.parameters():
            p.requires_grad = False
        model.anchor_tower.eval()
        model.positive_tower.eval()
        print("  encoder towers FROZEN — training only GNN + gates + temperature")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,} total  /  {n_train:,} trainable")
    check_memory("model loaded")

    # ---- Frozen initial doc embeddings for the GNN graph (stable) ----
    x_text = None
    if model.use_gnn:
        print("  encoding initial doc embeddings for GNN graph (frozen) …")
        d_emb = encode_all(model.positive_tower, tok_p, doc_texts, args.max_seq_length, device)
        x_text = torch.from_numpy(d_emb).float()
        check_memory("doc embs")

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
            [a for a, _, _ in train_pairs], [p for _, p, _ in train_pairs],
            k=args.k_neg, anchor_sim_threshold=args.anchor_threshold,
            model_name=args.mine_encoder,
        )
        np.save(mine_cache, hard_neg)
    print(f"  hard_neg: {hard_neg.shape}")
    check_memory("mining done")

    # ---- Loader ----
    train_ds = E2EDataset(train_pairs, hard_neg, id2idx)
    K = train_ds.k_neg
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0, drop_last=True)
    print(f"  train items={len(train_ds)}  K={K}  steps/epoch={len(loader)}")

    # ---- Optimizer ----
    loss_fn = HardNegInfoNCE(args.temperature_init).to(device)
    enc_params = [p for p in (list(model.anchor_tower.parameters()) +
                              list(model.positive_tower.parameters())) if p.requires_grad]
    groups = []
    max_lrs = []
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr, "weight_decay": 0.01})
        max_lrs.append(args.encoder_lr)
    if model.use_gnn:
        head_params = list(model.gnn.parameters()) + [model.gate_cause, model.gate_effect]
        groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": 1e-4})
        max_lrs.append(args.head_lr)
    groups.append({"params": [loss_fn.log_tau], "lr": args.temperature_lr, "weight_decay": 0.0})
    max_lrs.append(args.temperature_lr)
    opt = AdamW(groups)
    total_steps = len(loader) * args.epochs
    sched = OneCycleLR(opt, max_lr=max_lrs, total_steps=total_steps,
                       pct_start=args.warmup_ratio, anneal_strategy="cos")

    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
          if (args.bf16 and device.type == "cuda") else (lambda: nullcontext())

    cfg_out = {
        "dataset": args.dataset, "encoder": args.encoder, "use_gnn": model.use_gnn,
        "gnn_dim": args.gnn_dim, "num_gnn_layers": args.num_gnn_layers, "heads": args.heads,
        "dropout": args.dropout, "gate_init": args.gate_init, "k_neg": args.k_neg,
        "anchor_threshold": args.anchor_threshold, "mine_encoder": args.mine_encoder,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "temperature_init": args.temperature_init, "max_seq_length": args.max_seq_length,
        "text_dim": model.hidden, "n_docs": N_docs, "device": str(device),
    }
    (results_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))

    # ---- Train ----
    best_mrr = -1.0; history = []
    best_path = results_dir / "best.pt"
    x_dev = x_text.to(device) if x_text is not None else None
    ei_dev = edge_index.to(device)

    print()
    print("=" * 74)
    print(f"Training {args.epochs} epochs · batch={args.batch_size} · K={K} · "
          f"enc_lr={args.encoder_lr} · gnn={'on' if model.use_gnn else 'OFF'}")
    print("=" * 74)

    gstep = 0
    enc_ctx = torch.no_grad if args.freeze_encoder else nullcontext
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:                       # keep frozen towers in eval (no dropout)
            model.anchor_tower.eval(); model.positive_tower.eval()
        t0 = time.time(); rl = 0.0; nb = 0; last = {}
        for a_texts, p_texts, src, tgt, hn_texts, hn_docs in loader:
            src = src.to(device); tgt = tgt.to(device)
            opt.zero_grad()
            with enc_ctx():
                with amp():
                    a_ids, a_mask = tokenize(tok_a, a_texts, args.max_seq_length, device)
                    p_ids, p_mask = tokenize(tok_p, p_texts, args.max_seq_length, device)
                    a_emb = model.anchor_tower(a_ids, a_mask)
                    p_emb = model.positive_tower(p_ids, p_mask)
            a_emb = a_emb.float(); p_emb = p_emb.float()

            if model.use_gnn:
                h_nodes = model.encode_graph(x_dev, ei_dev)
                cause = model.cause_from(a_emb, h_nodes[src])
                effect = model.effect_from(p_emb, h_nodes[tgt])
            else:
                h_nodes = None
                cause, effect = a_emb, p_emb

            if K > 0:
                flat = [t for row in hn_texts for t in row]
                with enc_ctx():
                    with amp():
                        hn_ids, hn_mask = tokenize(tok_p, flat, args.max_seq_length, device)
                        hn_emb = model.positive_tower(hn_ids, hn_mask).float()
                if model.use_gnn:
                    flat_d = torch.tensor([d for row in hn_docs for d in row], device=device)
                    corr = model.gnn.effect_correction(h_nodes[flat_d])
                    hn_emb = F.normalize(hn_emb + model.gate_effect * corr, dim=-1)
                neg = hn_emb.view(len(a_texts), K, -1)
            else:
                neg = None

            loss, st = loss_fn(cause, effect, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            rl += loss.item(); nb += 1; last = st; gstep += 1

        vm = validate(model, tok_a, tok_p, val_pairs, id2idx, x_text, edge_index,
                      device, args.max_seq_length)
        gate_str = ""
        if model.use_gnn:
            gate_str = f"  gate_c={model.gate_cause.item():.3f} gate_e={model.gate_effect.item():.3f}"
        print(f"  ep {epoch+1}/{args.epochs}  {int(time.time()-t0)}s  "
              f"loss={rl/max(nb,1):.4f}  τ={last.get('tau',0):.4f}  "
              f"pos={last.get('pos',0):.3f} hn={last.get('hn',0):.3f}{gate_str}  "
              f"|| val MRR={vm['mrr']:.4f}  R@1={vm.get('recall@1',0):.3f}  "
              f"R@5={vm.get('recall@5',0):.3f}  R@10={vm.get('recall@10',0):.3f}")

        if vm["mrr"] > best_mrr:
            best_mrr = vm["mrr"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "loss_fn_state": loss_fn.state_dict(),
                        "val_metrics": vm, "cfg": cfg_out,
                        "id2idx": id2idx, "idx2id": idx2id}, best_path)
            # also cache the frozen doc embeddings for eval
            if x_text is not None:
                np.save(results_dir / "doc_embs.npy", x_text.numpy())
            print(f"    → best val MRR={best_mrr:.4f}  saved")
        history.append({"epoch": epoch + 1, "val": vm, **last})
        check_memory(f"ep{epoch+1}")

    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done]  best val MRR={best_mrr:.4f}  → {best_path}")


if __name__ == "__main__":
    main()
