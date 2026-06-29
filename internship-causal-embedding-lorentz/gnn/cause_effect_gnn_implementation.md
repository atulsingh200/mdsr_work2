# Cause→Effect GNN: Implementation Guide

A complete reference implementation of a **text-encoder + directional-GNN + asymmetric-scoring** architecture for cause→effect ranking, based on SGNN (Li, Ding & Liu, IJCAI 2018) modernized with a sentence-transformer front-end.

**Task:** Given a cause document and N candidate effect documents, rank them and pick the correct one.

---

## Architecture Overview

```
       ┌──────────────────────────┐
       │ Document text            │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Sentence Encoder         │   ← frozen pretrained model
       │ (sentence-transformers)  │     (computed once, cached)
       └────────────┬─────────────┘
                    │  h⁽⁰⁾ ∈ ℝ⁷⁶⁸
                    ▼
       ┌──────────────────────────┐
       │ Input projection         │   ← trainable
       │ Linear: 768 → 256        │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Directional GAT × L      │   ← trainable
       │ (incoming + outgoing)    │     L = 2 layers
       └────────────┬─────────────┘
                    │  h⁽ᴸ⁾ ∈ ℝ²⁵⁶
        ┌───────────┴───────────┐
        ▼                       ▼
    W_cause                 W_effect
        │                       │
        ▼                       ▼
      φ_c(c)  ────⟨·,·⟩────  φ_e(t)
                    │
                    ▼
                 score s(c, t)
```

**Key design choices:**
- **One shared text encoder + one shared GNN.** Same weights for cause and effect side.
- **Two asymmetric projection heads** (W_cause, W_effect) at the output. This is where the directional asymmetry lives.
- **Directional message passing** in the GAT: each layer maintains incoming-edge and outgoing-edge aggregations separately.
- **InfoNCE contrastive loss** with hard negatives.

---

## Requirements

```bash
pip install torch torch-geometric sentence-transformers numpy scikit-learn tqdm
```

Tested with:
- Python 3.10+
- PyTorch 2.0+
- torch-geometric 2.4+
- sentence-transformers 2.2+

---

## Project Layout

```
cause_effect_gnn/
├── data.py          # data loading + graph construction
├── encoder.py       # text encoding (cached)
├── model.py         # GNN + scoring heads
├── train.py         # training loop with InfoNCE
├── evaluate.py      # Hits@k, MRR
├── negatives.py     # hard negative mining
└── main.py          # entry point
```

For clarity this guide presents everything as a single self-contained module. Split into files when productionizing.

---

## 1. Data Preparation

### 1.1 Input format

Your training data is a CSV/JSONL of cause-effect pairs:

```json
{"cause_id": "doc_001", "effect_id": "doc_042", "relation": "sequential_workflow"}
{"cause_id": "doc_001", "effect_id": "doc_089", "relation": "prerequisite"}
{"cause_id": "doc_017", "effect_id": "doc_201", "relation": "conceptual_dependency"}
```

Plus a separate doc store with the actual text:

```json
{"doc_id": "doc_001", "title": "Create an XDM schema for Platform Mobile SDK", "body": "..."}
```

### 1.2 Graph construction

```python
import json
import torch
from torch_geometric.data import Data
from collections import defaultdict


def load_documents(doc_path: str) -> dict:
    """Load doc_id -> {title, body} mapping."""
    docs = {}
    with open(doc_path) as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d
    return docs


def load_pairs(pairs_path: str) -> list:
    """Load list of (cause_id, effect_id, relation) tuples."""
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            p = json.loads(line)
            pairs.append((p["cause_id"], p["effect_id"], p.get("relation", "default")))
    return pairs


def build_graph(docs: dict, pairs: list):
    """
    Build a PyG Data object representing the cause-effect graph.
    
    Returns:
        data: PyG Data with edge_index, edge_weight, edge_type
        id2idx: mapping from doc_id to integer node index
        idx2id: reverse mapping
    """
    # Build node index
    doc_ids = sorted(docs.keys())
    id2idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    idx2id = {i: doc_id for doc_id, i in id2idx.items()}
    N = len(doc_ids)

    # Count edge frequencies (for SGNN-style transition weights)
    edge_counts = defaultdict(int)
    edge_relations = {}
    for c_id, t_id, rel in pairs:
        if c_id in id2idx and t_id in id2idx:
            key = (id2idx[c_id], id2idx[t_id])
            edge_counts[key] += 1
            edge_relations[key] = rel  # keep last relation if duplicated

    # Compute out-going normalization: w(t|c) = count(c,t) / sum_k count(c,k)
    out_sum = defaultdict(int)
    for (c, t), cnt in edge_counts.items():
        out_sum[c] += cnt

    edge_index = []
    edge_weight = []
    edge_type_str = []
    for (c, t), cnt in edge_counts.items():
        edge_index.append([c, t])
        edge_weight.append(cnt / out_sum[c])
        edge_type_str.append(edge_relations[(c, t)])

    # Map relation strings to integer ids
    rel_vocab = sorted(set(edge_type_str))
    rel2id = {r: i for i, r in enumerate(rel_vocab)}
    edge_type = [rel2id[r] for r in edge_type_str]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)               # [E]
    edge_type = torch.tensor(edge_type, dtype=torch.long)                    # [E]

    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        num_nodes=N,
    )
    return data, id2idx, idx2id, rel2id
```

### 1.3 Train/val/test splits

Split **by pair**, not by node. Same node can appear in multiple splits in different roles.

```python
import random


def split_pairs(pairs, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    return pairs[n_val + n_test:], pairs[:n_val], pairs[n_val:n_val + n_test]
```

**Important:** Build the graph using **only training pairs**, never val/test pairs. Otherwise you're leaking test labels into the graph the model sees at training time.

---

## 2. Text Encoder (Frozen)

Encode every document once with a sentence-transformer, cache the result. This is by far the most expensive step and only needs to happen once per doc.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


def encode_documents(docs: dict, id2idx: dict, model_name="BAAI/bge-base-en-v1.5",
                     cache_path="doc_embeddings.npy"):
    """
    Encode all documents with a pretrained sentence model.
    Returns a [N, embed_dim] tensor aligned with id2idx.
    """
    if Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}")
        emb = np.load(cache_path)
        return torch.from_numpy(emb).float()

    print(f"Encoding {len(docs)} documents with {model_name}...")
    model = SentenceTransformer(model_name)

    # Build texts in node-index order
    texts = [None] * len(id2idx)
    for doc_id, idx in id2idx.items():
        d = docs[doc_id]
        texts[idx] = f"{d.get('title','')}. {d.get('body','')}"

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True)
    np.save(cache_path, embeddings)
    return torch.from_numpy(embeddings).float()
```

**Model choice:** `BAAI/bge-base-en-v1.5` is a strong default. For higher quality (slower), use `bge-large-en-v1.5`. For multilingual data, use `intfloat/multilingual-e5-base`.

---

## 3. Model

### 3.1 Directional GAT layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DirectionalGATLayer(nn.Module):
    """
    A GAT layer with separate attention over incoming and outgoing edges.

    Aggregates each node's neighbors in two ways:
      - h_in:  attention over predecessors (edges j -> i)
      - h_out: attention over successors   (edges i -> j)
    Then combines h_self, h_in, h_out into the next layer's representation.
    """
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        # GATv2Conv aggregates from source to target along edge_index
        # We use heads with concat=False -> output is [N, out_dim]
        self.gat_in = GATv2Conv(in_dim, out_dim, heads=heads, concat=False,
                                dropout=dropout, add_self_loops=False)
        self.gat_out = GATv2Conv(in_dim, out_dim, heads=heads, concat=False,
                                 dropout=dropout, add_self_loops=False)
        self.combine = nn.Linear(in_dim + 2 * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        # edge_index: [2, E] with row 0 = source (cause), row 1 = target (effect)

        # h_in: each node receives from its predecessors
        h_in = self.gat_in(x, edge_index)

        # h_out: reverse edges so each node receives from its successors
        reversed_edges = edge_index[[1, 0], :]
        h_out = self.gat_out(x, reversed_edges)

        # Combine self + incoming + outgoing
        h = self.combine(torch.cat([x, h_in, h_out], dim=-1))
        return self.norm(F.relu(h))
```

### 3.2 Full model

```python
class CauseEffectGNN(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, num_layers=2,
                 heads=4, dropout=0.1):
        super().__init__()
        # Project frozen text embeddings into trainable space
        self.input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Stack of directional GAT layers
        self.gnn_layers = nn.ModuleList([
            DirectionalGATLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Asymmetric scoring heads
        self.W_cause = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_effect = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def encode_graph(self, x_text, edge_index):
        """
        Forward pass through the full graph.
        Returns [N, hidden_dim] embeddings for every node.
        """
        h = self.input_proj(x_text)
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
        return h

    def score(self, h_cause, h_effect):
        """
        Score pairs of (cause, effect) embeddings.
        h_cause: [B, D]
        h_effect: [B, D] or [B, K, D]
        Returns: [B] or [B, K]
        """
        phi_c = self.W_cause(h_cause)
        phi_e = self.W_effect(h_effect)
        if phi_e.dim() == 3:
            return torch.einsum("bd,bkd->bk", phi_c, phi_e)
        return (phi_c * phi_e).sum(dim=-1)
```

**Parameter count** (with text_dim=768, hidden_dim=256, 2 layers, 4 heads):
- Input projection: ~200K
- GAT layers: ~1.2M
- Scoring heads: ~130K
- **Total trainable: ~1.5M** (encoder stays frozen — ~110M)

---

## 4. Hard Negative Mining

The single biggest accuracy lever. For each positive pair (c, t⁺), find docs that are textually similar to t⁺ but are NOT actual effects of c.

```python
import faiss


class HardNegativeMiner:
    def __init__(self, doc_embeddings: torch.Tensor, pairs: list, id2idx: dict):
        """
        doc_embeddings: [N, D] frozen text embeddings
        pairs: training pairs
        """
        self.emb = doc_embeddings.numpy().astype("float32")
        self.N, self.D = self.emb.shape

        # Build set of positive (cause, effect) for fast filtering
        self.positives = defaultdict(set)
        for c, t, _ in pairs:
            if c in id2idx and t in id2idx:
                self.positives[id2idx[c]].add(id2idx[t])

        # Build FAISS index for fast nearest-neighbor lookup
        # Note: embeddings should already be L2-normalized
        self.index = faiss.IndexFlatIP(self.D)
        self.index.add(self.emb)

    def mine(self, cause_idx: int, positive_idx: int, k: int = 10) -> list:
        """
        Return k hard negatives: docs most similar to positive_idx
        that are not real effects of cause_idx.
        """
        q = self.emb[positive_idx:positive_idx + 1]
        sims, neighbor_ids = self.index.search(q, k + len(self.positives[cause_idx]) + 10)
        neighbor_ids = neighbor_ids[0]

        negatives = []
        forbidden = self.positives[cause_idx] | {positive_idx}
        for nid in neighbor_ids:
            if nid not in forbidden:
                negatives.append(int(nid))
                if len(negatives) == k:
                    break
        return negatives
```

**Why this works:** the model is forced to distinguish the *correct* effect from textually-similar-but-wrong effects. Without hard negatives it just learns "cause and effect are topically related" — which a plain bi-encoder already does. The structural signal from the GNN matters most precisely when text similarity is misleading.

---

## 5. Training Loop with InfoNCE

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CauseEffectDataset(Dataset):
    def __init__(self, pairs, id2idx, miner, k_neg=8):
        self.pairs = [(id2idx[c], id2idx[t]) for c, t, _ in pairs
                      if c in id2idx and t in id2idx]
        self.miner = miner
        self.k_neg = k_neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c, t_pos = self.pairs[idx]
        t_negs = self.miner.mine(c, t_pos, k=self.k_neg)
        return {
            "cause": c,
            "pos": t_pos,
            "negs": t_negs,
        }


def collate(batch):
    causes = torch.tensor([b["cause"] for b in batch], dtype=torch.long)
    pos = torch.tensor([b["pos"] for b in batch], dtype=torch.long)
    negs = torch.tensor([b["negs"] for b in batch], dtype=torch.long)  # [B, K]
    return causes, pos, negs


def train_epoch(model, loader, optimizer, x_text, edge_index, device,
                temperature=0.1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for causes, pos, negs in loader:
        causes = causes.to(device)
        pos = pos.to(device)
        negs = negs.to(device)

        optimizer.zero_grad()
        # Forward pass on the whole graph once per batch
        all_h = model.encode_graph(x_text, edge_index)

        h_c = all_h[causes]                   # [B, D]
        h_pos = all_h[pos]                    # [B, D]
        h_negs = all_h[negs]                  # [B, K, D]

        # Project
        phi_c = model.W_cause(h_c)            # [B, D]
        phi_pos = model.W_effect(h_pos)       # [B, D]
        phi_negs = model.W_effect(h_negs)     # [B, K, D]

        # Scores
        pos_score = (phi_c * phi_pos).sum(-1, keepdim=True)               # [B, 1]
        neg_score = torch.einsum("bd,bkd->bk", phi_c, phi_negs)           # [B, K]
        logits = torch.cat([pos_score, neg_score], dim=1) / temperature   # [B, 1+K]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches
```

**Note on the graph forward pass:** we recompute `all_h` every batch. The graph itself is fixed; only the trainable GNN weights change. For large graphs (>100K nodes) consider neighborhood sampling with `NeighborLoader` from PyG — but for typical workflow-doc graphs (<50K nodes) full-batch is fine.

---

## 6. Evaluation

For each test instance: given a cause and K candidates, rank them and compute Hits@1, Hits@3, Hits@5, and MRR.

```python
@torch.no_grad()
def evaluate(model, test_instances, x_text, edge_index, device, k_list=(1, 3, 5)):
    """
    test_instances: list of dicts with keys:
        - cause_idx: int
        - candidate_idxs: list of int (length K)
        - true_idx: int (index within candidate_idxs of the correct answer)
    """
    model.eval()
    all_h = model.encode_graph(x_text, edge_index)

    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    n = 0

    for inst in test_instances:
        c = inst["cause_idx"]
        cands = inst["candidate_idxs"]
        true_idx = inst["true_idx"]

        h_c = all_h[c].unsqueeze(0)                       # [1, D]
        h_cands = all_h[torch.tensor(cands, device=device)]  # [K, D]

        phi_c = model.W_cause(h_c)
        phi_cands = model.W_effect(h_cands)
        scores = (phi_c * phi_cands).sum(-1)              # [K]

        # rank of the true candidate
        order = scores.argsort(descending=True)
        rank = (order == true_idx).nonzero(as_tuple=True)[0].item() + 1

        for k in k_list:
            if rank <= k:
                hits[k] += 1
        mrr_sum += 1.0 / rank
        n += 1

    return {
        **{f"Hits@{k}": hits[k] / n for k in k_list},
        "MRR": mrr_sum / n,
    }
```

### 6.1 Constructing test instances

For evaluation in the SGNN-style MCNC setup (1 correct + K-1 random distractors):

```python
def make_eval_instances(test_pairs, id2idx, all_doc_idxs, k_candidates=5, seed=42):
    """
    For each test pair, create an instance with 1 correct + (k-1) random distractors.
    """
    rng = random.Random(seed)
    instances = []
    test_positives = {(id2idx[c], id2idx[t]) for c, t, _ in test_pairs
                      if c in id2idx and t in id2idx}

    for c_id, t_id, _ in test_pairs:
        if c_id not in id2idx or t_id not in id2idx:
            continue
        c, t_pos = id2idx[c_id], id2idx[t_id]

        # Sample K-1 random docs that are not the true effect
        distractors = []
        while len(distractors) < k_candidates - 1:
            d = rng.choice(all_doc_idxs)
            if d != t_pos and d != c and (c, d) not in test_positives:
                distractors.append(d)

        candidates = [t_pos] + distractors
        rng.shuffle(candidates)
        true_idx = candidates.index(t_pos)

        instances.append({
            "cause_idx": c,
            "candidate_idxs": candidates,
            "true_idx": true_idx,
        })
    return instances
```

**Tip:** also evaluate with **hard distractors** (top-K most textually-similar docs that aren't real effects) to stress-test whether the GNN is actually using structure or just text similarity. The gap between random-distractor accuracy and hard-distractor accuracy is the *real* measure of structural learning.

---

## 7. End-to-End Main

```python
def main(docs_path, pairs_path, epochs=30, batch_size=64, lr=1e-3,
         hidden_dim=256, num_layers=2, k_neg=8, device="cuda"):

    # ----- Data -----
    docs = load_documents(docs_path)
    pairs = load_pairs(pairs_path)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    # Build graph from TRAINING pairs only
    graph_data, id2idx, idx2id, rel2id = build_graph(docs, train_pairs)
    edge_index = graph_data.edge_index.to(device)

    # ----- Text embeddings -----
    x_text = encode_documents(docs, id2idx).to(device)

    # ----- Hard negative miner -----
    miner = HardNegativeMiner(x_text.cpu(), train_pairs, id2idx)

    # ----- Model -----
    model = CauseEffectGNN(
        text_dim=x_text.size(1),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = CauseEffectDataset(train_pairs, id2idx, miner, k_neg=k_neg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)

    # ----- Eval instances -----
    all_doc_idxs = list(range(len(id2idx)))
    val_instances = make_eval_instances(val_pairs, id2idx, all_doc_idxs)
    test_instances = make_eval_instances(test_pairs, id2idx, all_doc_idxs)

    # ----- Training loop -----
    best_mrr = 0.0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 x_text, edge_index, device)
        scheduler.step()

        val_metrics = evaluate(model, val_instances, x_text, edge_index, device)
        print(f"Epoch {epoch+1:3d} | loss {train_loss:.4f} | "
              f"val MRR {val_metrics['MRR']:.4f} | "
              f"H@1 {val_metrics['Hits@1']:.4f} | H@5 {val_metrics['Hits@5']:.4f}")

        if val_metrics["MRR"] > best_mrr:
            best_mrr = val_metrics["MRR"]
            torch.save(model.state_dict(), "best_model.pt")

    # ----- Final test eval -----
    model.load_state_dict(torch.load("best_model.pt"))
    test_metrics = evaluate(model, test_instances, x_text, edge_index, device)
    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main("data/docs.jsonl", "data/pairs.jsonl")
```

---

## 8. Baseline: Text-Only Bi-Encoder (For Comparison)

You should always run this baseline to measure how much the GNN actually buys you.

```python
class TextBiEncoderBaseline(nn.Module):
    """No GNN, just asymmetric scoring on frozen text embeddings."""
    def __init__(self, text_dim=768, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_cause = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_effect = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def encode_graph(self, x_text, edge_index=None):
        # Ignore graph; just project text
        return self.proj(x_text)

    def score(self, h_cause, h_effect):
        phi_c = self.W_cause(h_cause)
        phi_e = self.W_effect(h_effect)
        if phi_e.dim() == 3:
            return torch.einsum("bd,bkd->bk", phi_c, phi_e)
        return (phi_c * phi_e).sum(dim=-1)
```

Drop this into the same training loop (the `edge_index` argument is ignored). Expected pattern of results:

| Setup | Hits@1 | Hits@5 | MRR |
|---|---|---|---|
| Zero-shot text similarity (BGE cosine) | ~0.35 | ~0.70 | ~0.50 |
| Trained bi-encoder (asymmetric heads, no GNN) | ~0.45 | ~0.80 | ~0.60 |
| **GNN (this implementation)** | **~0.55** | **~0.88** | **~0.68** |

Exact numbers depend on your data. The pattern — GNN buys 5–10 points on Hits@1, less on Hits@5 — is what to expect.

---

## 9. Hyperparameter Tips

| Hyperparameter | Default | Tuning notes |
|---|---|---|
| `hidden_dim` | 256 | 128 if data is small (<10K pairs), 512 if large (>100K) |
| `num_layers` | 2 | 1 if graph is dense, 3 only if you have lots of multi-hop paths. **Avoid 4+** — over-smoothing kills performance |
| `heads` | 4 | 8 if you have GPU memory; little gain beyond 8 |
| `k_neg` | 8 | More negatives → better contrastive signal but more memory. Try 16 if it fits |
| `temperature` | 0.1 | Lower (0.05) sharpens contrast; higher (0.2) softens. 0.05–0.15 usually optimal |
| `lr` | 1e-3 | If using a fine-tunable text encoder, drop to 1e-5 for encoder params |
| `batch_size` | 64 | Larger = more in-batch negatives. Memory-limited |

**Critical bug to avoid:** leaking test edges into the graph. The graph used at training and inference must be built **only** from training pairs.

---

## 10. Extending the Architecture

### 10.1 Add relation types (R-GAT)

If your data has relation labels (`prerequisite`, `sequential_workflow`, `conceptual_dependency`), use one GAT per relation and combine:

```python
class RelationalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, heads=4):
        super().__init__()
        self.gat_per_rel = nn.ModuleList([
            DirectionalGATLayer(in_dim, out_dim, heads=heads)
            for _ in range(num_relations)
        ])
        self.combine = nn.Linear(num_relations * out_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        outputs = []
        for r, layer in enumerate(self.gat_per_rel):
            mask = edge_type == r
            outputs.append(layer(x, edge_index[:, mask]))
        return self.combine(torch.cat(outputs, dim=-1))
```

### 10.2 Add chat / session context

If predictions depend on chat history, encode the session path with an LSTM or self-attention over the visited docs' embeddings, then concatenate with the cause embedding before the scoring head.

### 10.3 Inductive setting for unseen docs

If test docs are not in the training graph, just encode them with the text encoder and pass through `input_proj`. Skip the GNN aggregation (no neighbors to aggregate from). The model degrades to a bi-encoder for cold-start docs but works fine for in-graph docs.

---

## 11. References

1. **Li, Z., Ding, X., & Liu, T. (2018).** *Constructing Narrative Event Evolutionary Graph for Script Event Prediction.* IJCAI. — The SGNN paper this implementation is based on.
2. **Schlichtkrull, M., et al. (2018).** *Modeling Relational Data with Graph Convolutional Networks.* ESWC. — R-GCN; basis for the relation-aware extension.
3. **Veličković, P., et al. (2018).** *Graph Attention Networks.* ICLR. — Original GAT.
4. **Brody, S., Alon, U., & Yahav, E. (2022).** *How Attentive are Graph Attention Networks?* ICLR. — GATv2, the better attention variant used here.
5. **Karpukhin, V., et al. (2020).** *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. — The strong bi-encoder baseline.
6. **Yasunaga, M., et al. (2022).** *LinkBERT: Pretraining Language Models with Document Links.* ACL. — Closest related work for doc-link tasks.
7. **Hamilton, W., Ying, R., & Leskovec, J. (2017).** *Inductive Representation Learning on Large Graphs.* NeurIPS. — GraphSAGE; useful for inductive variants.

---

## Quick Start Checklist

- [ ] Prepare `docs.jsonl` (doc_id, title, body) and `pairs.jsonl` (cause_id, effect_id, relation)
- [ ] Run `encode_documents()` once to cache text embeddings (~10 min for 50K docs)
- [ ] Train baseline bi-encoder first — establishes the floor
- [ ] Train GNN model and compare on the same eval set
- [ ] Inspect failure cases: are they hard-negative confusions (GNN didn't help enough) or genuine ambiguity (no model can fix)?
- [ ] Try the relational variant if your data has labeled edge types
- [ ] Report Hits@1, Hits@3, Hits@5, MRR — both with random and hard distractors
