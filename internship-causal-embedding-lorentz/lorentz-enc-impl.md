# Lorentz-Enc — Implementation Specification

A causal-aware text encoder built on BGE-M3 with explicit space/time decomposition, Lorentzian-MDS pretraining, semantically-similar hard-negative mining, and a Cawai-style frozen-encoder regulariser. Untyped variant — works from `(anchor, positive, negative)` triples only.

This document is structured for Claude Code or any agentic coding assistant to implement directly. Code is in PyTorch. Each module is self-contained.

---

## 1. What the model does

**Input at inference:** a passage of text → 1024-dim vector split into:
- `space ∈ R^1004` (L2-normalised) — topical/semantic similarity
- `time ∈ R^20` (layer-normalised, unbounded) — causal precedence

**Score for a pair `(a, b)`:**
```
sim(a, b) = α · cos(space(a), space(b))
          + β · (1/20) · Σ_{k=1..20} σ(τ · (time(b)_k − time(a)_k))
```
- First term is symmetric (topical similarity).
- Second term is asymmetric — high when `b` is "after" `a` in every time dim. This is the Lorentzian time-ordering signal.
- `σ` is sigmoid, `τ = 10` (SteepSigmoid).
- Defaults: `α = 1.0, β = 0.5`.

**Training data assumed:** a JSONL file where each line is:
```json
{"anchor": "...", "positive": "...", "negatives": ["...", "...", ...]}
```
Negatives are pre-mined hard negatives (semantically similar to anchor, not a causal positive). The pipeline below will *also* mine fresh negatives iteratively.

---

## 2. Project structure

```
lorentz_enc/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── lorentz_enc/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch Dataset for (anchor, pos, negs) JSONL
│   │   ├── collator.py           # tokenisation + padding
│   │   ├── graph.py              # build DAG from positives for MDS
│   │   └── hard_neg_miner.py     # ANCE-style iterative mining with FAISS
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py            # BGE-M3 backbone + space/time projection heads
│   │   ├── scoring.py            # cos + SteepSigmoid scoring
│   │   └── losses.py             # InfoNCE, ordering, Cawai-reg, MDS distillation
│   ├── pretrain/
│   │   ├── __init__.py
│   │   ├── longest_path.py       # timelike separation via longest directed path
│   │   ├── spacelike.py          # naive spacelike distance (Clough & Evans)
│   │   └── lorentzian_mds.py     # generalised MDS with signature tensor G
│   ├── train/
│   │   ├── __init__.py
│   │   ├── stage1_mds.py         # offline: build graph, run Lorentzian MDS
│   │   ├── stage2_distill.py     # distil encoder to MDS coordinates
│   │   ├── stage3_contrastive.py # InfoNCE + ordering + Cawai-reg
│   │   └── stage5_final.py       # final ANCE-Tele pass
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── retrieval.py          # MRR, nDCG, Recall@k
│   │   ├── pair_classification.py
│   │   ├── clough_evans_auc.py   # speed-of-light c-sweep AUC
│   │   └── diagnostics.py        # directionality, distortion, ‖t‖/‖s‖ ratios
│   └── utils/
│       ├── __init__.py
│       ├── faiss_index.py
│       ├── llm_negatives.py      # optional: LLM-generated synthetic negatives
│       └── logging.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_stage1_mds.sh
│   ├── run_stage2_distill.sh
│   ├── run_stage3_contrastive.sh
│   ├── run_stage5_final.sh
│   ├── eval_all.sh
│   └── ablate.sh
└── tests/
    ├── test_dataset.py
    ├── test_encoder.py
    ├── test_lorentzian_mds.py
    ├── test_losses.py
    └── test_scoring.py
```

---

## 3. Dependencies (`requirements.txt`)

```
torch>=2.2.0
transformers>=4.40.0
sentence-transformers>=2.7.0
FlagEmbedding>=1.2.0
faiss-cpu>=1.7.4          # or faiss-gpu if available
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
networkx>=3.1
tqdm>=4.66.0
pyyaml>=6.0
einops>=0.7.0
accelerate>=0.27.0
wandb>=0.16.0             # optional, for logging
```

---

## 4. Configuration (`configs/default.yaml`)

```yaml
# Backbone
backbone:
  name: "BAAI/bge-m3"
  hidden_dim: 1024
  max_length: 512
  pooling: "cls"              # "cls" | "mean"

# Head architecture
heads:
  space_dim: 1004
  time_dim: 20
  space_hidden: 1024          # MLP hidden dim for space head
  time_hidden: 512            # MLP hidden dim for time head
  dropout: 0.1

# Scoring
scoring:
  alpha: 1.0                  # weight on space cosine
  beta: 0.5                   # weight on time sigmoid term
  tau: 10.0                   # SteepSigmoid steepness

# Lorentzian MDS pretraining (Stage 1)
mds:
  n_negative_eigvals: 20      # number of time dims taken from negative eigenvalues
  max_spacelike_distance: null # if null, set to longest path length in graph
  use_relation_types: false   # we have none
  # numerical stabilisation
  jitter: 1.0e-6

# Training (Stage 2: distillation)
stage2:
  batch_size: 64
  epochs: 1
  lr: 1.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.05
  grad_clip: 1.0
  lambda_dist: 1.0
  lambda_nce: 0.1             # weak InfoNCE
  
# Training (Stage 3: contrastive + ordering + Cawai-reg)
stage3:
  batch_size: 128
  epochs: 3
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.05
  grad_clip: 1.0
  lambda_nce: 1.0
  lambda_ord: 0.5
  lambda_reg: 1.0             # Cawai β
  nce_temperature: 0.05
  nce_margin: 0.02
  ord_margin: 0.5
  ord_slack: 0.1
  hard_neg_refresh_steps: 5000
  hard_negs_per_anchor: 7

# Hard negative mining
mining:
  top_k_candidates: 200
  rank_buckets: [[0, 10], [10, 50], [50, 200]]
  bucket_weights: [0.3, 0.4, 0.3]
  jaccard_dedup_threshold: 0.7
  use_bm25_pool: true
  use_frozen_bge_pool: true

# Stage 5: final
stage5:
  batch_size: 128
  epochs: 1
  lr: 5.0e-6
  use_teleportation_negatives: true

# Eval
eval:
  retrieval_pool_size: 50000  # or "full" for entire corpus
  metrics: ["mrr@10", "ndcg@10", "recall@10", "recall@100"]
  clough_evans:
    c_values: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    enabled: true

# Paths
paths:
  data_dir: "./data"
  train_jsonl: "./data/train.jsonl"
  val_jsonl: "./data/val.jsonl"
  test_jsonl: "./data/test.jsonl"
  unlabeled_corpus: "./data/corpus.jsonl"
  mds_coords: "./checkpoints/mds_coords.pt"
  stage2_ckpt: "./checkpoints/stage2/"
  stage3_ckpt: "./checkpoints/stage3/"
  stage5_ckpt: "./checkpoints/stage5/"
  frozen_bge_ckpt: "BAAI/bge-m3"   # huggingface id, for L_reg

# Hardware
hardware:
  device: "cuda"
  mixed_precision: "bf16"
  num_workers: 4

# Logging
logging:
  wandb_project: "lorentz-enc"
  log_every: 50
  eval_every: 1000
  save_every: 5000
```

---

## 5. Data format and loader

### `lorentz_enc/data/dataset.py`

```python
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class CausalPairDataset(Dataset):
    """
    Each item: (anchor_text, positive_text, [negative_texts]).
    No relation labels.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512,
                 max_negatives: int = 7):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                # anchor, positive, negatives (list of strings)
                assert "anchor" in ex and "positive" in ex
                if "negatives" not in ex:
                    ex["negatives"] = []
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_negatives = max_negatives

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict:
        ex = self.examples[idx]
        negs = ex["negatives"][:self.max_negatives]
        # pad with empty negatives if fewer; collator handles in-batch alternative
        return {
            "anchor": ex["anchor"],
            "positive": ex["positive"],
            "negatives": negs,
            "anchor_id": ex.get("anchor_id", idx),
            "positive_id": ex.get("positive_id", -1),
        }
```

### `lorentz_enc/data/collator.py`

```python
import torch
from typing import List, Dict


class TripletCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tok = tokenizer
        self.max_length = max_length

    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        N = max(len(ex["negatives"]) for ex in batch)
        # Pad negatives with empty strings; mask out later
        neg_mask = torch.zeros(B, N, dtype=torch.bool)
        anchors, positives, all_negatives = [], [], []
        for i, ex in enumerate(batch):
            anchors.append(ex["anchor"])
            positives.append(ex["positive"])
            negs = list(ex["negatives"])
            for j in range(N):
                if j < len(negs):
                    all_negatives.append(negs[j])
                    neg_mask[i, j] = True
                else:
                    all_negatives.append("")  # padded
                    neg_mask[i, j] = False
        return {
            "anchor": self._encode(anchors),
            "positive": self._encode(positives),
            "negatives": self._encode(all_negatives),  # (B*N) flat
            "neg_mask": neg_mask,                       # (B, N)
            "B": B, "N": N,
            "anchor_ids": torch.tensor([ex.get("anchor_id", i) for i, ex in enumerate(batch)]),
            "positive_ids": torch.tensor([ex.get("positive_id", -1) for ex in batch]),
        }
```

---

## 6. Model

### `lorentz_enc/model/encoder.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple


class LorentzEncoder(nn.Module):
    """
    BGE-M3 backbone + two projection heads:
      - space head:  R^hidden -> R^space_dim, L2-normalised
      - time head:   R^hidden -> R^time_dim,  layer-normalised (unbounded)
    """

    def __init__(
        self,
        backbone_name: str = "BAAI/bge-m3",
        space_dim: int = 1004,
        time_dim: int = 20,
        space_hidden: int = 1024,
        time_hidden: int = 512,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.pooling = pooling

        # Space head: 2-layer MLP with skip + L2 norm
        self.space_head = nn.Sequential(
            nn.Linear(hidden, space_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(space_hidden, space_dim),
        )

        # Time head: 2-layer MLP, layer-normalised (not L2 — magnitude carries info)
        self.time_head = nn.Sequential(
            nn.Linear(hidden, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, time_dim),
            nn.LayerNorm(time_dim),
        )

        self.space_dim = space_dim
        self.time_dim = time_dim

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(self.pooling)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self._pool(out.last_hidden_state, attention_mask)
        space = F.normalize(self.space_head(h), dim=-1, p=2)
        time  = self.time_head(h)
        return space, time

    @torch.no_grad()
    def encode(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        self.eval()
        s, t = self.forward(input_ids, attention_mask)
        return {"space": s, "time": t}
```

### `lorentz_enc/model/scoring.py`

```python
import torch
import torch.nn.functional as F


def steep_sigmoid_mean(t_a: torch.Tensor, t_b: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    """
    Asymmetric time-precedence score: high when t_b is 'after' t_a in every dim.
    Returns a scalar in (0, 1) per pair (mean over time dimensions).

    t_a, t_b: (..., time_dim)
    """
    diff = t_b - t_a                          # (..., D_t)
    return torch.sigmoid(tau * diff).mean(dim=-1)


def causal_similarity(
    space_a: torch.Tensor, time_a: torch.Tensor,
    space_b: torch.Tensor, time_b: torch.Tensor,
    alpha: float = 1.0, beta: float = 0.5, tau: float = 10.0,
) -> torch.Tensor:
    """
    sim(a,b) = α · cos(space) + β · SteepSigmoid_mean(time)
    Inputs are (B, D_s) and (B, D_t) — assumed already paired.
    """
    cos = (space_a * space_b).sum(dim=-1)     # since L2-normalised
    time_score = steep_sigmoid_mean(time_a, time_b, tau=tau)
    return alpha * cos + beta * time_score


def causal_similarity_matrix(
    space_q: torch.Tensor, time_q: torch.Tensor,
    space_k: torch.Tensor, time_k: torch.Tensor,
    alpha: float = 1.0, beta: float = 0.5, tau: float = 10.0,
) -> torch.Tensor:
    """
    All-pairs causal similarity. space_q,time_q: (Q, *). space_k,time_k: (K, *).
    Returns (Q, K).
    """
    cos = space_q @ space_k.t()                                       # (Q, K)
    # time: pairwise diffs in (Q, K, D_t)
    diff = time_k.unsqueeze(0) - time_q.unsqueeze(1)                  # (Q, K, D_t)
    time_score = torch.sigmoid(tau * diff).mean(dim=-1)               # (Q, K)
    return alpha * cos + beta * time_score
```

### `lorentz_enc/model/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scoring import causal_similarity_matrix, steep_sigmoid_mean


# ---------- 1. InfoNCE retrieval loss ----------

def info_nce_loss(
    space_a, time_a, space_pos, time_pos, space_neg, time_neg,
    neg_mask, alpha=1.0, beta=0.5, tau_score=10.0,
    tau_nce=0.05, margin=0.02,
):
    """
    Standard InfoNCE with additive margin on positive (SimKGC convention).

    space_a/time_a: (B, *)
    space_pos/time_pos: (B, *)
    space_neg/time_neg: (B*N, *)
    neg_mask: (B, N) bool
    """
    B = space_a.size(0)
    N = neg_mask.size(1)
    space_neg = space_neg.view(B, N, -1)
    time_neg = time_neg.view(B, N, -1)

    # Positive score (B,)
    pos_score = (
        alpha * (space_a * space_pos).sum(-1)
        + beta * steep_sigmoid_mean(time_a, time_pos, tau=tau_score)
    )

    # Negative scores (B, N)
    diff_t = time_neg - time_a.unsqueeze(1)
    neg_time = torch.sigmoid(tau_score * diff_t).mean(-1)
    neg_cos  = (space_a.unsqueeze(1) * space_neg).sum(-1)
    neg_score = alpha * neg_cos + beta * neg_time

    # In-batch negatives: all other positives in batch as additional negs
    # Build (B, B) score; mask out diagonal (self-positive)
    in_batch = causal_similarity_matrix(
        space_a, time_a, space_pos, time_pos,
        alpha=alpha, beta=beta, tau=tau_score,
    )                                                                   # (B, B)
    diag = torch.eye(B, dtype=torch.bool, device=space_a.device)
    in_batch = in_batch.masked_fill(diag, float("-inf"))                # remove self

    # Concatenate: positive | mined negatives | in-batch negatives
    pos_logit = (pos_score - margin) / tau_nce                          # (B,)
    neg_logit = neg_score / tau_nce                                     # (B, N)
    neg_logit = neg_logit.masked_fill(~neg_mask, float("-inf"))         # mask padded
    in_batch_logit = in_batch / tau_nce                                 # (B, B)

    logits = torch.cat(
        [pos_logit.unsqueeze(1), neg_logit, in_batch_logit], dim=1,
    )                                                                    # (B, 1+N+B)
    # Target is index 0 (the positive)
    target = torch.zeros(B, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


# ---------- 2. Asymmetric ordering loss on time dims ----------

def ordering_loss(
    time_a, time_pos, time_neg, neg_mask,
    margin=0.5, slack=0.1,
):
    """
    Push t_pos to be > t_a in every time dim by `margin`.
    Push t_neg to NOT be > t_a — i.e. require t_neg_k - t_a_k < -slack in at least one dim
    (we enforce dim-wise via hinge: penalise when t_neg_k > t_a_k - slack).
    """
    B, D = time_a.shape
    N = neg_mask.size(1)

    # Positive: want t_pos_k - t_a_k > margin
    diff_pos = time_pos - time_a                          # (B, D)
    loss_pos = F.relu(margin - diff_pos).mean(-1)         # (B,)

    # Negative: want t_neg_k - t_a_k < -slack for at least one k
    # Hinge per-dim: penalise where t_neg_k > t_a_k - slack
    diff_neg = time_neg.view(B, N, D) - time_a.unsqueeze(1)  # (B, N, D)
    # Strong form: average over dims (encourage all dims to be "before" anchor)
    per_neg = F.relu(slack + diff_neg).mean(-1)           # (B, N)
    per_neg = per_neg.masked_fill(~neg_mask, 0.0)
    n_valid = neg_mask.sum(-1).clamp(min=1).float()
    loss_neg = per_neg.sum(-1) / n_valid                  # (B,)

    return (loss_pos + loss_neg).mean()


# ---------- 3. Cawai-style frozen-encoder semantic regulariser ----------

def cawai_regulariser(
    space_a: torch.Tensor, space_frozen_a: torch.Tensor,
):
    """
    In-batch InfoNCE pulling each anchor's space rep toward its frozen-encoder twin
    and away from other anchors' frozen reps. Both inputs are L2-normalised.

    space_a, space_frozen_a: (B, D_s)
    """
    logits = space_a @ space_frozen_a.t()                # (B, B), already in [-1, 1]
    target = torch.arange(space_a.size(0), device=space_a.device)
    return F.cross_entropy(logits / 0.05, target)


# ---------- 4. Lorentzian MDS coordinate distillation ----------

def mds_distillation_loss(
    space_pred: torch.Tensor, time_pred: torch.Tensor,
    space_target: torch.Tensor, time_target: torch.Tensor,
):
    """
    MSE between predicted (space, time) and Lorentzian-MDS-derived target coordinates.
    Both predicted and target spaces are aligned; see lorentzian_mds.py for Procrustes.
    """
    return (
        F.mse_loss(space_pred, space_target) +
        F.mse_loss(time_pred,  time_target)
    )


# ---------- 5. Combined losses for stage 3 ----------

class Stage3Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        space_a, time_a, space_pos, time_pos, space_neg, time_neg,
        space_frozen_a, neg_mask,
    ):
        l_nce = info_nce_loss(
            space_a, time_a, space_pos, time_pos, space_neg, time_neg,
            neg_mask,
            alpha=self.cfg.scoring.alpha, beta=self.cfg.scoring.beta,
            tau_score=self.cfg.scoring.tau,
            tau_nce=self.cfg.stage3.nce_temperature, margin=self.cfg.stage3.nce_margin,
        )
        l_ord = ordering_loss(
            time_a, time_pos, time_neg, neg_mask,
            margin=self.cfg.stage3.ord_margin, slack=self.cfg.stage3.ord_slack,
        )
        l_reg = cawai_regulariser(space_a, space_frozen_a)
        total = (
            self.cfg.stage3.lambda_nce * l_nce
            + self.cfg.stage3.lambda_ord * l_ord
            + self.cfg.stage3.lambda_reg * l_reg
        )
        return total, {"nce": l_nce.item(), "ord": l_ord.item(), "reg": l_reg.item()}
```

---

## 7. Lorentzian-MDS pretraining (Stage 1)

### `lorentz_enc/data/graph.py`

```python
import networkx as nx
from typing import Dict, List, Tuple
import json


def build_dag_from_pairs(jsonl_path: str) -> nx.DiGraph:
    """
    Build a DAG: each (anchor -> positive) is a directed edge.
    Nodes are unique passage texts (use anchor_id / positive_id as node ids).
    """
    G = nx.DiGraph()
    text_of = {}  # node_id -> text

    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            a_id = ex.get("anchor_id", ex["anchor"])
            p_id = ex.get("positive_id", ex["positive"])
            G.add_node(a_id); G.add_node(p_id)
            G.add_edge(a_id, p_id, weight=1.0)
            text_of[a_id] = ex["anchor"]
            text_of[p_id] = ex["positive"]

    # If there are cycles (shouldn't be — but safety), break by removing back edges
    if not nx.is_directed_acyclic_graph(G):
        # Remove edges in any cycle (pick lower-rank deletion)
        while not nx.is_directed_acyclic_graph(G):
            try:
                cyc = nx.find_cycle(G, orientation="original")
                G.remove_edge(cyc[0][0], cyc[0][1])
            except nx.NetworkXNoCycle:
                break
    return G, text_of
```

### `lorentz_enc/pretrain/longest_path.py`

```python
import networkx as nx
import numpy as np
from typing import Dict, Tuple


def all_pairs_longest_path(G: nx.DiGraph) -> Dict[Tuple, int]:
    """
    Compute longest directed path length between every ordered timelike pair.
    Returns dict[(u, v)] -> length (number of edges).
    Only populated for u that is an ancestor of v (i.e. timelike-separated, u before v).

    For DAGs, the longest path between two nodes can be computed via topological sort.
    O(V * (V + E)) overall. Switch to dynamic-programming over topological order for speed.
    """
    topo = list(nx.topological_sort(G))
    idx = {n: i for i, n in enumerate(topo)}
    N = len(topo)
    # dp[i][j] = longest path from topo[i] to topo[j] (-inf if no path)
    NEG = -10**9
    dp = np.full((N, N), NEG, dtype=np.int32)
    for i in range(N):
        dp[i, i] = 0
    for u_i, u in enumerate(topo):
        for v in G.successors(u):
            v_i = idx[v]
            if dp[u_i, v_i] < 1:
                dp[u_i, v_i] = 1
    # propagate
    for k_i in range(N):
        for i in range(N):
            if dp[i, k_i] == NEG:
                continue
            for j in range(N):
                if dp[k_i, j] == NEG:
                    continue
                cand = dp[i, k_i] + dp[k_i, j]
                if cand > dp[i, j]:
                    dp[i, j] = cand
    out = {}
    for i, u in enumerate(topo):
        for j, v in enumerate(topo):
            if i != j and dp[i, j] > NEG and dp[i, j] >= 1:
                out[(u, v)] = int(dp[i, j])
    return out
```

> NOTE: This is `O(V³)` naive. For >10k nodes use the Floyd-Warshall variant with negative-weight max-plus semiring, or compute longest paths only from labeled anchors using a per-source DP in `O(V·(V+E))`.

### `lorentz_enc/pretrain/spacelike.py`

```python
import networkx as nx
from typing import Dict, Tuple


def naive_spacelike_distance(
    G: nx.DiGraph,
    longest_path: Dict[Tuple, int],
    max_distance: int,
) -> Dict[Tuple, int]:
    """
    Clough & Evans 'naive spatial distance' for spacelike pairs.

    For each unordered pair (i, j) with no directed path either way:
      - find all (k, l) such that:
          * k is in the future of both i and j (longest_path[(i,k)] > 0 AND longest_path[(j,k)] > 0)
          * l is in the past of both         (longest_path[(l,i)] > 0 AND longest_path[(l,j)] > 0)
      - pick (k*, l*) minimising longest_path[(l, k)]
      - assign N_ij^2 = longest_path[(l*, k*)]^2  (as a positive squared distance)
      - if no such (k, l) exists, use max_distance.
    """
    nodes = list(G.nodes())
    # Pre-compute futures and pasts (sets of descendants/ancestors)
    futures = {n: set(nx.descendants(G, n)) for n in nodes}
    pasts   = {n: set(nx.ancestors(G, n))   for n in nodes}

    out = {}
    for ii, i in enumerate(nodes):
        for j in nodes[ii+1:]:
            if (i, j) in longest_path or (j, i) in longest_path:
                continue                           # not spacelike
            common_future = futures[i] & futures[j]
            common_past   = pasts[i]   & pasts[j]
            best = None
            for k in common_future:
                for l in common_past:
                    d = longest_path.get((l, k))
                    if d is None: continue
                    if best is None or d < best:
                        best = d
            if best is None:
                best = max_distance
            out[(i, j)] = best
            out[(j, i)] = best
    return out
```

### `lorentz_enc/pretrain/lorentzian_mds.py`

```python
import numpy as np
import torch
from typing import Dict, Tuple, List
import networkx as nx
from .longest_path import all_pairs_longest_path
from .spacelike import naive_spacelike_distance


def lorentzian_mds(
    G: nx.DiGraph,
    space_dim: int = 1004,
    time_dim: int = 20,
    max_spacelike_distance: int | None = None,
    jitter: float = 1e-6,
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Generalised MDS for Lorentzian signature.

    Returns:
        space_coords: dict[node_id] -> np.ndarray(space_dim,)
        time_coords:  dict[node_id] -> np.ndarray(time_dim,)
        eigvals: full eigenvalue spectrum (sorted descending)
    """
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    print(f"[MDS] computing longest paths for {N} nodes…")
    lp = all_pairs_longest_path(G)

    if max_spacelike_distance is None:
        max_d = max(lp.values()) if lp else 1
    else:
        max_d = max_spacelike_distance

    print(f"[MDS] computing naive spacelike distances…")
    sl = naive_spacelike_distance(G, lp, max_distance=max_d)

    # Build separation matrix M (signed squared)
    print(f"[MDS] building separation matrix ({N}x{N})…")
    M = np.zeros((N, N), dtype=np.float64)
    for (u, v), d in lp.items():
        i, j = idx[u], idx[v]
        M[i, j] = -float(d) ** 2                # timelike: negative
        M[j, i] = -float(d) ** 2                # symmetric storage
    for (u, v), d in sl.items():
        i, j = idx[u], idx[v]
        M[i, j] = +float(d) ** 2                # spacelike: positive

    # Double-centre
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ M @ J
    # symmetrise + jitter
    B = 0.5 * (B + B.T) + jitter * np.eye(N)

    print(f"[MDS] eigendecomposition…")
    eigvals, eigvecs = np.linalg.eigh(B)         # ascending order
    # Sort by magnitude descending
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Separate negative (time) and positive (space) eigenvalues
    neg_mask = eigvals < 0
    pos_mask = eigvals > 0
    neg_vals = eigvals[neg_mask]
    neg_vecs = eigvecs[:, neg_mask]
    pos_vals = eigvals[pos_mask]
    pos_vecs = eigvecs[:, pos_mask]

    # Take top `time_dim` most-negative
    n_time = min(time_dim, neg_vals.shape[0])
    time_block = neg_vecs[:, :n_time] * np.sqrt(np.abs(neg_vals[:n_time]))   # (N, n_time)
    if n_time < time_dim:
        pad = np.zeros((N, time_dim - n_time))
        time_block = np.concatenate([time_block, pad], axis=1)

    # Take top `space_dim` largest-positive
    n_space = min(space_dim, pos_vals.shape[0])
    space_block = pos_vecs[:, :n_space] * np.sqrt(pos_vals[:n_space])         # (N, n_space)
    if n_space < space_dim:
        pad = np.zeros((N, space_dim - n_space))
        space_block = np.concatenate([space_block, pad], axis=1)
    # L2-normalise space (since the encoder space head is L2-normalised)
    norms = np.linalg.norm(space_block, axis=1, keepdims=True).clip(min=1e-9)
    space_block = space_block / norms

    space_coords = {n: space_block[idx[n]] for n in nodes}
    time_coords  = {n: time_block[idx[n]]  for n in nodes}
    return space_coords, time_coords, eigvals
```

### `lorentz_enc/train/stage1_mds.py`

```python
import torch
import yaml
import os
from ..data.graph import build_dag_from_pairs
from ..pretrain.lorentzian_mds import lorentzian_mds


def run_stage1(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    G, text_of = build_dag_from_pairs(cfg["paths"]["train_jsonl"])

    space_coords, time_coords, eigvals = lorentzian_mds(
        G,
        space_dim=cfg["heads"]["space_dim"],
        time_dim=cfg["heads"]["time_dim"],
        max_spacelike_distance=cfg["mds"].get("max_spacelike_distance"),
        jitter=cfg["mds"]["jitter"],
    )

    # Save
    os.makedirs(os.path.dirname(cfg["paths"]["mds_coords"]), exist_ok=True)
    torch.save({
        "space": {k: torch.tensor(v, dtype=torch.float32) for k, v in space_coords.items()},
        "time":  {k: torch.tensor(v, dtype=torch.float32) for k, v in time_coords.items()},
        "eigvals": torch.tensor(eigvals, dtype=torch.float32),
        "text_of": text_of,
    }, cfg["paths"]["mds_coords"])
    print(f"[Stage 1] saved MDS coordinates to {cfg['paths']['mds_coords']}")

    # Sanity stats
    n_neg = (eigvals < 0).sum()
    n_pos = (eigvals > 0).sum()
    print(f"[Stage 1] eigenvalue spectrum: {n_neg} negative, {n_pos} positive")
    print(f"[Stage 1] top-5 negative: {sorted(eigvals[eigvals<0])[:5]}")
    print(f"[Stage 1] top-5 positive: {sorted(eigvals[eigvals>0], reverse=True)[:5]}")
```

---

## 8. Stage 2 — coordinate distillation

### `lorentz_enc/train/stage2_distill.py`

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
import yaml, os
from ..model.encoder import LorentzEncoder
from ..model.losses import mds_distillation_loss, info_nce_loss
from ..data.dataset import CausalPairDataset
from ..data.collator import TripletCollator


def run_stage2(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    accel = Accelerator(mixed_precision=cfg["hardware"]["mixed_precision"])

    # Load tokenizer + model
    tok = AutoTokenizer.from_pretrained(cfg["backbone"]["name"])
    model = LorentzEncoder(
        backbone_name=cfg["backbone"]["name"],
        space_dim=cfg["heads"]["space_dim"],
        time_dim=cfg["heads"]["time_dim"],
        space_hidden=cfg["heads"]["space_hidden"],
        time_hidden=cfg["heads"]["time_hidden"],
        dropout=cfg["heads"]["dropout"],
        pooling=cfg["backbone"]["pooling"],
    )

    # Load MDS targets
    mds = torch.load(cfg["paths"]["mds_coords"])
    space_target = mds["space"]   # dict node_id -> (D_s,)
    time_target  = mds["time"]    # dict node_id -> (D_t,)

    # Dataset
    ds = CausalPairDataset(cfg["paths"]["train_jsonl"], tok, max_length=cfg["backbone"]["max_length"])
    coll = TripletCollator(tok, max_length=cfg["backbone"]["max_length"])
    dl = DataLoader(ds, batch_size=cfg["stage2"]["batch_size"], shuffle=True,
                    collate_fn=coll, num_workers=cfg["hardware"]["num_workers"])

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["stage2"]["lr"],
                              weight_decay=cfg["stage2"]["weight_decay"])
    total_steps = len(dl) * cfg["stage2"]["epochs"]
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * cfg["stage2"]["warmup_ratio"]), total_steps,
    )

    model, optim, dl, sched = accel.prepare(model, optim, dl, sched)

    step = 0
    for epoch in range(cfg["stage2"]["epochs"]):
        for batch in dl:
            anchor_ids = batch["anchor_ids"]
            positive_ids = batch["positive_ids"]

            # Forward
            s_a, t_a = model(**batch["anchor"])
            s_p, t_p = model(**batch["positive"])

            # Collect MDS targets (skip examples whose ids aren't in the MDS dict)
            valid = []
            t_s_a_list, t_t_a_list, t_s_p_list, t_t_p_list = [], [], [], []
            for i, (aid, pid) in enumerate(zip(anchor_ids.tolist(), positive_ids.tolist())):
                if aid in space_target and pid in space_target:
                    t_s_a_list.append(space_target[aid])
                    t_t_a_list.append(time_target[aid])
                    t_s_p_list.append(space_target[pid])
                    t_t_p_list.append(time_target[pid])
                    valid.append(i)
            if not valid:
                continue
            idx = torch.tensor(valid, device=s_a.device)
            ts_a = torch.stack(t_s_a_list).to(s_a.device)
            tt_a = torch.stack(t_t_a_list).to(s_a.device)
            ts_p = torch.stack(t_s_p_list).to(s_a.device)
            tt_p = torch.stack(t_t_p_list).to(s_a.device)

            l_dist = (
                mds_distillation_loss(s_a[idx], t_a[idx], ts_a, tt_a)
              + mds_distillation_loss(s_p[idx], t_p[idx], ts_p, tt_p)
            )

            # Weak InfoNCE for stability
            s_n, t_n = model(**batch["negatives"])
            l_nce = info_nce_loss(
                s_a, t_a, s_p, t_p, s_n, t_n, batch["neg_mask"],
                alpha=cfg["scoring"]["alpha"], beta=cfg["scoring"]["beta"],
                tau_score=cfg["scoring"]["tau"], tau_nce=0.05, margin=0.02,
            )

            loss = cfg["stage2"]["lambda_dist"] * l_dist + cfg["stage2"]["lambda_nce"] * l_nce

            optim.zero_grad()
            accel.backward(loss)
            accel.clip_grad_norm_(model.parameters(), cfg["stage2"]["grad_clip"])
            optim.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[Stage 2] step {step} loss={loss.item():.4f} "
                      f"(dist={l_dist.item():.4f} nce={l_nce.item():.4f})")

    accel.save_state(cfg["paths"]["stage2_ckpt"])
    print(f"[Stage 2] saved checkpoint to {cfg['paths']['stage2_ckpt']}")
```

---

## 9. Stage 3 — contrastive + ordering + Cawai-reg

### `lorentz_enc/train/stage3_contrastive.py`

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
import yaml, os
from copy import deepcopy
from ..model.encoder import LorentzEncoder
from ..model.losses import Stage3Loss
from ..data.dataset import CausalPairDataset
from ..data.collator import TripletCollator
from ..data.hard_neg_miner import HardNegativeMiner


def run_stage3(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    accel = Accelerator(mixed_precision=cfg["hardware"]["mixed_precision"])

    tok = AutoTokenizer.from_pretrained(cfg["backbone"]["name"])
    model = LorentzEncoder(
        backbone_name=cfg["backbone"]["name"],
        space_dim=cfg["heads"]["space_dim"],
        time_dim=cfg["heads"]["time_dim"],
        space_hidden=cfg["heads"]["space_hidden"],
        time_hidden=cfg["heads"]["time_hidden"],
        dropout=cfg["heads"]["dropout"],
        pooling=cfg["backbone"]["pooling"],
    )
    # Load stage 2 weights
    if os.path.exists(cfg["paths"]["stage2_ckpt"]):
        model.load_state_dict(torch.load(os.path.join(cfg["paths"]["stage2_ckpt"], "model.pt")))

    # Frozen encoder for Cawai regulariser (just the base BGE-M3 with NO heads — use mean pooling on backbone)
    frozen = LorentzEncoder(
        backbone_name=cfg["paths"]["frozen_bge_ckpt"],
        space_dim=cfg["heads"]["space_dim"],
        time_dim=cfg["heads"]["time_dim"],
        space_hidden=cfg["heads"]["space_hidden"],
        time_hidden=cfg["heads"]["time_hidden"],
        dropout=0.0,
        pooling=cfg["backbone"]["pooling"],
    )
    # IMPORTANT: re-initialise frozen's space head with identity-like projection
    # so it behaves like off-the-shelf BGE-M3 (use the backbone CLS directly, L2-normed).
    # Simplest: make space_head an identity-truncate to space_dim — we provide a helper:
    frozen = _freeze_baseline(frozen, cfg)
    for p in frozen.parameters():
        p.requires_grad = False
    frozen.eval()

    # Loss
    loss_fn = Stage3Loss(cfg=_cfg_to_obj(cfg))

    # Data
    ds = CausalPairDataset(cfg["paths"]["train_jsonl"], tok, max_length=cfg["backbone"]["max_length"])
    coll = TripletCollator(tok, max_length=cfg["backbone"]["max_length"])
    dl = DataLoader(ds, batch_size=cfg["stage3"]["batch_size"], shuffle=True,
                    collate_fn=coll, num_workers=cfg["hardware"]["num_workers"])

    # Hard-negative miner
    miner = HardNegativeMiner(
        corpus_path=cfg["paths"]["unlabeled_corpus"],
        tokenizer=tok,
        k=cfg["mining"]["top_k_candidates"],
        rank_buckets=cfg["mining"]["rank_buckets"],
        bucket_weights=cfg["mining"]["bucket_weights"],
        jaccard_threshold=cfg["mining"]["jaccard_dedup_threshold"],
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["stage3"]["lr"],
                              weight_decay=cfg["stage3"]["weight_decay"])
    total_steps = len(dl) * cfg["stage3"]["epochs"]
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * cfg["stage3"]["warmup_ratio"]), total_steps,
    )

    model, frozen, optim, dl, sched = accel.prepare(model, frozen, optim, dl, sched)

    step = 0
    for epoch in range(cfg["stage3"]["epochs"]):
        # Refresh hard negatives at start of each epoch
        if epoch > 0 or step == 0:
            miner.refresh(model, accel.device)
            ds = miner.write_back_negatives(ds, n_per_anchor=cfg["stage3"]["hard_negs_per_anchor"])
            dl = DataLoader(ds, batch_size=cfg["stage3"]["batch_size"], shuffle=True,
                            collate_fn=coll, num_workers=cfg["hardware"]["num_workers"])
            dl = accel.prepare(dl)

        for batch in dl:
            s_a, t_a = model(**batch["anchor"])
            s_p, t_p = model(**batch["positive"])
            s_n, t_n = model(**batch["negatives"])
            with torch.no_grad():
                s_a_frozen, _ = frozen(**batch["anchor"])

            loss, parts = loss_fn(s_a, t_a, s_p, t_p, s_n, t_n, s_a_frozen, batch["neg_mask"])

            optim.zero_grad()
            accel.backward(loss)
            accel.clip_grad_norm_(model.parameters(), cfg["stage3"]["grad_clip"])
            optim.step()
            sched.step()
            step += 1

            if step % cfg["logging"]["log_every"] == 0:
                print(f"[Stage 3] step {step} loss={loss.item():.4f} {parts}")

            # Mid-epoch hard-neg refresh
            if step % cfg["stage3"]["hard_neg_refresh_steps"] == 0 and step > 0:
                miner.refresh(model, accel.device)

    accel.save_state(cfg["paths"]["stage3_ckpt"])


def _freeze_baseline(frozen_model, cfg):
    """
    Replace frozen.space_head with a small linear that picks the first space_dim
    components of the CLS, so frozen output ≈ off-the-shelf BGE-M3.
    """
    import torch.nn as nn
    hidden = frozen_model.backbone.config.hidden_size
    sd = cfg["heads"]["space_dim"]
    # Identity-like projection
    proj = nn.Linear(hidden, sd, bias=False)
    with torch.no_grad():
        proj.weight.zero_()
        for i in range(min(sd, hidden)):
            proj.weight[i, i] = 1.0
    frozen_model.space_head = nn.Sequential(proj)
    # time head doesn't matter for the regulariser
    return frozen_model


def _cfg_to_obj(d):
    """Recursively convert dict to attribute-accessible object."""
    class N: pass
    if isinstance(d, dict):
        o = N()
        for k, v in d.items():
            setattr(o, k, _cfg_to_obj(v))
        return o
    if isinstance(d, list):
        return [_cfg_to_obj(x) for x in d]
    return d
```

---

## 10. Hard-negative miner

### `lorentz_enc/data/hard_neg_miner.py`

```python
import torch
import numpy as np
import faiss
from tqdm import tqdm
import json
from typing import List
from torch.utils.data import DataLoader


class HardNegativeMiner:
    """
    ANCE-style: encode the full unlabeled corpus, retrieve top-K by space-cosine,
    bucket by rank, sample from buckets. Optionally augment with BM25 and frozen-BGE pool.
    """

    def __init__(self, corpus_path, tokenizer, k=200,
                 rank_buckets=[[0,10],[10,50],[50,200]],
                 bucket_weights=[0.3, 0.4, 0.3],
                 jaccard_threshold=0.7):
        self.tok = tokenizer
        self.k = k
        self.rank_buckets = rank_buckets
        self.bucket_weights = bucket_weights
        self.jaccard_threshold = jaccard_threshold
        # Load corpus
        self.corpus_ids, self.corpus_texts = [], []
        with open(corpus_path) as f:
            for line in f:
                ex = json.loads(line)
                self.corpus_ids.append(ex["id"])
                self.corpus_texts.append(ex["text"])
        self.index = None
        self.corpus_emb = None

    def _encode_corpus(self, model, device, batch_size=64, max_length=512):
        model.eval()
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.corpus_texts), batch_size), desc="encode corpus"):
                batch = self.corpus_texts[i:i+batch_size]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
                s, _ = model(**enc)                       # only space
                all_embs.append(s.float().cpu().numpy())
        self.corpus_emb = np.concatenate(all_embs, axis=0)
        return self.corpus_emb

    def refresh(self, model, device, max_length=512):
        emb = self._encode_corpus(model, device, max_length=max_length)
        d = emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(emb.astype(np.float32))

    def _jaccard(self, a: str, b: str, n: int = 3) -> float:
        def ngrams(s):
            toks = s.lower().split()
            return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
        A, B = ngrams(a), ngrams(b)
        if not A or not B: return 0.0
        return len(A & B) / len(A | B)

    def mine(self, anchor_text: str, positive_text: str,
             known_positive_ids: set, n_per_anchor: int = 7) -> List[str]:
        if self.index is None:
            raise RuntimeError("Miner not refreshed yet.")
        # Encode anchor
        # NOTE: caller should pass anchor embedding for efficiency; this is the slow path.
        raise NotImplementedError("Use mine_batch for efficiency.")

    def mine_batch(self, anchor_embs: np.ndarray, positives_per_anchor: List[set],
                   positive_texts: List[str], n_per_anchor: int = 7) -> List[List[str]]:
        """
        anchor_embs: (B, D_s) numpy
        positives_per_anchor: list of length B; each is a set of corpus ids that are
            known positives for this anchor (to exclude from negatives).
        positive_texts: list of length B; for jaccard dedup.
        """
        D, I = self.index.search(anchor_embs.astype(np.float32), self.k)
        out = []
        for i, ranks in enumerate(I):
            filtered = []
            for r, cid_idx in enumerate(ranks):
                cid = self.corpus_ids[cid_idx]
                if cid in positives_per_anchor[i]:
                    continue
                txt = self.corpus_texts[cid_idx]
                if self._jaccard(txt, positive_texts[i]) >= self.jaccard_threshold:
                    continue
                filtered.append((r, txt))
            # Bucket sample
            sampled = []
            for (lo, hi), w in zip(self.rank_buckets, self.bucket_weights):
                n_b = max(1, int(round(w * n_per_anchor)))
                pool = [t for r, t in filtered if lo <= r < hi]
                if pool:
                    take = np.random.choice(len(pool), size=min(n_b, len(pool)), replace=False)
                    sampled.extend([pool[k] for k in take])
            sampled = sampled[:n_per_anchor]
            while len(sampled) < n_per_anchor and filtered:
                sampled.append(filtered[len(sampled) % len(filtered)][1])
            out.append(sampled)
        return out

    def write_back_negatives(self, dataset, n_per_anchor=7):
        """
        Mutate dataset.examples in place: replace each ex['negatives'] with freshly mined ones.
        Caller must have called .refresh() with the current model first.
        """
        # Encode all anchors
        anchor_texts = [ex["anchor"] for ex in dataset.examples]
        # Quick batched encode using the same index device — assume CPU here for portability
        # Production: re-use the model from refresh().
        # For simplicity in this spec, we leave the encode-anchors step inline:
        raise NotImplementedError(
            "Compose with the calling Accelerator: encode anchors with current model, "
            "then call mine_batch and assign back to dataset.examples[i]['negatives']."
        )
```

> Implementation note: `write_back_negatives` is left as a hook for the caller because it needs access to the live `model` and `accelerator`. In `stage3_contrastive.py`, wire it up by encoding all anchors with `model` (no_grad), calling `miner.mine_batch`, then mutating `ds.examples`.

---

## 11. Optional: LLM-generated synthetic negatives

### `lorentz_enc/utils/llm_negatives.py`

```python
"""
Promptagator-style synthetic hard negatives.
Calls an LLM (Claude / GPT-4 / local Llama) to generate semantically-similar
but causally-unrelated passages for each anchor.

Usage (offline, run once):
    python -m lorentz_enc.utils.llm_negatives \
        --input data/train.jsonl \
        --output data/train_with_llm_negs.jsonl \
        --provider anthropic \
        --model claude-opus-4-7 \
        --n_per_anchor 3
"""

import json, argparse, os
from tqdm import tqdm


PROMPT = """You will be shown a SOURCE passage and its TRUE CAUSAL CONTINUATION.

Your task: write {n} new passages that are TOPICALLY SIMILAR to the source
(same domain, same terminology, similar style) but are NOT a causal continuation
of it — i.e. they don't follow logically, don't address the same need, or talk
about an adjacent topic that wouldn't be cited from the source.

SOURCE:
{source}

TRUE CAUSAL CONTINUATION:
{positive}

Output as a JSON array of strings, nothing else."""


def generate_negatives(source, positive, n, provider, model):
    prompt = PROMPT.format(n=n, source=source, positive=positive)
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
    else:
        raise ValueError(provider)
    # Parse JSON
    try:
        return json.loads(text)
    except Exception:
        # Try to extract array
        import re
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--n_per_anchor", type=int, default=3)
    args = ap.parse_args()

    with open(args.input) as fi, open(args.output, "w") as fo:
        for line in tqdm(fi):
            ex = json.loads(line)
            negs = generate_negatives(ex["anchor"], ex["positive"],
                                      args.n_per_anchor, args.provider, args.model)
            ex["negatives"] = list(ex.get("negatives", [])) + negs
            fo.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
```

---

## 12. Evaluation

### `lorentz_enc/eval/retrieval.py`

```python
import torch, numpy as np, faiss
from typing import Dict, List
from ..model.scoring import causal_similarity_matrix


def evaluate_retrieval(
    model, tokenizer, accel,
    queries: List[Dict],         # [{"anchor": str, "positive_id": int, "anchor_id": int}, ...]
    corpus: List[Dict],          # [{"id": int, "text": str}, ...]
    alpha: float, beta: float, tau: float,
    ks: List[int] = (1, 10, 100),
    use_time: bool = True,
    max_length: int = 512,
):
    """
    Standard retrieval evaluation: encode queries, encode corpus, rank by combined
    similarity, compute MRR@10, nDCG@10, Recall@k.
    """
    device = accel.device
    model.eval()

    # Encode corpus
    corpus_s, corpus_t = _encode_all(model, tokenizer, [c["text"] for c in corpus],
                                     device, max_length=max_length)
    corpus_ids = np.array([c["id"] for c in corpus])

    # Encode queries
    query_s, query_t = _encode_all(model, tokenizer, [q["anchor"] for q in queries],
                                   device, max_length=max_length)

    # Score
    if use_time:
        sims = causal_similarity_matrix(
            torch.tensor(query_s), torch.tensor(query_t),
            torch.tensor(corpus_s), torch.tensor(corpus_t),
            alpha=alpha, beta=beta, tau=tau,
        ).numpy()
    else:
        sims = query_s @ corpus_s.T

    # Rank
    targets = [q["positive_id"] for q in queries]
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics["mrr@10"] = 0.0
    metrics["ndcg@10"] = 0.0
    for qi, target in enumerate(targets):
        ranked = np.argsort(-sims[qi])
        ranked_ids = corpus_ids[ranked]
        if target in ranked_ids:
            rank = np.where(ranked_ids == target)[0][0] + 1
            for k in ks:
                if rank <= k:
                    metrics[f"recall@{k}"] += 1.0
            if rank <= 10:
                metrics["mrr@10"] += 1.0 / rank
                metrics["ndcg@10"] += 1.0 / np.log2(rank + 1)
    Q = len(queries)
    for k in metrics:
        metrics[k] /= Q
    return metrics


def _encode_all(model, tokenizer, texts, device, batch_size=64, max_length=512):
    space_all, time_all = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            s, t = model(**enc)
            space_all.append(s.float().cpu().numpy())
            time_all.append(t.float().cpu().numpy())
    return np.concatenate(space_all), np.concatenate(time_all)
```

### `lorentz_enc/eval/clough_evans_auc.py`

```python
"""
Speed-of-light AUC sweep, adapted from Clough & Evans (2017) Fig. 5.

For each candidate value of c (which corresponds to the ratio α/β in our
scoring function), classify pairs as causal (positive) vs non-causal (negative)
using a threshold sweep over the combined Minkowski-like score:
  s(a, b) = α(c) · cos(space) + β(c) · SteepSigmoid(time)
Plot sensitivity vs (1 - specificity), compute AUC.

If your space/time blocks carry real signal, you'll see AUC > 0.8 for some c.
A random-DAG control should be near 0.5.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import List, Dict
from ..model.scoring import steep_sigmoid_mean


def clough_evans_auc(
    model, tokenizer, accel,
    positive_pairs: List[Dict],   # [{"a": str, "b": str}, ...]   labeled causal
    negative_pairs: List[Dict],   # [{"a": str, "b": str}, ...]   non-causal (semantically similar but not in DAG)
    c_values: List[float] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    tau: float = 10.0,
):
    device = accel.device
    model.eval()
    # Encode all
    all_a = [p["a"] for p in positive_pairs] + [p["a"] for p in negative_pairs]
    all_b = [p["b"] for p in positive_pairs] + [p["b"] for p in negative_pairs]
    s_a, t_a = _encode_pairs(model, tokenizer, all_a, device)
    s_b, t_b = _encode_pairs(model, tokenizer, all_b, device)
    s_a, t_a = torch.tensor(s_a), torch.tensor(t_a)
    s_b, t_b = torch.tensor(s_b), torch.tensor(t_b)

    cos = (s_a * s_b).sum(-1).numpy()
    sig = steep_sigmoid_mean(t_a, t_b, tau=tau).numpy()

    labels = np.array([1]*len(positive_pairs) + [0]*len(negative_pairs))
    results = {}
    for c in c_values:
        # Score: c controls weight of time vs space (analogue of Minkowski c)
        alpha = 1.0
        beta = c
        scores = alpha * cos + beta * sig
        auc = roc_auc_score(labels, scores)
        results[f"c={c}"] = auc
    return results


def _encode_pairs(model, tokenizer, texts, device, batch_size=64, max_length=512):
    import numpy as np
    space_all, time_all = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            s, t = model(**enc)
            space_all.append(s.float().cpu().numpy())
            time_all.append(t.float().cpu().numpy())
    return np.concatenate(space_all), np.concatenate(time_all)
```

### `lorentz_enc/eval/diagnostics.py`

```python
"""
Geometric health checks:
  - directionality: for held-out (a, b+) pairs, what fraction have
    SteepSigmoid_mean(t_a, t_b) > 0.5?
  - time/space magnitude ratio: ‖t‖ / ‖s‖ over the test set.
    If this collapses toward 0, the time block is being ignored.
  - distortion: graph-theoretic distance vs space-cosine for non-causal pairs.
"""

import torch, numpy as np
from ..model.scoring import steep_sigmoid_mean


def directionality(model, tokenizer, accel, pairs, threshold=0.5):
    device = accel.device
    model.eval()
    s_a, t_a, s_b, t_b = _encode_pairs_split(model, tokenizer, pairs, device)
    s_a, t_a = torch.tensor(s_a), torch.tensor(t_a)
    s_b, t_b = torch.tensor(s_b), torch.tensor(t_b)
    scores = steep_sigmoid_mean(t_a, t_b).numpy()
    return float((scores > threshold).mean())


def time_space_magnitude(model, tokenizer, accel, texts):
    device = accel.device
    model.eval()
    s, t = _encode_single(model, tokenizer, texts, device)
    ns = np.linalg.norm(s, axis=1)
    nt = np.linalg.norm(t, axis=1)
    return {"space_norm_mean": float(ns.mean()),
            "time_norm_mean": float(nt.mean()),
            "ratio": float(nt.mean() / max(ns.mean(), 1e-9))}


def _encode_pairs_split(model, tok, pairs, device, batch_size=64, max_length=512):
    a = [p["a"] for p in pairs]
    b = [p["b"] for p in pairs]
    s_a, t_a = _encode_single(model, tok, a, device, batch_size, max_length)
    s_b, t_b = _encode_single(model, tok, b, device, batch_size, max_length)
    return s_a, t_a, s_b, t_b


def _encode_single(model, tok, texts, device, batch_size=64, max_length=512):
    space_all, time_all = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            s, t = model(**enc)
            space_all.append(s.float().cpu().numpy())
            time_all.append(t.float().cpu().numpy())
    return np.concatenate(space_all), np.concatenate(time_all)
```

---

## 13. Tests

Each module gets one minimal unit test. Examples:

### `tests/test_encoder.py`

```python
import torch
from transformers import AutoTokenizer
from lorentz_enc.model.encoder import LorentzEncoder


def test_encoder_shapes():
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    m = LorentzEncoder(space_dim=1004, time_dim=20)
    enc = tok(["hello world", "another passage"], padding=True, return_tensors="pt")
    s, t = m(**enc)
    assert s.shape == (2, 1004)
    assert t.shape == (2, 20)
    # space is L2-normalised
    assert torch.allclose(s.norm(dim=-1), torch.ones(2), atol=1e-5)
```

### `tests/test_lorentzian_mds.py`

```python
import numpy as np
import networkx as nx
from lorentz_enc.pretrain.lorentzian_mds import lorentzian_mds


def test_mds_on_chain():
    """A directed chain A->B->C->D should produce monotonically increasing time coord."""
    G = nx.DiGraph()
    G.add_edges_from([("A","B"), ("B","C"), ("C","D")])
    space, time, eig = lorentzian_mds(G, space_dim=2, time_dim=1)
    times = [float(time[n][0]) for n in ["A", "B", "C", "D"]]
    # Either monotone increasing or monotone decreasing (sign ambiguity)
    diffs = np.diff(times)
    assert (diffs > 0).all() or (diffs < 0).all()
```

### `tests/test_losses.py`

```python
import torch
from lorentz_enc.model.losses import info_nce_loss, ordering_loss, cawai_regulariser


def test_info_nce_runs():
    B, N, Ds, Dt = 4, 3, 16, 4
    s_a = torch.nn.functional.normalize(torch.randn(B, Ds), dim=-1)
    s_p = torch.nn.functional.normalize(torch.randn(B, Ds), dim=-1)
    s_n = torch.nn.functional.normalize(torch.randn(B*N, Ds), dim=-1)
    t_a = torch.randn(B, Dt); t_p = torch.randn(B, Dt); t_n = torch.randn(B*N, Dt)
    mask = torch.ones(B, N, dtype=torch.bool)
    loss = info_nce_loss(s_a, t_a, s_p, t_p, s_n, t_n, mask)
    assert torch.isfinite(loss)
    loss.backward  # should be differentiable


def test_ordering_loss():
    B, N, Dt = 2, 2, 4
    t_a = torch.zeros(B, Dt)
    t_p = torch.ones(B, Dt)             # positive in the future — good
    t_n = -torch.ones(B*N, Dt)          # negative in the past — good
    mask = torch.ones(B, N, dtype=torch.bool)
    l = ordering_loss(t_a, t_p, t_n, mask, margin=0.5, slack=0.1)
    assert l.item() < 0.1                # should be near zero
```

---

## 14. Run scripts

### `scripts/run_stage1_mds.sh`

```bash
#!/usr/bin/env bash
set -e
python -m lorentz_enc.train.stage1_mds configs/default.yaml
```

### `scripts/run_stage2_distill.sh`

```bash
#!/usr/bin/env bash
set -e
accelerate launch -m lorentz_enc.train.stage2_distill configs/default.yaml
```

### `scripts/run_stage3_contrastive.sh`

```bash
#!/usr/bin/env bash
set -e
accelerate launch -m lorentz_enc.train.stage3_contrastive configs/default.yaml
```

### `scripts/eval_all.sh`

```bash
#!/usr/bin/env bash
set -e
python -m lorentz_enc.eval.retrieval configs/default.yaml --ckpt checkpoints/stage3/
python -m lorentz_enc.eval.clough_evans_auc configs/default.yaml --ckpt checkpoints/stage3/
python -m lorentz_enc.eval.diagnostics configs/default.yaml --ckpt checkpoints/stage3/
```

### `scripts/ablate.sh`

```bash
#!/usr/bin/env bash
# Run the prioritised ablations: A, B, C, D, E from the research report
set -e

# Ablation A: no space/time decomposition (full 1024-d cosine)
python -m lorentz_enc.train.stage3_contrastive configs/ablate_no_decomp.yaml

# Ablation C: no MDS pretraining
python -m lorentz_enc.train.stage3_contrastive configs/ablate_no_mds.yaml

# Ablation D: no Cawai regulariser
python -m lorentz_enc.train.stage3_contrastive configs/ablate_no_reg.yaml

# Ablation E: no ordering loss
python -m lorentz_enc.train.stage3_contrastive configs/ablate_no_ord.yaml

# Ablation G: random negatives instead of mined
python -m lorentz_enc.train.stage3_contrastive configs/ablate_random_negs.yaml
```

For each ablation config, copy `default.yaml` and override the relevant `lambda_*` to 0 (or set the relevant block to disabled).

---

## 15. Order of operations (the go/no-go gates)

Run in this order and check the gate at each step before continuing:

| Step | Command | Gate |
|---|---|---|
| 0. Prep | `python scripts/prepare_data.py` | `train.jsonl`, `val.jsonl`, `test.jsonl`, `corpus.jsonl` exist |
| 1. Stage 3 only (skip MDS) | `bash scripts/run_stage3_contrastive.sh` after pointing `stage2_ckpt` to vanilla BGE-M3 | Beat SBERT baseline by ≥ 5 pts MRR@10 |
| 2. Add MDS pretraining | `bash scripts/run_stage1_mds.sh && bash scripts/run_stage2_distill.sh && bash scripts/run_stage3_contrastive.sh` | Beat step-1 by ≥ 0.3 pts MRR@10 AND positive Clough–Evans AUC gain |
| 3. Add LLM negatives | `python -m lorentz_enc.utils.llm_negatives …` then re-run stage 3 | Beat step-2 by ≥ 0.5 pts MRR@10 |
| 4. Final pass | `bash scripts/run_stage5_final.sh` | Beat step-3 |
| 5. Ablations | `bash scripts/ablate.sh` | Each ablation should drop performance; if any ablation doesn't hurt, remove that component |

If step 2's gate fails, drop MDS pretraining and cite Clough & Evans only as evaluation inspiration. The paper still stands.

---

## 16. What to log

For each step, log to W&B:

- `loss/{nce, ord, reg, dist, total}`
- `eval/{mrr@10, ndcg@10, recall@10, recall@100}` on val
- `geom/space_norm_mean`, `geom/time_norm_mean`, `geom/ratio` — to detect time-block collapse
- `geom/directionality` — fraction of held-out positives where t(b) > t(a)
- `geom/auc_c={0.1,…,10}` — Clough–Evans AUC at each speed-of-light
- `mining/{n_negs_mined, refresh_time}`

---

## 17. Critical correctness checks (do these before publishing)

1. **Time-block isn't collapsing.** Track `‖t‖` over training. If it goes to 0, the encoder learned to ignore time. Fix: increase `λ_ord`, decrease `α`, or freeze the space head for a few hundred steps to force the time block to carry load.

2. **MDS targets are reasonable.** After Stage 1, plot a 2-D PCA of the time block. Anchors should have lower time-coordinates than their positives on average. If not, the longest-path estimator is broken (likely a cycle in the graph).

3. **Hard negatives are actually hard.** Track `mean(cos(anchor, negative)) − mean(cos(anchor, random))`. Should be > 0.2; if not, your mining is broken.

4. **Cawai regulariser is doing work.** Track `cos(model_space(a), frozen_space(a))`. Should stay > 0.6 throughout training; if it drops below 0.4 the encoder has drifted from BGE-M3 and may have lost generic retrieval ability.

5. **Reproduce Clough & Evans Fig. 5 ordering on your own graph.** Compute AUC on (your labeled DAG) vs (shuffled-label random DAG). The real DAG should give significantly higher AUC. If it doesn't, your DAG structure isn't carrying the signal you think it is, and Stage 1 won't help.

---

## 18. What this implementation does NOT include (intentional cuts)

- **Per-relation typing.** You don't have relation labels — all the multi-time-dim-per-relation machinery has been collapsed into a single 20-dim time block. Multiple negative eigenvalues from MDS are still kept (so you can get up to 20 effective "time dimensions"), but they aren't aligned to specific relation types.
- **Auxiliary relation classifier.** Removed.
- **Off-relation suppression term** (`γ · ‖t_¬r(a) - t_¬r(b)‖²`). Removed — no relations.
- **Curriculum stage 4 on follow-up data.** Removed — all data is `(anchor, pos, negs)` already.
- **Teleportation negatives in Stage 5.** Stubbed out; only run if you want the last bit of MRR.

---

## 19. Quick start

```bash
# 1. Setup
git clone <repo> lorentz_enc && cd lorentz_enc
pip install -r requirements.txt

# 2. Prepare data (you provide train/val/test/corpus jsonl in expected schema)
python scripts/prepare_data.py --input data/raw/ --output data/

# 3. (Optional) Generate LLM negatives
python -m lorentz_enc.utils.llm_negatives \
    --input data/train.jsonl --output data/train_with_llm.jsonl \
    --provider anthropic --model claude-opus-4-7 --n_per_anchor 3
mv data/train_with_llm.jsonl data/train.jsonl

# 4. Run the pipeline
bash scripts/run_stage1_mds.sh
bash scripts/run_stage2_distill.sh
bash scripts/run_stage3_contrastive.sh

# 5. Evaluate
bash scripts/eval_all.sh

# 6. Ablate
bash scripts/ablate.sh
```

---

## 20. Expected data schema

`train.jsonl`, `val.jsonl`, `test.jsonl`:
```json
{"anchor_id": 0, "anchor": "How do I create a custom XDM schema...",
 "positive_id": 12, "positive": "What are the best practices for extending standard XDM classes...",
 "negatives": ["What is the maximum number of records...", "..."]}
```

`corpus.jsonl` (for hard-negative mining and retrieval evaluation):
```json
{"id": 0, "text": "Some passage about AEP..."}
{"id": 1, "text": "Another passage..."}
```

IDs must be integers and consistent across files. `negatives` may start empty — the miner will fill them.

---

That's the entire spec. The hand-off is: feed this document plus your dataset to Claude Code with the instruction "implement the project structure in §2 with the contents specified in §3–§13, then run §19".
