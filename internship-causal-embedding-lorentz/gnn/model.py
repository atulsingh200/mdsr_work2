"""Cause→Effect GNN: DirectionalGAT + asymmetric scoring heads.

Two models are provided:
  1. HybridCauseEffectGNN  — text-pair + GNN graph context (recommended).
  2. CauseEffectGNN         — pure document-level GNN (for ablation).

Key design choices in HybridCauseEffectGNN:
  - TEXT path: anchor_emb → W_cause (linear, NO ReLU — preserves all sign info
    from L2-normalised BGE embeddings).
  - GRAPH path: GNN(doc_node) → W_gnn_cause (linear).
  - FUSION: additive residual: text_proj + gnn_proj, then L2-normalise.
  - This means at random init the model ≈ "random linear projection of BGE"
    which after 1 epoch → "asymmetric linear scoring on BGE" (≈ BGE zero-shot).
    The GNN adds on top of that.

WHY NO ReLU after text projection:
  BGE-base-en-v1.5 produces unit-norm vectors with components in [-1, 1].
  ReLU zeros out negative components → loses ~50% of semantic information.
  A plain linear layer (W@x) preserves all information at initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DirectionalGATLayer(nn.Module):
    """GAT layer with separate attention over incoming vs outgoing edges.

    For each node i:
      h_in:  attention-weighted aggregate of predecessors (j → i)
      h_out: attention-weighted aggregate of successors  (i → j)
    Then: h' = LayerNorm(ReLU(W·[x; h_in; h_out]))
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat_in = GATv2Conv(
            in_dim, out_dim, heads=heads, concat=False,
            dropout=dropout, add_self_loops=False,
        )
        self.gat_out = GATv2Conv(
            in_dim, out_dim, heads=heads, concat=False,
            dropout=dropout, add_self_loops=False,
        )
        self.combine = nn.Linear(in_dim + 2 * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_in  = self.gat_in(x, edge_index)
        h_out = self.gat_out(x, edge_index[[1, 0], :])
        h = self.combine(torch.cat([x, h_in, h_out], dim=-1))
        return self.norm(F.relu(h))


class HybridCauseEffectGNN(nn.Module):
    """Hybrid text-pair + GNN model with residual fusion.

    Scoring:
      cause  = L2_norm( text_emb + W_gnn_c(GNN(src_doc)) )
      effect = L2_norm( text_emb + W_gnn_e(GNN(tgt_doc)) )
      score  = cause · effect

    WHY no text projection (W_text_cause/effect):
      A 768→256 linear compression loses 2/3 of the information from fine-tuned
      BERT/BGE embeddings. The text component would then perform WORSE than the
      raw BERT two-tower (which keeps the full 768-dim).

      Instead, we keep the full text_dim in the output and only ADD a gnn_dim
      correction term. At GNN init (near-zero weights), the model starts identical
      to the original text encoder (cosine similarity), and the GNN adds structure.

    Optionally, an asymmetric text bias can be added via W_text_cause/effect.
    Set use_text_projection=True to re-enable the linear text projection.
    """

    def __init__(
        self,
        text_dim: int = 768,
        gnn_dim: int = 256,
        out_dim: int = 768,          # default: preserve full text_dim
        num_gnn_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_text_projection: bool = False,  # set True to project text to out_dim
    ):
        super().__init__()
        self.text_dim  = text_dim
        self.gnn_dim   = gnn_dim
        self.out_dim   = text_dim if not use_text_projection else out_dim

        # --- Optional asymmetric text projections ---
        self.use_text_projection = use_text_projection
        if use_text_projection:
            self.W_text_cause  = nn.Linear(text_dim, out_dim, bias=False)
            self.W_text_effect = nn.Linear(text_dim, out_dim, bias=False)
        else:
            # No text projection — use raw text embedding directly
            self.W_text_cause  = None
            self.W_text_effect = None

        # --- GNN graph encoder ---
        self.gnn_input_proj = nn.Linear(text_dim, gnn_dim, bias=False)
        self.gnn_layers = nn.ModuleList([
            DirectionalGATLayer(gnn_dim, gnn_dim, heads=heads, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        nn.init.normal_(self.gnn_input_proj.weight, std=0.001)

        # --- GNN-to-output projections: gnn_dim → out_dim ---
        gnn_out = self.out_dim
        self.W_gnn_cause  = nn.Linear(gnn_dim, gnn_out, bias=False)
        self.W_gnn_effect = nn.Linear(gnn_dim, gnn_out, bias=False)
        # Very near-zero init: GNN starts as a tiny perturbation, text dominates
        nn.init.normal_(self.W_gnn_cause.weight,  std=0.001)
        nn.init.normal_(self.W_gnn_effect.weight, std=0.001)

        self.dropout = nn.Dropout(dropout)

    def encode_graph(self, x_text: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run GNN; returns [N_docs, gnn_dim]."""
        h = self.gnn_input_proj(x_text)
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
        return h

    def _text_cause(self, text_emb: torch.Tensor) -> torch.Tensor:
        if self.W_text_cause is not None:
            return self.W_text_cause(text_emb)
        return text_emb

    def _text_effect(self, text_emb: torch.Tensor) -> torch.Tensor:
        if self.W_text_effect is not None:
            return self.W_text_effect(text_emb)
        return text_emb

    def fuse_cause(
        self,
        text_emb: torch.Tensor,  # [B, text_dim]
        gnn_emb: torch.Tensor,   # [B, gnn_dim]
    ) -> torch.Tensor:
        """Returns L2-normalised cause embeddings [B, out_dim]."""
        t = self._text_cause(text_emb)
        g = self.W_gnn_cause(self.dropout(gnn_emb))
        return F.normalize(t + g, dim=-1)

    def fuse_effect(
        self,
        text_emb: torch.Tensor,  # [B, text_dim] or [B, K, text_dim]
        gnn_emb: torch.Tensor,   # [B, gnn_dim]  or [B, K, gnn_dim]
    ) -> torch.Tensor:
        """Returns L2-normalised effect embeddings [B, out_dim] or [B, K, out_dim]."""
        if text_emb.dim() == 3:
            B, K, _ = text_emb.shape
            t = self._text_effect(text_emb)
            g = self.W_gnn_effect(self.dropout(gnn_emb))
            return F.normalize(t + g, dim=-1)
        t = self._text_effect(text_emb)
        g = self.W_gnn_effect(self.dropout(gnn_emb))
        return F.normalize(t + g, dim=-1)

    def score(
        self,
        cause_emb: torch.Tensor,
        effect_emb: torch.Tensor,
    ) -> torch.Tensor:
        if effect_emb.dim() == 3:
            return torch.einsum("bd,bkd->bk", cause_emb, effect_emb)
        return (cause_emb * effect_emb).sum(dim=-1)


class CauseEffectGNN(nn.Module):
    """Pure document-level GNN with L2-normalised heads (for ablation)."""

    def __init__(
        self,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.gnn_layers = nn.ModuleList([
            DirectionalGATLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.W_cause  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_effect = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def encode_graph(self, x_text: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_text)
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
        return h

    def score(self, h_cause: torch.Tensor, h_effect: torch.Tensor) -> torch.Tensor:
        phi_c = F.normalize(self.W_cause(h_cause), dim=-1)
        phi_e = F.normalize(self.W_effect(h_effect), dim=-1)
        if phi_e.dim() == 3:
            return torch.einsum("bd,bkd->bk", phi_c, phi_e)
        return (phi_c * phi_e).sum(dim=-1)
