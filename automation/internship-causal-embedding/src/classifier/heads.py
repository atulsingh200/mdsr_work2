"""
Pluggable classification heads for the two-encoder directional classifier.

Convention: every head takes a feature vector of size `in_dim` (= 4 * encoder
hidden size, since training concatenates [A; B; A-B; A*B]) and returns a
single raw logit per row. BCEWithLogitsLoss is applied externally — heads
do NOT apply sigmoid.

Use `build_head(cfg, in_dim)` to construct a head from a small dict that is
saved alongside the checkpoint, so weights can be loaded back into the right
architecture.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Linear head (default — matches all v1..v4 runs).
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MLP head with LayerNorm + activation + dropout BETWEEN layers.
#
# Layout (hidden_dims = [h1, h2, ...]):
#   in_dim
#     -> LayerNorm -> Linear(in_dim, h1) -> activation -> dropout
#     -> LayerNorm -> Linear(h1, h2)     -> activation -> dropout
#     ...
#     -> LayerNorm -> Linear(h_last, 1)             # final logit, no dropout/act
#
# Notes:
#   - Dropout is applied only BETWEEN hidden layers (not after the final
#     output projection), matching the user-confirmed spec.
#   - LayerNorm sits BEFORE each Linear (pre-norm); this stabilizes training
#     when L2-normalized CLS features are stacked into the [A;B;A-B;A*B] cat.
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.1,
        activation: str = "gelu",
        norm: str = "layer",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLPHead requires at least one hidden dim.")
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.activation_name = activation
        self.norm_name = norm

        act_layer = _build_activation(activation)
        norm_layer = _norm_factory(norm)

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(norm_layer(prev))
            layers.append(nn.Linear(prev, h))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # Final projection to a single logit (no dropout/activation after it).
        layers.append(norm_layer(prev))
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Xavier for activations like gelu/relu/silu — works well in practice.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _build_activation(name: str) -> type[nn.Module]:
    n = name.lower()
    if n == "gelu":
        return nn.GELU
    if n == "relu":
        return nn.ReLU
    if n in ("silu", "swish"):
        return nn.SiLU
    raise ValueError(f"unknown activation: {name}")


def _norm_factory(name: str):
    n = (name or "none").lower()
    if n == "layer":
        return nn.LayerNorm
    if n == "none":
        return nn.Identity  # ignores the (in_dim) arg silently
    raise ValueError(f"unknown norm: {name}")


def build_head(cfg: dict, in_dim: int) -> nn.Module:
    """
    cfg = {
      "head_type": "linear" | "mlp",
      # MLPHead-only:
      "hidden_dims": [512, 128],
      "dropout": 0.1,
      "activation": "gelu",
      "norm": "layer",
    }
    """
    head_type = (cfg.get("head_type") or "linear").lower()
    if head_type == "linear":
        return LinearHead(in_dim)
    if head_type == "mlp":
        return MLPHead(
            in_dim=in_dim,
            hidden_dims=cfg.get("hidden_dims", (512, 128)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=cfg.get("activation", "gelu"),
            norm=cfg.get("norm", "layer"),
        )
    raise ValueError(f"unknown head_type: {head_type}")


def head_cfg_from_args(args) -> dict:
    """Translate argparse Namespace -> head_cfg dict (small, JSON-safe)."""
    out: dict = {"head_type": getattr(args, "head_type", "linear")}
    if out["head_type"] == "mlp":
        out["hidden_dims"] = list(getattr(args, "head_hidden_dims", [512, 128]))
        out["dropout"]     = float(getattr(args, "head_dropout", 0.1))
        out["activation"]  = getattr(args, "head_activation", "gelu")
        out["norm"]        = getattr(args, "head_norm", "layer")
    return out


def parse_hidden_dims(spec: str) -> list[int]:
    """Parse '512,128' -> [512, 128]. Empty/None -> default [512, 128]."""
    if not spec:
        return [512, 128]
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    return [int(p) for p in parts]
