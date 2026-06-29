"""CDEv2 — Dual-encoder Causal Density Embedding with normalized hybrid scoring.

Key improvements over v1:
  1. Dual BERT encoders: cause_enc and effect_enc for specialized representations
  2. 2-layer MLP projection heads with GELU + LayerNorm
  3. proj_dim = 256 (2x larger)
  4. Normalized hybrid score: -(KL/D) + lambda_cos * cosine(mu_ta, mu_b)
     - Dividing by D removes scale sensitivity → dramatically improves AUC calibration
     - Cosine term provides bounded similarity for random-negative discrimination
     - lambda_cos is learned (init 1.0)
  5. Separate encode_cause / encode_effect paths for clean inference API
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class _MLP(nn.Module):
    """2-layer MLP: Linear -> GELU -> LayerNorm -> Linear."""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        mid = max(out_dim * 2, (in_dim + out_dim) // 2)
        self.fc1 = nn.Linear(in_dim, mid)
        self.norm = nn.LayerNorm(mid)
        self.fc2 = nn.Linear(mid, out_dim)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(F.gelu(self.fc1(x))))


class CDEv2(nn.Module):
    """Dual-encoder Causal Density Embedding v2."""

    def __init__(
        self,
        backbone: str = "google-bert/bert-base-uncased",
        proj_dim: int = 256,
        shared_encoder: bool = False,
        kl_scale: str = "dim",
        init_cos_weight: float = 1.0,
        pooling: str = "mean",
        mu_identity: bool = False,
        log_sigma_init: float = 0.0,
        score_type: str = "kl",
        mmd_gamma_init: float = 0.01,
    ) -> None:
        """
        shared_encoder: if True, cause and effect share one BERT backbone
            (asymmetry comes only from the projection heads). This matches
            CDE v1, uses half the params, and trains better on small data.
        kl_scale: "dim" divides KL by proj_dim (scale-invariant, good for AUC);
            "none" uses raw summed KL (stronger MRR signal, like v1).
        init_cos_weight: initial weight of the cosine term in the hybrid score.
            Set to 0 to disable the cosine term (pure KL scoring like v1).
        pooling: "mean" or "cls" — must match the warm-start checkpoint.
        mu_identity: if True, mu = the pooled embedding directly (no projection),
            so proj_dim must equal hidden_size. Used for BiEncoder warm-start:
            the cosine term then equals the BiEncoder dot-product at init.
        log_sigma_init: constant bias for log_sigma heads. A large value (e.g. 2.0)
            makes initial variances big so the KL term is near-zero at start,
            letting a warm-started model reproduce the BiEncoder's cosine score.
        """
        super().__init__()
        self.proj_dim = proj_dim
        self.shared_encoder = shared_encoder
        self.kl_scale = kl_scale
        self.pooling = pooling
        self.mu_identity = mu_identity
        self.score_type = score_type

        # Encoder(s) for cause and effect sides.
        self.cause_enc = AutoModel.from_pretrained(backbone)
        self.effect_enc = self.cause_enc if shared_encoder else AutoModel.from_pretrained(backbone)
        h = self.cause_enc.config.hidden_size
        if mu_identity and proj_dim != h:
            raise ValueError(f"mu_identity requires proj_dim==hidden_size ({h}), got {proj_dim}")

        # Cause-side projections: mu, log_sigma, causal drift delta.
        self.cause_mu = nn.Identity() if mu_identity else _MLP(h, proj_dim)
        self.cause_log_sigma = _MLP(h, proj_dim)
        self.cause_delta = _MLP(h, proj_dim)

        # Effect-side projections: mu, log_sigma only.
        self.effect_mu = nn.Identity() if mu_identity else _MLP(h, proj_dim)
        self.effect_log_sigma = _MLP(h, proj_dim)

        # Learned scalar: variance inflation per causal transport step.
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # ── Asymmetric directed-MMD scoring params (used when score_type=="mmd") ──
        # gamma_k: RBF kernel bandwidth (kept positive via exp of a free param).
        # w, b:    learned directional gate  sigma(w·(mu_B - mu_A') + b).
        self.log_gamma_k = nn.Parameter(torch.tensor(float(mmd_gamma_init)).log())
        self.gate_w = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.gate_b = nn.Parameter(torch.tensor(0.0))
        # Learned cosine weight in hybrid score. init_cos_weight=0 disables it.
        self.use_cos = init_cos_weight > 0
        if self.use_cos:
            self.log_cos_weight = nn.Parameter(torch.tensor(float(init_cos_weight)).log())
        else:
            self.register_buffer("_zero_cos", torch.zeros(1))

        # Initialize delta and log_sigma heads near 0 for stable startup;
        # log_sigma gets a constant bias (log_sigma_init) so variances can be
        # large at warm-start, suppressing the KL term initially.
        for mlp in (self.cause_delta, self.cause_log_sigma, self.effect_log_sigma):
            nn.init.normal_(mlp.fc2.weight, std=1e-3)
            nn.init.zeros_(mlp.fc2.bias)
        for mlp in (self.cause_log_sigma, self.effect_log_sigma):
            nn.init.constant_(mlp.fc2.bias, log_sigma_init)

    def _pool(self, out, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return out.last_hidden_state[:, 0]
        return _mean_pool(out.last_hidden_state, attention_mask)

    @torch.no_grad()
    def load_from_biencoder(self, ckpt_path: str) -> None:
        """Warm-start cause_enc/effect_enc from a trained BiEncoder checkpoint.

        Maps anchor_encoder -> cause_enc and positive_encoder -> effect_enc.
        With mu_identity + matching pooling + large log_sigma_init, the model's
        cosine score at init equals the BiEncoder's dot-product score.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        anchor_sd = {k[len("anchor_encoder."):]: v for k, v in sd.items()
                     if k.startswith("anchor_encoder.")}
        pos_sd = {k[len("positive_encoder."):]: v for k, v in sd.items()
                  if k.startswith("positive_encoder.")}
        missing_a, unexpected_a = self.cause_enc.load_state_dict(anchor_sd, strict=False)
        if not self.shared_encoder:
            self.effect_enc.load_state_dict(pos_sd, strict=False)
        n = len(anchor_sd)
        print(f"  [warm-start] loaded {n} anchor tensors -> cause_enc, "
              f"{len(pos_sd)} positive tensors -> effect_enc "
              f"(missing={len(missing_a)}, unexpected={len(unexpected_a)})")

    # ── Encoders ──────────────────────────────────────────────────────────────

    def encode_cause(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, log_sigma, delta) from the cause encoder."""
        out = self.cause_enc(input_ids=input_ids, attention_mask=attention_mask)
        h = self._pool(out, attention_mask)
        mu = self.cause_mu(h)
        log_sigma = self.cause_log_sigma(h).clamp(-5.0, 5.0)
        delta = self.cause_delta(h)
        return mu, log_sigma, delta

    def encode_effect(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_sigma) from the effect encoder."""
        out = self.effect_enc(input_ids=input_ids, attention_mask=attention_mask)
        h = self._pool(out, attention_mask)
        mu = self.effect_mu(h)
        log_sigma = self.effect_log_sigma(h).clamp(-5.0, 5.0)
        return mu, log_sigma

    def transport(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal transport: shift mean by delta, inflate variance by gamma/2."""
        return mu + delta, log_sigma + 0.5 * self.gamma

    # ── KL helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def kl_pointwise(
        mu_p: torch.Tensor,
        log_sigma_p: torch.Tensor,
        mu_q: torch.Tensor,
        log_sigma_q: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N_p || N_q) for N independent pairs — shape (N,)."""
        sigma_p_sq = (2 * log_sigma_p).exp()
        sigma_q_sq = (2 * log_sigma_q).exp().clamp(min=1e-9)
        log_ratio = log_sigma_q - log_sigma_p
        mean_term = (sigma_p_sq + (mu_p - mu_q).pow(2)) / (2 * sigma_q_sq)
        return (log_ratio + mean_term - 0.5).sum(dim=-1)

    @staticmethod
    def kl_matrix(
        mu_q: torch.Tensor,
        log_sigma_q: torch.Tensor,
        mu_c: torch.Tensor,
        log_sigma_c: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N_c[j] || N_q[i]) for all (i,j) — shape (N_q, N_c)."""
        D = mu_q.shape[-1]
        sigma_q_sq = (2 * log_sigma_q).exp().clamp(min=1e-9)
        sigma_c_sq = (2 * log_sigma_c).exp()
        inv_sigma_q_sq = 1.0 / sigma_q_sq

        t1 = log_sigma_q.sum(-1).unsqueeze(1) - log_sigma_c.sum(-1).unsqueeze(0)
        t2 = 0.5 * (inv_sigma_q_sq @ sigma_c_sq.T)
        half_inv = 0.5 * inv_sigma_q_sq
        t3 = (
            half_inv @ (mu_c.pow(2)).T
            - (mu_q * inv_sigma_q_sq) @ mu_c.T
            + (mu_q.pow(2) * half_inv).sum(-1, keepdim=True)
        )
        return t1 + t2 + t3 - 0.5 * D

    # ── Hybrid scoring ────────────────────────────────────────────────────────

    @property
    def cos_weight(self) -> torch.Tensor:
        if self.use_cos:
            return self.log_cos_weight.exp()
        return self._zero_cos

    def _kl_denom(self) -> float:
        return float(self.proj_dim) if self.kl_scale == "dim" else 1.0

    # ── Directed-MMD helpers (closed-form expected RBF between two Gaussians,
    #    times a learned directional sigmoid gate) ─────────────────────────────
    #
    #   score(A→B) = log E_{x~N_A', y~N_B}[exp(-gamma||x-y||^2)]  +  log sigma(gate)
    #   expected-RBF (per dim d, diagonal Gaussians):
    #       E[..] = prod_d (1+2g(s2A_d+s2B_d))^{-1/2} * exp(-g (mA_d-mB_d)^2/(1+2g(..)))
    #   gate = w·(mu_B - mu_A') + b   (asymmetric: nonzero only in the causal dir)
    #
    #   The denominator (1+2g·var) >= 1 so this never divides by zero and the
    #   gradient stays smooth at long range (vs KL which blows up / plateaus).

    def mmd_matrix(self, mu_a, log_sigma_a, mu_b, log_sigma_b):
        """Directed-MMD score for all (i,j) pairs — shape (N_a, N_b)."""
        g = self.log_gamma_k.exp()
        s2a = (2 * log_sigma_a).exp()                       # (Na, D)
        s2b = (2 * log_sigma_b).exp()                       # (Nb, D)
        # denom[i,j,d] = 1 + 2g (s2a[i,d] + s2b[j,d])
        denom = 1.0 + 2.0 * g * (s2a[:, None, :] + s2b[None, :, :])   # (Na,Nb,D)
        diff_sq = (mu_a[:, None, :] - mu_b[None, :, :]).pow(2)        # (Na,Nb,D)
        log_rbf = (-0.5 * denom.log() - g * diff_sq / denom).sum(-1)  # (Na,Nb)
        # directional gate: w·(mu_b - mu_a') = (mu_b@w) - (mu_a@w)
        gate = (mu_b @ self.gate_w)[None, :] - (mu_a @ self.gate_w)[:, None] + self.gate_b
        return log_rbf + F.logsigmoid(gate)

    def mmd_pointwise(self, mu_a, log_sigma_a, mu_b, log_sigma_b):
        """Directed-MMD score for N independent pairs — shape (N,)."""
        g = self.log_gamma_k.exp()
        s2a = (2 * log_sigma_a).exp()
        s2b = (2 * log_sigma_b).exp()
        denom = 1.0 + 2.0 * g * (s2a + s2b)
        diff_sq = (mu_a - mu_b).pow(2)
        log_rbf = (-0.5 * denom.log() - g * diff_sq / denom).sum(-1)
        gate = ((mu_b - mu_a) * self.gate_w).sum(-1) + self.gate_b
        return log_rbf + F.logsigmoid(gate)

    def score_matrix(
        self,
        mu_ta: torch.Tensor,
        log_sigma_ta: torch.Tensor,
        mu_b: torch.Tensor,
        log_sigma_b: torch.Tensor,
    ) -> torch.Tensor:
        """Score matrix S[i,j] for query i vs candidate j (higher = better)."""
        if self.score_type == "mmd":
            S = self.mmd_matrix(mu_ta, log_sigma_ta, mu_b, log_sigma_b)
        else:
            S = -self.kl_matrix(mu_ta, log_sigma_ta, mu_b, log_sigma_b) / self._kl_denom()
        if self.use_cos:
            mu_ta_n = F.normalize(mu_ta, dim=-1)
            mu_b_n = F.normalize(mu_b, dim=-1)
            S = S + self.cos_weight * (mu_ta_n @ mu_b_n.T)
        return S

    def score_pointwise(
        self,
        mu_ta: torch.Tensor,
        log_sigma_ta: torch.Tensor,
        mu_b: torch.Tensor,
        log_sigma_b: torch.Tensor,
    ) -> torch.Tensor:
        """Pointwise score for (B,) pairs."""
        if self.score_type == "mmd":
            s = self.mmd_pointwise(mu_ta, log_sigma_ta, mu_b, log_sigma_b)
        else:
            s = -self.kl_pointwise(mu_b, log_sigma_b, mu_ta, log_sigma_ta) / self._kl_denom()
        if self.use_cos:
            s = s + self.cos_weight * F.cosine_similarity(mu_ta, mu_b, dim=-1)
        return s

    # ── Forward features for multi-task loss ─────────────────────────────────

    def forward_features(
        self,
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
        include_reverse: bool = False,
    ) -> dict:
        """Compute forward pass tensors for loss computation.

        If include_reverse=True (Phase B/C), also encodes B as cause and A as
        effect to compute the reverse score(B->A) for antisymmetry.
        To save compute, concatenates [A, B] per encoder so it's 2 passes
        of 2B instead of 4 passes of B.
        """
        if include_reverse:
            B = a_ids.size(0)
            # Combine a+b for cause encoder, a+b for effect encoder.
            ab_ids = torch.cat([a_ids, b_ids], dim=0)
            ab_mask = torch.cat([a_mask, b_mask], dim=0)

            cause_out = self.cause_enc(input_ids=ab_ids, attention_mask=ab_mask)
            h_cause = self._pool(cause_out, ab_mask)
            h_cause_a, h_cause_b = h_cause[:B], h_cause[B:]

            effect_out = self.effect_enc(input_ids=ab_ids, attention_mask=ab_mask)
            h_eff = self._pool(effect_out, ab_mask)
            h_eff_a, h_eff_b = h_eff[:B], h_eff[B:]

            # A as cause (for A→B)
            mu_a = self.cause_mu(h_cause_a)
            log_sigma_a = self.cause_log_sigma(h_cause_a).clamp(-5.0, 5.0)
            delta_a = self.cause_delta(h_cause_a)
            mu_ta, log_sigma_ta = self.transport(mu_a, log_sigma_a, delta_a)

            # B as effect (for A→B)
            mu_b = self.effect_mu(h_eff_b)
            log_sigma_b = self.effect_log_sigma(h_eff_b).clamp(-5.0, 5.0)

            # B as cause (for B→A)
            mu_bc = self.cause_mu(h_cause_b)
            log_sigma_bc = self.cause_log_sigma(h_cause_b).clamp(-5.0, 5.0)
            delta_b = self.cause_delta(h_cause_b)
            mu_tb, log_sigma_tb = self.transport(mu_bc, log_sigma_bc, delta_b)

            # A as effect (for B→A)
            mu_ae = self.effect_mu(h_eff_a)
            log_sigma_ae = self.effect_log_sigma(h_eff_a).clamp(-5.0, 5.0)

            return {
                "mu_a": mu_a, "log_sigma_a": log_sigma_a, "delta_a": delta_a,
                "mu_b": mu_b, "log_sigma_b": log_sigma_b,
                "mu_ta": mu_ta, "log_sigma_ta": log_sigma_ta,
                "mu_tb": mu_tb, "log_sigma_tb": log_sigma_tb,
                "mu_ae": mu_ae, "log_sigma_ae": log_sigma_ae,
            }
        else:
            mu_a, log_sigma_a, delta_a = self.encode_cause(a_ids, a_mask)
            mu_b, log_sigma_b = self.encode_effect(b_ids, b_mask)
            mu_ta, log_sigma_ta = self.transport(mu_a, log_sigma_a, delta_a)
            return {
                "mu_a": mu_a, "log_sigma_a": log_sigma_a, "delta_a": delta_a,
                "mu_b": mu_b, "log_sigma_b": log_sigma_b,
                "mu_ta": mu_ta, "log_sigma_ta": log_sigma_ta,
            }
