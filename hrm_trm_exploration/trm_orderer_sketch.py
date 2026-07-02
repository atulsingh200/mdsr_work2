"""
trm_orderer_sketch.py
=====================
A *sketch* (runnable-shape, NOT a trained system) of a TRM-style recursive
reasoner repurposed as a learned workflow-orderer for the AEP/AJO causal
ordering problem.

Idea in one line
----------------
Instead of solving the global ordering with a hand-written rank-aggregator
(MFAS / Kemeny over noisy pairwise scores -- the survey's Tier-1 recommendation),
we let a tiny recursive network (TRM-style, ~5-7M params) *learn* the aggregation:
it takes the step embeddings + the cross-encoder pairwise relation matrix as a
"puzzle" and iteratively refines a global ordering in latent space, exactly the
way TRM refines a Sudoku/ARC grid.

Mapping to TRM (arXiv:2510.04871):
    TRM                          ->  Workflow orderer
    question x (embedded grid)   ->  step embeddings  E  [N, d]
                                     + pairwise relation tensor R [N, N, r]
    current answer y             ->  current ordering logits  Y  [N, N]  (soft
                                     permutation / precedence matrix)
    latent state z               ->  scratchpad latent  Z  [N, h]
    recursion  z<-net(x,y,z);    ->  same: refine Z from (E,R,Y), then refine Y
               y<-net(y,z)              from (Y,Z)
    deep supervision (<=16 steps)->  same; Kendall-tau-aware loss at each step
    ACT halting                  ->  same: stop refining easy (short) workflows

This is a SKETCH. It:
  * defines the module and runs a forward pass on RANDOM tensors to prove the
    shapes line up,
  * guards all heavy ops (no training loop, no real data, tiny sizes),
  * falls back to a numpy `--dry-run` that just prints shapes if torch is absent.

Run:
    python trm_orderer_sketch.py            # torch forward on random tensors
    python trm_orderer_sketch.py --dry-run  # numpy-only shape walk-through
"""

import argparse
import sys

# --------------------------------------------------------------------------- #
# Hyperparameters (tiny on purpose -- this is the whole point of TRM)          #
# --------------------------------------------------------------------------- #
N_STEPS = 8        # workflow length (3-12 in practice); padded/masked in real use
D_EMB   = 384      # frozen encoder dim (e.g. BGE-small / MiniLM); 768/1024 for large
R_DIM   = 3        # pairwise relation channels: P(A->B), P(B->A), P(neutral)
H_LAT   = 256      # TRM latent width
LAYERS  = 2        # TRM uses a *single 2-layer* net -- "less is more"
N_INNER = 6        # latent_recursion steps per supervision step (TRM n=6)
T_OUTER = 3        # gradient-free pre-convergence passes (TRM T=3)
N_SUP   = 8        # deep-supervision steps (TRM uses up to 16)


# =========================================================================== #
#  TORCH IMPLEMENTATION (forward-shape proof only -- DO NOT add a train loop)  #
# =========================================================================== #
def build_torch_module():
    import torch
    import torch.nn as nn

    class TinyBlock(nn.Module):
        """One TRM 'layer'. For tiny N we use an MLP-mixer-ish token+channel mix
        rather than attention (TRM: when L<=D a linear [L,L] token-mix is cheap
        and generalizes better on small grids). Kept deliberately small."""
        def __init__(self, n_tok, width):
            super().__init__()
            self.tok_mix = nn.Linear(n_tok, n_tok)      # mixes across the N steps
            self.ch_norm = nn.LayerNorm(width)
            self.ch_mlp  = nn.Sequential(
                nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
            )
            self.tok_norm = nn.LayerNorm(width)

        def forward(self, h):                            # h: [B, N, width]
            # token mixing
            t = self.tok_norm(h).transpose(1, 2)          # [B, width, N]
            t = self.tok_mix(t).transpose(1, 2)           # [B, N, width]
            h = h + t
            # channel mixing
            h = h + self.ch_mlp(self.ch_norm(h))
            return h

    class TRMOrderer(nn.Module):
        """TRM-style recursive workflow orderer.

        Inputs per workflow:
            E : [B, N, D_EMB]    frozen step embeddings (encoder front-end)
            R : [B, N, N, R_DIM] pairwise relation tensor from the cross-encoder
                                 (calibrated P(A->B), P(B->A), P(neutral))
        Output:
            Y : [B, N, N]        precedence logits; row i > col j => i precedes j.
                                 A Kendall-tau / Plackett-Luce loss is applied to Y
                                 at every supervision step (deep supervision).
        """
        def __init__(self):
            super().__init__()
            # --- encoder front-end projections ---
            self.emb_proj = nn.Linear(D_EMB, H_LAT)
            # relation tensor -> per-step context: pool the N*R relation channels
            self.rel_proj = nn.Linear(N_STEPS * R_DIM, H_LAT)
            # --- the SINGLE tiny recurring net (TRM: one net, reused everywhere) ---
            self.net = nn.ModuleList([TinyBlock(N_STEPS, H_LAT) for _ in range(LAYERS)])
            # --- heads ---
            self.y_head = nn.Linear(H_LAT, N_STEPS)       # produces [B,N,N] precedence
            self.halt   = nn.Linear(H_LAT, 1)             # ACT: P(stop) per workflow

        # ---- the TRM core: refine latent z given (x, y, z), then refine answer y
        def _embed_question(self, E, R):
            B, N, _ = E.shape
            x = self.emb_proj(E)                          # [B,N,H]
            r = self.rel_proj(R.reshape(B, N, N * R_DIM)) # [B,N,H]
            return x + r                                  # fused "question" tokens

        def _latent_recursion(self, x, y_tok, z, n):
            """TRM: for i in range(n): z = net(x+y+z); then y = net(y+z)."""
            for _ in range(n):
                h = x + y_tok + z
                for blk in self.net:
                    h = blk(h)
                z = h
            h = y_tok + z
            for blk in self.net:
                h = blk(h)
            y_tok = h
            return y_tok, z

        def forward(self, E, R, n_sup=N_SUP, t_outer=T_OUTER, n_inner=N_INNER):
            import torch
            B, N, _ = E.shape
            x = self._embed_question(E, R)                # [B,N,H]
            # init answer-token and latent scratchpad from the relation prior
            y_tok = x.clone()
            z = torch.zeros_like(x)
            outputs, halts = [], []
            for _step in range(n_sup):                    # DEEP SUPERVISION
                # TRM deep_recursion: T-1 grad-free pre-converge, 1 grad pass.
                # In this *shape sketch* we don't toggle grad; just run the passes.
                for _ in range(t_outer - 1):
                    y_tok, z = self._latent_recursion(x, y_tok, z, n_inner)
                y_tok, z = self._latent_recursion(x, y_tok, z, n_inner)
                # answer head: precedence logits Y[b,i,j]
                Y = self.y_head(y_tok)                    # [B,N,N]
                outputs.append(Y)
                halts.append(self.halt(z.mean(dim=1)))    # [B,1] ACT halting score
                # detach latent before next supervision step (TRM carries state fwd)
                z = z.detach()
                y_tok = y_tok.detach()
            return outputs, halts                         # list of [B,N,N], [B,1]

    return TRMOrderer, torch


def run_torch_forward():
    TRMOrderer, torch = build_torch_module()
    torch.manual_seed(0)
    model = TRMOrderer()
    n_params = sum(p.numel() for p in model.parameters())
    B = 2
    E = torch.randn(B, N_STEPS, D_EMB)                    # frozen encoder output
    R = torch.softmax(torch.randn(B, N_STEPS, N_STEPS, R_DIM), dim=-1)
    with torch.no_grad():                                 # GUARD: no training
        outputs, halts = model(E, R)
    print(f"[torch] params: {n_params/1e6:.2f}M  (target: tiny, well under 1B)")
    print(f"[torch] E (step emb)        : {tuple(E.shape)}")
    print(f"[torch] R (relation tensor) : {tuple(R.shape)}")
    print(f"[torch] supervision steps   : {len(outputs)}")
    print(f"[torch] each Y (precedence) : {tuple(outputs[-1].shape)}  (B,N,N)")
    print(f"[torch] each halt score     : {tuple(halts[-1].shape)}  (B,1)")
    # derive an ordering from the final precedence matrix (argsort of net wins)
    Y = outputs[-1][0]                                    # [N,N]
    net_wins = (Y - Y.transpose(0, 1)).sum(dim=1)         # row beats col mass
    order = torch.argsort(net_wins, descending=True).tolist()
    print(f"[torch] decoded order (b=0) : {order}")
    print("[torch] forward-shape check OK. (No training run -- by design.)")


# =========================================================================== #
#  NUMPY DRY-RUN (works even if torch is unavailable -- pure shape walk)        #
# =========================================================================== #
def run_numpy_dryrun():
    import numpy as np
    rng = np.random.default_rng(0)
    B = 2
    E = rng.standard_normal((B, N_STEPS, D_EMB)).astype("float32")
    R = rng.random((B, N_STEPS, N_STEPS, R_DIM)).astype("float32")
    R = R / R.sum(-1, keepdims=True)
    print("[dry-run] (numpy) shape walk-through of TRM-orderer data flow")
    print(f"  E step embeddings     : {E.shape}   <- frozen encoder front-end")
    print(f"  R relation tensor     : {R.shape}   <- cross-encoder P(->),P(<-),P(0)")
    x = E.reshape(B, N_STEPS, -1)[:, :, :H_LAT]            # mock emb_proj
    print(f"  x question tokens     : {x.shape}   <- emb_proj + rel_proj fuse")
    z = np.zeros_like(x); y = x.copy()
    print(f"  z latent scratchpad   : {z.shape}")
    for s in range(N_SUP):
        for _ in range(N_INNER):
            z = np.tanh(x + y + z)                         # mock latent_recursion
        y = np.tanh(y + z)
    Y = rng.standard_normal((B, N_STEPS, N_STEPS))         # mock y_head output
    print(f"  Y precedence logits   : {Y.shape}   <- per supervision step (x{N_SUP})")
    net_wins = (Y - np.transpose(Y, (0, 2, 1))).sum(2)
    order = np.argsort(-net_wins[0]).tolist()
    print(f"  decoded order (b=0)   : {order}")
    print("[dry-run] OK. This is pseudocode-grade; see torch path for the real shapes.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="numpy-only shape walk-through (no torch needed)")
    args = ap.parse_args()
    if args.dry_run:
        run_numpy_dryrun(); return
    try:
        import torch  # noqa: F401
    except Exception as e:                                # pragma: no cover
        print(f"[info] torch unavailable ({e}); falling back to --dry-run.\n")
        run_numpy_dryrun(); return
    run_torch_forward()


if __name__ == "__main__":
    sys.exit(main())
