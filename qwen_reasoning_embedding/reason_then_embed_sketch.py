#!/usr/bin/env python3
"""
reason_then_embed_sketch.py
===========================

Architecture SKETCH (not a training script) for adding *reasoning* to a sub-1B
embedder for the AEP/AJO causal-precedence / workflow-ordering project.

Recommended architecture (see REPORT.md, "Verdict"):
    Reason-then-embed wrapper around ONE Qwen3-Embedding-0.6B backbone, used in
    TWO modes via instruction switching (GritLM-style):
      (1) GENERATE  mode -> emit a short causal "thought" for the QUERY only.
      (2) EMBED     mode -> last-token pool an instruction-conditioned embedding.
    On top of EMBED we add an *asymmetric directional head*: two learned
    projections W_cause / W_effect so the pair score is antisymmetric
    (cosine alone is symmetric and cannot represent precedence).

Latency rule baked into the design:
    - Generation (the "thought") is QUERY-SIDE ONLY. The corpus/candidate side is
      a plain instructed embedding so it stays PRE-COMPUTABLE and cacheable.
      Generating a thought per corpus doc would destroy the bi-encoder advantage.

This file is deliberately SAFE to run with no GPU, no torch, and no weights:
    python reason_then_embed_sketch.py --dry-run
prints the tensor shapes through the whole pipeline using only numpy. The real
torch path is guarded and clearly marked PSEUDOCODE; it is NOT exercised here and
will NOT download weights.

NO TRAINING. NO WEIGHT DOWNLOAD. Shapes only.
"""

import argparse
import sys

# ---- config (Qwen3-Embedding-0.6B real specs, arXiv:2506.05176) -------------
HIDDEN = 1024          # embedding dim (MRL: truncatable 32..1024)
N_LAYERS = 28
MAX_CTX = 32768        # 32k context
VOCAB = 151_669        # approx Qwen3 tokenizer vocab (illustrative)
MRL_DIM = 256          # example Matryoshka truncation for stage-1 recall

# ----------------------------------------------------------------------------
# REAL TORCH PATH -- PSEUDOCODE, guarded so the file imports without torch.
# ----------------------------------------------------------------------------
def build_real_model():  # pragma: no cover  (never called in --dry-run)
    """
    PSEUDOCODE ONLY. Do not call this in this session (it would load weights).
    Sketches the recommended reason-then-embed + asymmetric-head wrapper.
    """
    import torch                                   # guarded heavy import
    import torch.nn as nn
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    NAME = "Qwen/Qwen3-Embedding-0.6B"             # do NOT download this session

    class ReasonThenEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            # ONE backbone, two views. In practice Qwen3-Embedding ships the
            # embedding head; for true GENERATE mode you would attach the base
            # Qwen3-0.6B LM head (GritLM-style shared trunk) or run a sibling
            # Qwen3-0.6B-Base. Shown as two handles for clarity.
            self.tok = AutoTokenizer.from_pretrained(NAME)
            self.embed_lm = AutoModel.from_pretrained(NAME)            # EMBED mode
            self.gen_lm = AutoModelForCausalLM.from_pretrained(NAME)   # GENERATE mode (shared/sibling)
            # Asymmetric directional head: project the pooled embedding into a
            # "cause" space and an "effect" space. Score(A->B) = <cause(A), effect(B)>.
            self.W_cause = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.W_effect = nn.Linear(HIDDEN, HIDDEN, bias=False)

        @staticmethod
        def last_token_pool(hidden_states, attn_mask):
            # Qwen3 uses left/right padding aware last-token (EOS) pooling.
            lengths = attn_mask.sum(dim=1) - 1
            idx = torch.arange(hidden_states.size(0))
            return hidden_states[idx, lengths]      # (B, HIDDEN)

        def generate_thought(self, query, instruction):
            # GENERATE mode: short causal expansion / "thought" for the QUERY.
            # e.g. instruction = "Think step by step about what must happen
            #      BEFORE this step in an Adobe Journey Optimizer workflow."
            prompt = f"{instruction}\nStep: {query}\nThought:"
            ids = self.tok(prompt, return_tensors="pt")
            out = self.gen_lm.generate(**ids, max_new_tokens=64)   # short thought
            return self.tok.decode(out[0][ids["input_ids"].shape[1]:])

        def embed(self, text, instruction):
            # EMBED mode: instruction-conditioned, last-token pooled, L2-normed.
            s = f"Instruct: {instruction}\nQuery: {text}"
            ids = self.tok(s, return_tensors="pt", truncation=True, max_length=MAX_CTX)
            h = self.embed_lm(**ids).last_hidden_state          # (B, T, HIDDEN)
            v = self.last_token_pool(h, ids["attention_mask"])  # (B, HIDDEN)
            return torch.nn.functional.normalize(v, dim=-1)

        def score_precedence(self, e_a, e_b):
            # Antisymmetric: s(A->B) != s(B->A) in general.
            return (self.W_cause(e_a) * self.W_effect(e_b)).sum(-1)

    return ReasonThenEmbed()


# ----------------------------------------------------------------------------
# DRY-RUN PATH -- numpy only, prints shapes through the pipeline.
# ----------------------------------------------------------------------------
def dry_run():
    import numpy as np
    rng = np.random.default_rng(0)

    B, T = 4, 37                      # batch of 4 steps, 37 tokens after tokenize
    print("== reason-then-embed dry-run (numpy; no torch, no weights) ==")
    print(f"backbone           : Qwen3-Embedding-0.6B  (HIDDEN={HIDDEN}, "
          f"LAYERS={N_LAYERS}, MAX_CTX={MAX_CTX})")

    # ---- GENERATE mode (QUERY-SIDE ONLY) ----
    # A short causal "thought" appended to the query before embedding.
    thought_tokens = 64
    print("\n[GENERATE mode] query-side only (corpus side skips this):")
    print(f"  query_ids            : {(1, T)}")
    print(f"  generated thought    : {(1, thought_tokens)}  (<= 64 new tokens)")
    print(f"  query+thought_ids    : {(1, T + thought_tokens)}")

    # ---- EMBED mode ----
    print("\n[EMBED mode] instruction-conditioned, last-token pool:")
    hidden_states = rng.standard_normal((B, T, HIDDEN)).astype("float32")
    attn = np.ones((B, T), dtype="int64")
    print(f"  input_ids            : {(B, T)}")
    print(f"  last_hidden_state    : {hidden_states.shape}")
    lengths = attn.sum(1) - 1
    pooled = hidden_states[np.arange(B), lengths]            # last-token pool
    pooled = pooled / (np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-9)
    print(f"  pooled (last-token)  : {pooled.shape}   L2-normed")
    print(f"  MRL truncated        : {(B, MRL_DIM)}   (stage-1 recall view)")

    # ---- Asymmetric directional head ----
    print("\n[ASYMMETRIC HEAD] cause/effect projections -> antisymmetric score:")
    W_cause = rng.standard_normal((HIDDEN, HIDDEN)).astype("float32") / np.sqrt(HIDDEN)
    W_effect = rng.standard_normal((HIDDEN, HIDDEN)).astype("float32") / np.sqrt(HIDDEN)
    e_a, e_b = pooled[0:1], pooled[1:2]                       # one pair (A, B)
    cause_a = e_a @ W_cause                                   # (1, HIDDEN)
    effect_b = e_b @ W_effect                                 # (1, HIDDEN)
    s_ab = float((cause_a * effect_b).sum())
    cause_b = e_b @ W_cause
    effect_a = e_a @ W_effect
    s_ba = float((cause_b * effect_a).sum())
    print(f"  W_cause              : {W_cause.shape}")
    print(f"  W_effect             : {W_effect.shape}")
    print(f"  score(A->B)          : {s_ab:+.4f}")
    print(f"  score(B->A)          : {s_ba:+.4f}")
    print(f"  asymmetry |s_ab-s_ba|: {abs(s_ab - s_ba):.4f}   "
          f"(cosine alone would give 0 -> cannot order)")

    # ---- pairwise direction logit for a full workflow (n steps) ----
    n = 6
    P = np.zeros((n, n), dtype="float32")
    embs = rng.standard_normal((n, HIDDEN)).astype("float32")
    embs /= (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-9)
    C = embs @ W_cause
    E = embs @ W_effect
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = 1.0 / (1.0 + np.exp(-(C[i] @ E[j])))   # sigmoid logit
    print(f"\n[ORDERING] pairwise P(precedes) matrix for {n}-step workflow: {P.shape}")
    print("  -> feed log-odds(P) as edge weights to weighted-MFAS/Kemeny")
    print("     orderer (see survey sec. 07/11). NEUTRAL pairs -> antichains.")

    print("\nOK: shapes flow end-to-end. No torch, no weights, no training.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print tensor shapes with numpy only (safe, no weights)")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    print("This is an architecture SKETCH. Run with --dry-run for a safe shape check.")
    print("build_real_model() is PSEUDOCODE and would load weights; do NOT call it "
          "in a constrained session.", file=sys.stderr)


if __name__ == "__main__":
    main()
