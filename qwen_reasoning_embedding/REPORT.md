# Adding Reasoning to Qwen3-Embedding-0.6B — Feasibility & Architecture Report

> Focused study for the AEP/AJO causal-embedding project. Scope: can we make a
> sub-1B embedder **reason, then answer/embed**? Survey of SOTA reasoning-augmented
> embedders/retrievers under 1B params, a grounded feasibility analysis for *our*
> three tasks, a clear verdict, and a minimal runnable-shape sketch.
> **No training, no weight downloads this session** — feasibility + sketch only.
>
> Companion deliverables: `reason_then_embed_sketch.py` (numpy `--dry-run`),
> survey section `sections/06c_reasoning_embeddings.tex`.
> Complements survey **sec. 06** (reasoning *reranker* angle), **sec. 03**
> (embeddings), **sec. 05** (asymmetric geometry), **sec. 11** (system design).
> This report is the reasoning *embedder* angle.

---

## 0. TL;DR verdict

**Yes, but in a specific, latency-disciplined form, and *not* as the primary
ordering model.** The grounded recommendation:

- **Primary causal scorer / orderer stays as the survey already recommends**:
  the **DeBERTa-v3-large cross-encoder + distilled rationale head** (think-free,
  zero added latency), feeding calibrated weighted-MFAS/Kemeny aggregation. The
  reasoning-embedder does **not** displace it — pairwise/cross-attention wins on
  ordering (21/30 vs 0/30).
- **The reasoning-embedder's real value is in TASK 2 (follow-up reranking) and as
  a stage-1 directional recaller** where a bi-encoder must pre-compute the corpus
  side. There the recommended architecture is:
  **Reason-then-embed on the QUERY side only (O1-Embedder style) over a
  Qwen3-Embedding-0.6B backbone, with an asymmetric cause/effect head.** This keeps
  the corpus side pre-computable while injecting genuine query-side reasoning, and
  it fixes the symmetry problem cosine cannot.
- **Worth it?** As a *complement*, yes (low risk, real upside on follow-up rerank
  and directional recall). As a *replacement for the cross-encoder on ordering*,
  no. The SOTA-grounded expectation under 1B is **moderate** gains on
  reasoning-intensive retrieval (BRIGHT-style), **not** a leap on strict ordering.

Strongest argument **for**: BRIGHT shows plain embedders fail on reasoning-intensive
retrieval and CoT helps by up to +12.2 nDCG@10; O1-Embedder, ReasonIR, RaDeR, DIVER
all confirm reason-augmentation lifts retrieval; Qwen3-Embedding-0.6B's backbone is a
full causal LM that *can* generate, so the capability is already latent. Strongest
argument **against**: generation latency breaks the bi-encoder's pre-compute/caching
on the corpus side; the survey's controlled experiment shows ordering is a *signal*
problem the cross-encoder already solves, and aggregation (not the embedder) is the
0/30 lever.

---

## 1. The backbone: Qwen3-Embedding-0.6B (arXiv:2506.05176)

Verified specs (technical report + model card):

| Property | Value |
|---|---|
| Parameters | 0.6B (28 layers) |
| Backbone | Qwen3-0.6B decoder-only **causal LM** |
| Pooling | **last-token / EOS** pooling |
| Embedding dim | up to **1024**, user-truncatable **32–1024** (Matryoshka / MRL) |
| Context length | **32k** |
| Instruction-aware | **Yes** — queries are prefixed with a natural-language instruction |
| MTEB (multilingual mean) | ~64.33 (task) |
| MTEB English v2 (task mean) | ~70.70 |
| Siblings | Qwen3-Embedding-4B/8B; **Qwen3-Reranker-0.6B/4B/8B** (generative pointwise relevance) |
| Training recipe | large-scale **synthetic weak supervision → supervised contrastive → model merging**; Qwen3 LLMs synthesize the data |

**Knobs to inject reasoning** (in increasing invasiveness):

1. **Instruction prompt** — it is instruction-aware, so `"Instruct: represent this
   step as a CAUSE …"` vs `"… as an EFFECT …"` already steers the embedding
   (zero-cost, no training). This is the cheapest directional lever.
2. **The backbone can generate.** It is a causal LM; the *base* Qwen3-0.6B (or the
   embedding model's trunk) can emit a `<think>`/thought string before pooling — the
   enabler for reason-then-embed and GritLM-style dual-mode use.
3. **A learned head** on top of the pooled vector (e.g. asymmetric cause/effect
   projections) to break cosine symmetry — needs light training.
4. **Fine-tune / distill** reasoning into the weights (think-free) — most invasive,
   highest ceiling, latency-safe at inference.

---

## 2. SOTA survey: reasoning-augmented embedders/retrievers under (and around) 1B

### 2.1 Reason-then-embed (the direct answer to the brief)
- **O1-Embedder** (arXiv:2502.07555, "Let Retrievers Think Before Action"). The model
  first **generates "thoughts" for the query**, then embeds the thought-augmented
  query; trained by **behavior cloning** of thoughts (data synthesized by an
  LLM-expert + a retrieval committee) **jointly** with contrastive dense retrieval.
  This is exactly the "reason then embed" pattern. Helps zero-shot, multi-task, and
  reasoning-intensive retrieval. *Key caveat for us: the reasoning is **query-side**;
  documents are embedded normally — preserving pre-computation.*
- **ReasonIR-8B** (arXiv:2504.20595, Meta 2025). First retriever **trained for
  reasoning tasks**; the contribution is the **ReasonIR-Synthesizer** that generates
  varied-length reasoning queries + hard, plausibly-related-but-unhelpful negatives.
  SOTA on BRIGHT (29.9 nDCG@10 no reranker, 36.9 with reranker). 8B (>1B) but the
  **synthetic-data recipe transfers to a 0.6B student**.
- **RaDeR** (arXiv:2505.18405, ICML 2025). Reasoning-aware dense retriever trained
  on **MCTS-explored, retrieval-augmented reasoning trajectories** from math; first
  dense retriever to **beat BM25 when queries are CoT steps**; generalizes to BRIGHT.
- **DIVER** (arXiv:2508.07995). A 4-stage *pipeline*: doc preprocessing, **LLM query
  expansion with explicit reasoning**, a reasoning-fine-tuned retriever (synthetic
  + hard negatives), and a pointwise reranker. SOTA on BRIGHT (~41.6 nDCG@10).
- **BRIGHT** (arXiv:2407.12883) — the pivotal benchmark: **reasoning-intensive
  retrieval where plain embedders fail** (max ~24.3 nDCG@10; strong standard-benchmark
  models drop to ~18.3), and **CoT improves retrieval by up to +12.2**. This is the
  empirical backbone for "reasoning helps embedding retrieval."

### 2.2 Test-time reasoning for retrieval (no embedder change)
- **HyDE** (arXiv:2212.10496) — LLM writes a hypothetical *document*; embed that.
- **query2doc** (arXiv:2303.07678) — few-shot LLM pseudo-document expands the query;
  +3–15% BM25, also helps dense.
- These are **training-free reason-then-embed** at the prompt level; cheap to try
  first, but add LLM-generation latency per query.

### 2.3 Unified generate + embed under/around 1B
- **GritLM / GRIT** (arXiv:2402.09906) — ONE model does **generation AND embedding via
  instruction routing**, at no loss to either; speeds RAG >60%. The blueprint for
  "one Qwen3-0.6B in two modes" (generate a rationale, then embed). *Repo caveat: our
  GritLM-7B zero-shot probe stayed semantic and ignored causal instructions — unified
  capability != directionality without targeted training.*
- **LLM2Vec** (arXiv:2404.05961) — turn a decoder LLM into an encoder via
  **bidirectional attention + MNTP + SimCSE**. Relevant if we want Qwen3-0.6B as a
  bidirectional encoder rather than last-token pooled.
- **Echo embeddings** (arXiv:2402.15449) — repeat input twice, pool the second copy, so
  early tokens "see" later ones — a training-free fix for causal-attention pooling.
- **SGPT** (arXiv:2202.08904) — early decoder-as-embedder (bi- and cross-encoder).

### 2.4 Generative pointwise reasoner-rerankers at ~0.6B (cross-ref sec. 06)
- **TFRank** (think-free): trains with CoT, runs **think-free** at inference; ships a
  **0.6B** checkpoint — existence proof of a sub-1B reasoning scorer at no inference
  latency. **Rank1** (CoT at inference). **Qwen3-Reranker-0.6B** (the matching
  generative pointwise scorer). These are the *reranker* route — see sec. 06; we do
  **not** re-derive them here.

### 2.5 Instruction-following / controllable embedders (the directional lever)
- **Promptriever** (arXiv:2409.11136) — first **zero-shot promptable** retriever;
  instruction-trained so it can be prompted like an LM (+14.3 p-MRR on FollowIR).
- **INSTRUCTOR** (instruction-prefixed embeddings), **InstructIR** (arXiv:2402.14334)
  and **FollowIR** (arXiv:2403.15246) — benchmarks showing instruction-following in IR
  is real but **fragile** (instruction-tuned retrievers can *underperform* their base).
- Relevance: "embed as CAUSE" vs "embed as EFFECT" is exactly an instruction; a
  promptable embedder gives an **asymmetric/directional** score for free-ish.

---

## 3. Architectures to add reasoning to Qwen3-0.6B (for OUR tasks)

| # | Architecture | What it does | Pro | Con | Latency | Fit |
|---|---|---|---|---|---|---|
| **A** | **Reason-then-embed** (O1-Embedder) | Qwen generates a short causal "thought" for the query, embeds query+thought | Real reasoning; lifts reasoning-intensive recall (BRIGHT) | Generation latency; **must be query-side only** or it kills corpus pre-compute | Query: +gen; Corpus: cacheable | **Task 2 (follow-up rerank), directional recall** |
| **B** | **Unified GritLM-style** | One Qwen3-0.6B: GENERATE rationale, then EMBED via instruction switch | Single model, flexible | Needs joint training; GritLM-7B probe stayed semantic on causal | Mode-dependent | Research track / Task 2 |
| **C** | **Think-free / distilled** | Train with rationales, none at inference (TFRank-style); cross-ref sec. 06 | **Zero added latency**; high ceiling | Needs teacher rationales + retrain | None at inference | **Latency-safe embedder route**; pairs with cross-encoder |
| **D** | **Latent reasoning before pooling** | Looped/recurrent depth or learned "think" tokens before last-token pool; cross-ref 06b (HRM/TRM) | No token generation; cheap-ish | Speculative for embedders; training-heavy | Small | Exploratory |
| **E** | **Generative pointwise reasoner-reranker** | Qwen3-0.6B reasons then scores a pair/precedence (TFRank/Rank1/Qwen3-Reranker) | Best directional accuracy; keeps cross-attention | O(N) gen cost; reranker not embedder | High | **Task 1 (causal direction)** — see sec. 06 |
| **F** | **Asymmetric / instructed directional embedder** | Instructions "represent as cause" / "as effect" + dual cause/effect projection heads; cross-ref sec. 05 | Antisymmetric at embedding speed; pre-computable | Lower ceiling than cross-attention | Embedding-speed | **Task 1 & 3 stage-1**, directional recall |

### Per-task mapping
- **Task 1 — Pairwise causal direction.** Inherently asymmetric. Cosine
  `s(a,b)=s(b,a)` **cannot** represent precedence (sec. 03/05). Best fit: **E**
  (generative pointwise reasoner-reranker, cross-attention) for accuracy, or **F**
  (asymmetric cause/effect head) for embedding-speed. A reason-then-embed **A** with
  an asymmetric head is the middle ground if we need pre-computed candidate vectors.
- **Task 2 — Follow-up reranking.** This is the *natural home* of reasoning-augmented
  embedders: "given this step, what should I ask/do next?" is a reasoning-intensive,
  BRIGHT-like query. **A** (reason-then-embed, query-side) + **F** (asymmetric head)
  is the recommended combination; instruction-prompting (Promptriever-style) is a
  zero-train first try.
- **Task 3 — Workflow ordering.** The survey's controlled result is decisive: ordering
  is a **signal + aggregation** problem, and the **cross-encoder already makes the
  signal aggregable** (~96% pairwise -> all workflows ordered correctly under MFAS),
  while the bi-encoder signal (~41%) is unfixable by any aggregator. So the embedder is
  **not** the ordering lever; keep **C** (distilled cross-encoder) + MFAS/Kemeny. A
  reasoning-embedder helps here only as an optional stage-1 directional recaller (**F**).

### The asymmetry problem (central)
Every cosine/inner-product embedder is **symmetric**; causal precedence is
**antisymmetric**. A reasoning/instructed embedder addresses this two ways:
(1) **instruction conditioning** — embed the *same* text differently as "cause" vs
"effect" so the comparison is no longer symmetric; (2) **asymmetric scoring head** —
`score(A->B) = <W_cause·e_A, W_effect·e_B> != score(B->A)` (the sketch). Reasoning
("what must precede this?") makes the query embedding *encode direction*, which a raw
similarity never will.

---

## 4. Verdict, with latency/accuracy vs the survey's current recommendation

**Recommended architecture (this report): A + F** — *reason-then-embed on the query
side over Qwen3-Embedding-0.6B, with an asymmetric cause/effect head*, used as a
**complement** for Task 2 and directional stage-1 recall; **not** a replacement for the
cross-encoder on ordering.

**Comparison to the survey recommendation (cross-encoder + MFAS + distilled rationale):**

| Dimension | Survey rec (CE + MFAS + distilled rationale) | Reasoning-embedder (A+F) |
|---|---|---|
| Ordering (Task 3) | **Strong** (21/30 -> ~all under MFAS) | Weak alone; only stage-1 helper |
| Follow-up rerank (Task 2) | Good (rerank shortlist) | **Best** (reasoning-intensive query) |
| Directional recall (stage-1) | N/A (CE too slow for recall) | **Enables** asymmetric recall at embedding speed |
| Inference latency | Low (think-free) | Query-side gen cost; corpus cacheable |
| Risk / cost | Low (extends existing ReasoningClassifier) | Medium (gen + head training) |
| SOTA-grounded ceiling | High on ordering | Moderate on reasoning-intensive retrieval |

**Is it worth it?** Worth building as a **complementary stage-1 / Task-2 component**,
where it has unique value (asymmetric, reasoning-aware recall and follow-up ranking).
**Not** worth it as the ordering model — the cross-encoder + aggregation owns that and
the reasoning-embedder's generation latency is a liability on the corpus side. The
honest SOTA-grounded expectation under 1B: BRIGHT-style **moderate** gains on
reasoning-intensive retrieval, contingent on the O1-Embedder/ReasonIR synthetic-data
recipe; do **not** expect a strict-ordering breakthrough from the embedder.

**Cheapest first experiments (no training):** (1) instruction-prompt Qwen3-Embedding-0.6B
with "represent as cause/effect" and measure directional separation; (2) HyDE/query2doc
query-side expansion on the follow-up task. Then, if promising, train the asymmetric head
and distill query-side thoughts (O1-Embedder behavior cloning).

---

## 5. Recommended architecture sketch

See `reason_then_embed_sketch.py` (`--dry-run` prints shapes with numpy, no torch, no
weights). Core idea, PyTorch-shape pseudocode:

```
backbone  = Qwen3-Embedding-0.6B           # HIDDEN=1024, 28 layers, 32k ctx, last-token pool
# GENERATE mode (QUERY ONLY): thought = backbone.generate("what must precede this step? " + q)  # <=64 tok
# EMBED mode: e = L2norm(last_token_pool(backbone("Instruct: <role> \n Query: <text+thought>")))  # (B,1024)
W_cause, W_effect : Linear(1024,1024, bias=False)   # asymmetric head
score(A->B) = < W_cause @ e_A , W_effect @ e_B >     # antisymmetric; cosine would be symmetric
# corpus side: embed WITHOUT generation -> precomputable/cacheable
# pairwise P(precedes) -> log-odds edges -> weighted-MFAS/Kemeny orderer (sec. 07/11)
```

Dry-run confirms shapes flow end-to-end and that the asymmetric head yields
`score(A->B) != score(B->A)` (cosine alone gives a 0 gap and cannot order).

---

## References (verified, real arXiv ids)

- Qwen3-Embedding — arXiv:2506.05176
- O1-Embedder — arXiv:2502.07555
- ReasonIR — arXiv:2504.20595
- RaDeR — arXiv:2505.18405
- DIVER — arXiv:2508.07995
- BRIGHT — arXiv:2407.12883
- HyDE — arXiv:2212.10496
- query2doc — arXiv:2303.07678
- GritLM / GRIT — arXiv:2402.09906
- LLM2Vec — arXiv:2404.05961
- Echo embeddings (Repetition Improves LM Embeddings) — arXiv:2402.15449
- SGPT — arXiv:2202.08904
- Promptriever — arXiv:2409.11136
- InstructIR — arXiv:2402.14334
- FollowIR — arXiv:2403.15246
- (TFRank / Rank1 / Qwen3-Reranker — see survey sec. 06 bib)
