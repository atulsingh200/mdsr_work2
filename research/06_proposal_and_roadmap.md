# 06 — Proposal & Roadmap: A Sub-1B Causal Reasoning System for AEP/AJO Workflows

> Synthesis of `01`–`05`. This is the deliverable: a recommended architecture, a
> multi-agent system design (one agent per task), a training pipeline, an evaluation
> protocol, and a phased roadmap. **No training is started in this round** — this is the plan.

---

## 0. Executive summary

- **Keep the strong bi-encoder as stage-1 retriever** (BGE-large / all-mpnet, +21.3% MRR over BERT baseline). The GNN study showed graph structure adds nothing; don't revisit it.
- **Add a sub-1B *generative* reasoning model as stage-2** — a **Qwen3-0.6B** causal-precedence reranker trained **TFRank-style** (learns to reason in training, runs "think-free" at inference for speed; can emit a `<think>` trace on demand). This is the only way to get genuine "reason-then-answer" under 1B; TFRank ships a released **0.6B** checkpoint as existence proof (`03`).
- **Train it distillation-first, then a short GRPO polish.** Every one of the four sub-reports converges on this: pure RL on sub-1B models is fragile; DeepSeek-R1-Distill shows distillation >> from-scratch RL at small scale (`02`, `05`).
- **Frame the task as directional entailment**: `PRECEDES / FOLLOWS / NEUTRAL`, with a causal CoT trace (CausalCoT + VeriCoT-validated) (`04`).
- **Workflow ordering** = aggregate pairwise precedence scores via topological / tournament sort (`order_events.py`), with a verifier resolving cycles. This is the 0/30 task we most want to fix.
- **Manage expectations:** causal *direction* is genuinely hard — GPT-4 is near-random on Corr2Cause (F1 ≈ 29); supervised direction prediction tops out ~0.79–0.86 F1 (`04`). Reasoning + distillation is exactly the lever shown to help here (CausalCoT +8.37 on CLADDER).

---

## 1. Recommended architecture

```
                         ┌─────────────────────────────────────────────┐
   workflow steps  ──►   │  STAGE 1: Bi-encoder retriever (BGE-large)   │  (existing, keep)
   / candidate set       │  cosine cause→effect → top-k candidates      │  ~335M
                         └───────────────────────┬─────────────────────┘
                                                 │ top-k (anchor, candidate) pairs
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │  STAGE 2: Sub-1B Causal Reasoner             │  Qwen3-0.6B
                         │  pointwise: (t1,t2) → [optional <think>] →   │  TFRank-style
                         │  {PRECEDES|FOLLOWS|NEUTRAL} + calibrated score│  think-free @ infer
                         └───────────────────────┬─────────────────────┘
                                                 │ pairwise precedence matrix P[i,j]
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │  STAGE 3: Orderer + Verifier                 │
                         │  topological/tournament sort (order_events)  │
                         │  + cycle resolution + rationale consistency   │
                         └─────────────────────────────────────────────┘
```

**Why pointwise (not listwise/setwise)?** It is the closest paradigm to the existing
DeBERTa cross-encoder, it is the cheapest for a sub-1B model, and its calibrated per-pair
scores feed directly into `order_events.py`. Listwise needs long context the 0.6B model
won't handle well. TFRank is explicitly a pointwise think-free design (`03`).

**Primary recommendation:** Qwen3-0.6B reasoner (path A).
**Secondary / efficiency baseline & ablation:** extend the existing **`ReasoningClassifier`** (DeBERTa-v3 cross-encoder + distilled-rationale auxiliary head, CosineEmbeddingLoss) — no test-time reasoning but very fast (path B). Run both; report the accuracy/latency trade-off.

---

## 2. Task framing (label scheme)

Cast each pair as **directional entailment** (`04`):

| Label | Meaning |
|---|---|
| `PRECEDES` | text_1 causally enables/precedes text_2 |
| `FOLLOWS` | text_2 precedes text_1 (reverse) |
| `NEUTRAL` | no causal precedence (co-occurring / unrelated) |

The current binary `label∈{0,1}` maps to PRECEDES/FOLLOWS; add NEUTRAL from hard negatives.
Output format: `<think> ...causal reasoning... </think><answer>PRECEDES</answer>` plus a
0–4 graded relevance head (TFRank expected-value score) for calibrated ranking.

---

## 3. Training pipeline (distillation-first → GRPO)

This is the consensus recipe across `02`, `04`, `05`. Four steps:

**Step 1 — Bootstrap rationales (offline, teacher LLM).**
We have labels but no gold reasoning. Use a large teacher (e.g., a hosted 70B-class or
DeepSeek-R1-class model) to generate a causal CoT for each `(t1, t2, gold_label)` using a
**CausalCoT-style** prompt. **Rejection-sample / rationalize**: keep a trace only if the
teacher's stated answer matches the gold label (STaR/RFT/COLLATE-likelihood filtering),
optionally validated for logical consistency by a **VeriCoT-style** checker (`04`, `05`).
Output: SFT dataset of `(t1, t2) → <think>rationale</think><answer>label</answer>`.

**Step 2 — SFT the sub-1B student.**
Fine-tune Qwen3-0.6B on the bootstrapped traces with the **TFRank "think-mode switch"** so
it can run think-free at inference. Multi-task: direction classification + graded score.
(Distilling-Step-by-Step precedent: a 770M model beat 540B PaLM with a rationale aux objective — `05`.)

**Step 3 — GRPO polish (short).**
Refine with GRPO (`02`): group size 8, KL to the SFT reference, **Dr.GRPO/DAPO** fixes for
length bias and entropy collapse. **Verifiable reward:**
- Direction: `+1` iff format valid **and** answer == gold label, else `0` (Rank-R1 precedent).
- Ordering (sequence-level reward on sampled workflows): **Kendall-τ / NDCG** vs gold order (REARANK / Rank-GRPO precedent).
Tooling: TRL `GRPOTrainer` + vLLM sampling; LoRA r=16 first, full-FT feasible on 4×A100.

**Step 4 — Aggregate to ordering.**
Build pairwise precedence matrix → topological sort (Kahn) with tournament tie-break
(`order_events.py`); the Verifier agent resolves cycles by dropping the lowest-confidence edge.

---

## 4. Multi-agent system design (one agent per task)

Per your request, the production/inference system is decomposed into specialized agents.
These are *runtime/pipeline* roles (not the research sub-agents), several of which are also
reused offline to build training data.

| Agent | Task | Backed by | Used at |
|---|---|---|---|
| **Retriever Agent** | Fetch top-k causally-related candidates for an anchor step | Bi-encoder (BGE-large) | inference |
| **Causal Reasoner Agent** | Pairwise: reason → PRECEDES/FOLLOWS/NEUTRAL + score | **Qwen3-0.6B** (this project) | inference |
| **Ordering Agent** | Aggregate pairwise matrix → workflow order; resolve ties | `order_events.py` (topo/tournament) | inference |
| **Verifier / Critic Agent** | Check rationale logical consistency (VeriCoT-style); reject bad reasoning; break cycles | small NLI checker / rules | inference **+** offline data filtering |
| **Rationale Bootstrapper Agent** | Generate + reject-sample teacher CoT for SFT data | large teacher LLM | offline (training only) |
| **Orchestrator** | Route a workflow-generation/reranking request through the above | thin controller | inference |

The Verifier doubles as the Step-1 distillation filter — the same logical-consistency check
that cleans training data also guards inference.

---

## 5. Evaluation protocol

**Pre-req (do first):** resolve the DeBERTa-CE premise — run `score_milan_biencoder.py` for
DeBERTa-CE vs all-mpnet/BGE-large bi-encoders on the milan + AJO ordering sets and record
numbers (`01` §4). Lock the stage-1 retriever from that.

| Level | Metric | Datasets |
|---|---|---|
| Pairwise direction | Accuracy, F1, AUC, per-tier-pair acc | `aep_causal_cls34`, `*_hard_neg_semantic` |
| Retrieval / rerank | MRR, Recall@{1,5}, P@1 | GNN-study pool (562 pairs) |
| **Workflow ordering** | **Exact-order acc, Kendall-τ, pairwise acc** | `test_samples_milan*`, `ajo_orchestrated_*` (target: beat 0/30) |
| Efficiency | latency / throughput, params (<1B check) | all |

Ablations: (a) no-reasoning vs think-free vs full-think; (b) SFT-only vs SFT+GRPO;
(c) path A (Qwen3-0.6B) vs path B (DeBERTa + rationale head); (d) binary vs 3-way (NEUTRAL) labels.

---

## 6. Phased roadmap

| Phase | Goal | Key actions | Exit criterion |
|---|---|---|---|
| **P0 — Verify & set up** | Trustworthy baseline | Resolve DeBERTa-CE premise; lock stage-1 retriever; build 3-way label set incl. NEUTRAL | Baseline ordering number on milan/AJO recorded |
| **P1 — Rationale bootstrap** | SFT data | Teacher CoT + reject-sampling + VeriCoT filter on cls34/hard-neg | ≥80% of pairs have a label-consistent trace |
| **P2 — SFT student** | Sub-1B reasoner v1 | Fine-tune Qwen3-0.6B (think-mode switch); also DeBERTa rationale-head baseline | Pairwise F1 ≥ all-mpnet bi-encoder (0.88 on cls34) |
| **P3 — GRPO polish** | Reasoning gains | GRPO w/ direction + Kendall-τ reward; Dr.GRPO/DAPO fixes | Ordering Kendall-τ and exact-order acc improve over P2 |
| **P4 — Multi-agent assembly** | End-to-end | Wire Retriever→Reasoner→Orderer→Verifier; eval on milan/AJO | **Beat 0/30** on `ajo_orchestrated`; report full table |

---

## 7. Risks & mitigations

| Risk | Evidence | Mitigation |
|---|---|---|
| Sub-1B too weak to learn reasoning via RL alone | DeepSeek-R1-Distill >> from-scratch RL at small scale (`02`,`05`) | **Distill first**, GRPO only to polish |
| Causal direction is intrinsically hard | GPT-4 F1≈29 on Corr2Cause; supervised caps ~0.79–0.86 F1 (`04`) | Reasoning traces (CausalCoT +8.37 on CLADDER); strong hard negatives; realistic targets |
| Rationale quality / hallucinated reasoning | distillation agent caught a hallucinated detail; reward hacking in GRPO | VeriCoT consistency filter; label-consistency reject-sampling; format+correctness reward only |
| Latency from generation | think-free helps but still > encoder | TFRank think-free at inference; keep DeBERTa path B as fast fallback |
| CPU OOM crashes box | user constraint | check mem before heavy ops; large single-GPU batch over small DataParallel (memory rule) |

---

## 8. Open questions for you

1. **Teacher model** for rationale bootstrapping — do we have API access to a large model (DeepSeek-R1 / 70B-class), or must the teacher also run locally on the 4×A100?
2. **Latency budget** at inference — does "think-free" suffice, or do we need the DeBERTa path B for production speed?
3. **NEUTRAL class** — is there a clean source of genuinely non-causal pairs, or do we synthesize from hard negatives?
4. Confirm **Qwen3-0.6B** as the student (vs Qwen2.5-0.5B / SmolLM2-360M) — Qwen3-0.6B has the strongest reasoning prior under 1B and a TFRank checkpoint precedent.

*See `02`–`05` for the cited evidence behind every claim here. Numbers flagged "verify" in
those reports should be confirmed against primary PDFs before quoting externally.*
