# 01 — Project Context: Causal Embedding & Reasoning for AEP/AJO Workflows

> Grounding document for the `research/` effort. Written from a full sweep of the
> existing codebase at `/mnt/localssd` (June 2026). Everything here is from the
> repo's own results — it is the baseline that any new sub-1B reasoning model must beat.

---

## 1. The problem

Predict **directional causality** between two pieces of Adobe Experience Platform (AEP) /
Adobe Journey Optimizer (AJO) documentation or workflow steps:

> Given `text_1` and `text_2`, estimate **P(text_1 causally precedes / enables text_2)**.

This single pairwise primitive is used for three downstream tasks:

1. **Pairwise causal direction** — binary classification (does A precede B?).
2. **Retrieval / reranking** — given an anchor step, rank candidate next-steps by causal precedence (MRR, Recall@K).
3. **End-to-end workflow ordering** — order a multi-step workflow from pairwise scores (the real goal; **currently failing — 0/30 on the AJO orchestrated test**).

**Decision (this round):** optimize research toward **all three, workflow ordering first**.

---

## 2. Current architectures in the repo

| Path | Architecture | Notes |
|---|---|---|
| `automation/internship-causal-embedding/src/classifier/model.py` | **Bi-encoder** `DirectionalClassifier` | Two untied transformers, CLS + L2-norm, fusion `[A; B; A−B; A*B]` → MLP `4d→512→128→1`. **Best-performing family.** |
| `.../classifier/crossencoder/` (`CrossEncoderClassifier`, `train_ce2x.py`) | **Cross-encoder** | Single transformer over `[CLS] t1 [SEP] t2`, CLS→Linear→logit. Truncation starves text. |
| `.../classifier/reasoning_model.py` | **ReasoningClassifier** (multi-task) | Trainable dual-encoders + **frozen encoder for explanation embeddings**; prediction head (BCE) + reasoning head (CosineEmbeddingLoss vs. explanation embedding). *Already a rationale-distillation seed.* Not deployed at scale. |
| `automation/.../biencoder/` | `BiEncoder` + `InfoNCELoss` | Retrieval/contrastive path. |
| `internship-causal-embedding-lorentz/gnn/` | `HybridCauseEffectGNN` | Directional GAT over doc graph + text. **Concluded: GNN adds no value here** (mixing gate collapsed to ~0.01–0.03) because AEP causal links are text-predictable. |

---

## 3. Baseline results (the numbers to beat)

### 3.1 Classification sweep — `baseline_sweep/` (7 datasets × 6 models, 42 runs)

Best model **per dataset** (all are **bi-encoder**; metric = accuracy / AUC / F1):

| Dataset | Best model | Acc | AUC | F1 | Difficulty |
|---|---|---|---|---|---|
| `ajo_doc_not_tier1` | all-mpnet-base-v2 | 99.4% | 0.997 | 0.994 | easiest |
| `aep_dataset` | all-mpnet-base-v2 | 93.3% | 0.965 | 0.934 | easy |
| `aep_causal_cls34` | all-mpnet-base-v2 | 88.8% | 0.946 | 0.880 | medium |
| `ajo_newstyle` | all-mpnet-base-v2 | 65.3% | 0.755 | 0.657 | **hardest (ordering)** |

Model ranking (overall): **all-mpnet-base-v2 (278M) > BGE-large (335M) > E5-large (560M) > MiniLM-L6 (22M) > BGE-base (109M) > DeBERTa-v3-large (336M)**.

### 3.2 Cross-encoder sweep — `crossencoder_sweep/` (on `aep_causal_cls34`)

| Model | Cross-encoder Acc | Bi-encoder Acc |
|---|---|---|
| all-MiniLM-L6-v2 | 78.8% | 87.4% |
| BGE-base-en-v1.5 | 81.5% | 86.1% |

→ **Bi-encoder beats cross-encoder by ~4.6%** on this data (cross-encoder text starvation from 512-token shared budget).

### 3.3 Retrieval / GNN study — `internship-causal-embedding-lorentz/gnn/` (562 test pairs)

| Model | N×N MRR | pool MRR | pool R@1 | pool R@5 | AUC |
|---|---|---|---|---|---|
| BERT BiEncoder (baseline) | 0.4033 | 0.3574 | 0.183 | 0.571 | 0.969 |
| **BGE-large e2e (no GNN)** | **0.4893** | **0.4269** | **0.244** | **0.648** | **0.973** |

→ Fine-tuned BGE-large bi-encoder: **+21.3% MRR** over BERT baseline; GNN adds nothing.

### 3.4 End-to-end workflow ordering — `/mnt/localssd`

- `ajo_orchestrated_workflows_flat_scored.json`: **0 / 30 correct** orderings currently.
- `test_samples_milan*.json`: flat + 3-step ordering tests.
- `score_milan_biencoder.py`: main scoring harness (bi-encoder / cross-encoder / AJO ordering).
- `order_events.py`: tournament ranking from pairwise scores (tie-break by net precedence).

---

## 4. ✅ Premise resolved: DeBERTa-CE is best *on ordering*, not on classification

The brief stated *"best performance is with the cross-encoder DeBERTa-v3-large."* This is
**correct for the downstream ordering task**, even though it looks wrong in the classification sweep:

- In the **classification sweep**, DeBERTa-v3-large is the **worst** of six models (~61.9% on `aep_causal_cls34`) and the cross-encoder paradigm trails the bi-encoder by ~4.6%.
- But on the **downstream workflow-ordering test** (the metric we actually care about), the **DeBERTa-v3-large cross-encoder scores 21/30** and produces nearly-correct order on a 6-step workflow — far ahead of the bi-encoder (**0/30**). *(User-confirmed, tested in a separate cross-encoder setup.)*

**Key lesson:** *pairwise classification accuracy ≠ ordering quality.* The cross-encoder's
**joint cross-attention over both texts** generalizes better to off-distribution pairs (far-apart
/ non-adjacent steps), where the bi-encoder's independent embeddings collapse. **This makes
the cross-encoder the right backbone**, and reframes the goal as *"add reasoning to a
cross-encoder"* — which means **internalized/think-free reasoning** (a DeBERTa CE cannot
generate tokens) or a **sub-1B generative model used pointwise** that keeps cross-attention.

## 4b. ⚠️ Root-cause of the ordering failure (from the data builders)

Reading `build_workflow_order.py` and `new_aep_workflow_scrap/extract_workflows.py` reveals
*why* full-ordering is hard and how to fix the data:

1. **Label = positional order within ONE document/transcript** (earlier phase/sentence ⇒ precedes), emitted as **antisymmetric forward(1)/reverse(0)** pairs. Good (kills topic shortcut) but it teaches *narration order*, which conflates true **causal necessity** with mere sequence.
2. **Pairs are same-document AND windowed**: `MAX_GAP=2` (skip-one) in `extract_workflows.py`, `=4` in `build_workflow_order.py`. The model **only ever sees adjacent steps** in training, but at eval must order **all pairs of a full workflow** (including far-apart ones) → **distribution shift**.
3. **No NEUTRAL / non-causal class** — every pair is forced a direction (a *total-order* assumption). Real workflows are **partial orders** (parallel/independent steps). The orderer (`order_events.py`) then forces a total order over noisy/cyclic pairwise scores, and **errors compound** → 0/30.
4. Phase granularity (24–56 words, 3–12 phases) is deliberately matched to the eval targets (`ajo_orchestrated_workflows_flat.json`, `test_samples_milan.json`).

**Implications for the fix (detailed in `07` + `08`):** (a) train on **non-adjacent pairs too** (full transitive closure within a doc, not just gap≤2/4) to close the distribution shift; (b) **add a NEUTRAL/abstain class** for parallel steps so the order becomes a DAG, not a forced chain; (c) replace naive topological sort with **robust rank aggregation** (Bradley-Terry / minimum-feedback-arc-set / Kemeny) that tolerates cycles; (d) consider a **listwise/global** objective so per-pair errors don't compound.

---

## 5. Data

JSONL pairs:
```json
{"text_1": "...", "text_2": "...", "label": 1,
 "tier_1": 5, "tier_2": 8, "sub_1": "...", "sub_2": "...",
 "url_1": "...", "url_2": "..."}
```
`label=1` ⇒ `text_1` precedes `text_2`. `tier_*` = workflow tier level (T1–T15); adjacent tiers are hardest.

| Dataset | Train | Val | Test |
|---|---|---|---|
| `aep_causal_classification_34` | 110.7K | 6K | 6K |
| `aep_causal_classification_hard_neg_semantic` | 261.2K | 14.7K | 6K |
| `new_aep_workflow_scrap` | 13.9K | 1.7K | 1.5K |
| `workflowllm_aep_merged_2` | 210.8K | 6K | 6K |

We have **labels but generally not gold reasoning traces** — relevant for the distillation strategy (must bootstrap rationales). Hard-negative datasets (semantic + anchor-filter) improve generalization.

---

## 6. Constraints (from the user)

- **Model size:** strictly **< 1B parameters**.
- **Hardware:** 4× A100 + ~1.01 TB CPU RAM. CPU OOM crashes the whole box → always check memory before heavy ops; prefer large single-GPU batch over small DataParallel batches.
- **Open to explore** architectures; this round is **research + plan only** (no training).

---

## 7. What this implies for the reasoning model

1. A DeBERTa cross-encoder **cannot generate `<think>` reasoning** — genuine "reason-then-answer" needs a **generative** model (or reasoning must be *internalized/distilled*, à la TFRank "think-free").
2. The sub-1B constraint means **pure GRPO from scratch is risky** (the literature suggests distillation-then-RL for small models — see `02_rl_methods_sub1b.md`).
3. The strong bi-encoder retriever should likely be **kept as stage-1**; the reasoning model is best positioned as a **stage-2 reranker / orderer** (see `03_reasoning_rerankers.md`).
4. Workflow ordering (0/30) is the clearest opportunity: a reasoning model that emits a causal rationale before a pairwise verdict, feeding `order_events.py`, is the target (see `06_proposal_and_roadmap.md`).
