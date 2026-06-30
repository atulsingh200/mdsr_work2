# 08 — Best Ideas: The Most Optimal Way to Do This Task

> The deliverable. Synthesizes `01`–`07` with the two decisive facts you confirmed:
> (1) the **DeBERTa-v3-large cross-encoder is the best model on the real metric** (21/30 ordering
> vs 0/30 bi-encoder), and (2) the data is **positional within-doc, adjacent-only, no NEUTRAL**.
> Ideas are ranked by **return-on-investment** (impact ÷ cost/risk). Everything is < 1B params.

---

## The reframe: this is the *sentence-ordering* problem, and it has three sub-problems

Your task = the well-studied **sentence-ordering / step-ordering** NLP task. Your current
pipeline (BERT/CE pairwise → topological sort) is literally **B-TSort** (ACL 2020). The
research says the optimal system attacks **three** distinct sub-problems — and your 0/30 is
caused mostly by #2 and #3, *not* by the model:

1. **The pairwise model** — judge "does step A causally precede B?" (CE already strong: 21/30).
2. **The data** — what pairs/labels the model is trained on (currently adjacent-only, total-order).
3. **The aggregation** — turning noisy pairwise scores into a global order (currently naive topo-sort, which **cannot survive a single cycle** → 0/30).

> **Headline:** the cheapest, highest-impact wins are in **#2 and #3** and need *no new model*.
> The "add reasoning + RL" work (#1) raises the ceiling on top of that.

---

## Tier 1 — Do these first (highest ROI, low cost, no new model)

### Idea 1 — Robust rank aggregation instead of naive topological sort  ⭐ biggest single win
**What:** Replace `order_events.py`'s tournament/topo-sort with **calibrated → weighted
Minimum-Feedback-Arc-Set / Kemeny ("minimum-violations") ranking**.
**Why:** Topo-sort fails on *any* cycle (A>B>C>A) in the predicted graph — and noisy pairwise
scores always produce cycles, so the global order collapses (your 0/30). MFAS/Kemeny finds the
order that *violates the fewest (confidence-weighted) pairwise judgments* and degrades
gracefully (`07`). This is the single most likely fix for the 0/30 → many-correct jump.
**How:** edges weighted by **log-odds** of the pairwise score; solve with the linear-time
**Eades–Lin–Smyth** heuristic or exact ILP for small workflows (3–12 steps is tiny). Use
**`python-igraph`** (`feedback_arc_set`) — *not* NetworkX (it has no FAS function). Libs: `igraph`, `choix` (Bradley-Terry), `scipy`.
**Cost/risk:** ~1 day, pure post-processing, no retraining. **Do this first.**

### Idea 2 — Calibrate the pairwise scores before aggregating
**What:** Temperature-scale the CE logits on the val set so scores are probabilities.
**Why:** Confidence-weighted aggregation, thresholding, and abstention only behave if scores
are calibrated; temperature scaling preserves arg-max so it can't hurt pairwise accuracy (`07`).
**Cost/risk:** trivial (one scalar). Pairs with Idea 1.

### Idea 3 — Train on the full within-doc transitive closure (drop `MAX_GAP`)  ⭐ fixes the distribution shift
**What:** In `extract_workflows.py` / `build_workflow_order.py`, emit **all `i<j` pairs** within
a document, not just `gap ≤ 2`/`4`.
**Why:** Positional order is **transitive**, so every `i<j` pair is a *valid* label — yet the model
is trained only on **adjacent** steps and then tested on **all** pairs of a full workflow. That
distribution shift is a core reason per-pair accuracy doesn't translate to whole-sequence
order (`01` §4b). Add **distance as a feature/auxiliary target** (curriculum: easy far-apart →
hard adjacent) to sharpen long-range judgments.
**Cost/risk:** a few lines + a retrain. Likely the biggest *training-side* win.

### Idea 4 — Evaluate as a partial order, not an exact chain
**What:** Report **tie-aware Kendall-τ (τ-b/τ-c)** and **precedence-pair precision/recall**, not
just exact-permutation accuracy.
**Why:** Real workflows have parallel steps; exact-chain matching punishes correct partial
orders and hides real progress (`07`). Also gives a usable RL reward (Idea 8).
**Cost/risk:** metric-only.

---

## Tier 2 — The reasoning model (your core ask), built on the CE that already wins

> Naive chain-of-thought **degrades** pointwise rerankers (overconfidence, lost partial-relevance
> discrimination — `03`/web). So for the pointwise causal scorer, prefer **internalized /
> "think-free" reasoning**, and reserve explicit token-reasoning for a generative variant.

### Idea 5 — Distill causal rationales *into* the DeBERTa cross-encoder (think-free)  ⭐ recommended primary
**What:** Keep the winning DeBERTa-v3 CE. Add an **auxiliary rationale objective**: a teacher LLM
writes a short causal explanation per pair (filtered to agree with the gold label); the CE learns
to predict direction **and** to match the rationale representation. This **extends the team's
existing `ReasoningClassifier`** (frozen-encoder explanation embedding + CosineEmbeddingLoss).
**Why:** Genuine reasoning signal, **zero inference-latency cost**, no generation, keeps the
architecture that already scores 21/30. Strongly supported: TFRank "think-free", *Distilling
Step-by-Step* (770M T5 > 540B PaLM with a rationale aux head), R1-Distill > small-model RL
(`05`). This is the safest high-ceiling move.
**Cost/risk:** medium — needs teacher rationales (Idea 7) + a retrain. Low architectural risk.

### Idea 6 — Sub-1B *generative* pointwise reasoner (Qwen3-0.6B, TFRank-style)  — higher ceiling
**What:** Train **Qwen3-0.6B** as a pointwise causal-precedence scorer with a **think-mode
switch**: reasons during training, runs **think-free at inference**, can emit a `<think>` trace on
hard pairs. Output `PRECEDES / FOLLOWS / NEUTRAL` + graded 0–4 score.
**Why:** This is the genuine "reason-then-answer" model under 1B, and **TFRank ships a 0.6B
checkpoint** as existence proof; a generative pointwise model keeps the cross-attention that
makes the CE robust off-distribution (`03`). Frame as **directional entailment** (`04`).
**Cost/risk:** higher (new model, generation). Run as the **ceiling-pushing track** alongside Idea 5; pick the winner on the latency/accuracy curve.

### Idea 7 — Bootstrap rationales + a NEUTRAL class (enables Ideas 5, 6, 8)
**What:** (a) Teacher LLM generates causal CoT per pair, **reject-sampled** to match the gold
label (STaR/RFT/COLLATE-likelihood filtering), optionally **VeriCoT-validated** for logical
consistency (`04`,`05`). (b) **Mine NEUTRAL pairs** you don't have today: cross-document
unrelated steps, and same-doc steps under sibling/parallel headings whose order is
swap-invariant. This turns the forced total order into a **DAG/partial order**.
**Why:** You confirmed you only have causal/anti-causal pairs; NEUTRAL is what lets the orderer
produce antichains instead of hallucinating edges between parallel steps (`01` §4b, `07`).
**Cost/risk:** medium (teacher calls, mining heuristics). Unlocks the whole reasoning track.

---

## Tier 3 — Push the ceiling (global objective + RL)

### Idea 8 — GRPO polish with an *ordering* reward (the RL you wanted)
**What:** After SFT/distillation, a **short** GRPO run. Reward = format + **direction
correctness** (Rank-R1 binary reward) for pairs, and **Kendall-τ / NDCG** vs gold order for
sampled whole workflows (REARANK / Rank-GRPO precedent). Use **Dr.GRPO / DAPO** fixes
(length bias, entropy collapse); TRL `GRPOTrainer` + vLLM; LoRA r=16 first (`02`).
**Why:** Directly optimizes the metric you care about; the literature is unanimous that for
sub-1B you must **distill first, then polish with RL** — not RL from scratch (`02`,`05`).
**Cost/risk:** higher; only meaningful after Ideas 5–7 land. This is where the paper-style RL fits.

### Idea 9 — Listwise / global ordering to kill error-compounding
**What:** Move from pairwise-then-aggregate toward a **global** objective: **ListMLE /
Plackett-Luce NLL**, a **pointer/seq2seq decoder**, or a **Set-Encoder-style listwise
cross-encoder** (permutation-invariant inter-passage attention) that scores all steps jointly.
**Why:** O(n²) independent pairwise decisions compound — one wrong high-leverage edge
reorders everything (explains 21/30-pairwise → 0/30-sequence). Listwise models (RankTxNet,
BERSON) beat pairwise-then-topo-sort in the sentence-ordering literature (`07`). A listwise
**cross-encoder** keeps the cross-attention that's winning for you.
**Cost/risk:** highest; longer-term upgrade after Tier 1–2 prove out.

---

## The single recommended system (combining the best)

```
Stage 1  Retriever (existing bi-encoder, BGE-large) ─ optional, only if candidate set is large
Stage 2  Causal Reasoner (PRIMARY: DeBERTa-v3 CE + distilled rationale head [Idea 5];
                          TRACK B: Qwen3-0.6B think-free generative [Idea 6])
            → calibrated pairwise P(PRECEDES) [Idea 2], 3-way incl. NEUTRAL [Idea 7]
Stage 3  Orderer = weighted MFAS/Kemeny over log-odds edges [Idea 1],
            low-confidence pairs → antichains (DAG) [Idea 7]
Stage 4  Verifier (VeriCoT-style) breaks residual cycles + filters training rationales
Eval     tie-aware Kendall-τ + precedence P/R [Idea 4]; later GRPO τ-reward [Idea 8]
```

Training order: **distill rationales [7] → SFT CE/0.6B [5/6] → (optional) GRPO τ-reward [8]**;
data upgraded to **full transitive closure + NEUTRAL [3/7]**; aggregation upgraded to
**calibrated Kemeny [1/2]**. Longer term, fold in **listwise [9]**.

---

## What to do in the first week (no training required for most of the gain)

1. **Idea 1 + 2** — swap topo-sort for calibrated weighted-MFAS (`igraph`). *Re-score the existing DeBERTa-CE outputs on the milan/AJO sets — this alone should move 0/30 sharply.*
2. **Idea 4** — add Kendall-τ + precedence-P/R to the eval harness.
3. **Idea 3** — drop `MAX_GAP`, regenerate full-transitive-closure pairs, retrain the CE.
4. Measure. *Then* invest in the reasoning track (Ideas 5–8).

> Rationale: Ideas 1–4 are cheap and target the actual failure (aggregation + data shift). They
> de-risk the expensive reasoning/RL work and give a clean baseline to measure reasoning gains against.

---

## Honest expectation-setting

Causal **direction** is hard even for frontier models (GPT-4 ≈ F1 29 on Corr2Cause; supervised
caps ~0.79–0.86 F1 — `04`). Reasoning + distillation is precisely the lever shown to help
(CausalCoT +8.37 on CLADDER), but don't expect a perfect orderer. The realistic, defensible win
is: **fix aggregation + data (Tier 1) to convert your 21/30 pairwise strength into a much higher
ordering score, then use distilled reasoning + a τ-reward GRPO polish (Tier 2–3) to push further.**

*All quantitative claims trace to `02`–`07`; items flagged "verify" there should be confirmed
against primary PDFs before external use.*
