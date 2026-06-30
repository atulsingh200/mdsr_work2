# Research: Adding Reasoning to a Sub-1B Causal Model for AEP/AJO Workflows

This folder collects the research and the implementation plan for adding **reasoning**
(reasoning-then-answer, trained with RL and/or distillation) to a **sub-1B-parameter**
model for causal-direction prediction, reranking, and workflow ordering.

**Goal:** beat the current bi-encoder baselines on (1) pairwise causal direction,
(2) reranking MRR, and especially (3) **end-to-end workflow ordering** (currently 0/30),
using a model **< 1B parameters**.

## Reading order

| File | What it covers | Author |
|---|---|---|
| `01_project_context.md` | Current architectures, baseline metrics, data, constraints, and a correction to the "DeBERTa-CE is best" premise. **Start here.** | grounded in repo |
| `02_rl_methods_sub1b.md` | RL for reasoning in sub-1B models: GRPO + variants (Dr.GRPO, DAPO, S-GRPO), PPO/DPO, distillation-then-RL verdict, reward design, practical recipe. | research agent |
| `03_reasoning_rerankers.md` | SOTA reasoning rerankers: Rank-R1, REARANK, TFRank, RankZephyr/Setwise/InsertRank; pointwise vs listwise vs setwise; what's realistic sub-1B. | research agent |
| `04_entailment_causal_reasoning.md` | NLI/entailment SOTA, e-SNLI/VeriCoT/Atomic-SNLI, causal-direction datasets (COPA/e-CARE), framing causal precedence as reasoning/entailment. | research agent |
| `05_reasoning_distillation.md` | Teaching small models to reason via distillation: COLLATE (the linked paper), Distilling Step-by-Step, R1-Distill, STaR/RFT, think-free internalization, bootstrap-rationales→SFT→GRPO. | research agent |
| `06_proposal_and_roadmap.md` | Architecture proposal, multi-agent system design, training pipeline, eval protocol, milestones, risks. | synthesis |
| `07_rank_aggregation_ordering.md` | **The fix for 0/30:** turning noisy pairwise scores into a global order — Bradley-Terry, Kemeny/MFAS, B-TSort, listwise losses, calibration, partial-order metrics. | research agent |
| `08_best_ideas.md` | **★ THE DELIVERABLE ★** — the ranked, concrete "best possible ideas" tailored to your confirmed facts (CE wins ordering 21/30; positional within-doc adjacent-only data; no NEUTRAL; sub-1B). **Read this.** | synthesis |

## TL;DR (see `08` for the full ranked argument)

- This is the **sentence-ordering** problem; your pairwise→topo-sort pipeline is literally **B-TSort**. The 0/30 is caused mostly by **aggregation + data**, not the model.
- **DeBERTa-CE is the right backbone** (21/30 ordering vs bi-encoder 0/30) — classification accuracy ≠ ordering quality.
- **Tier-1 wins need no new model:** (1) replace naive topo-sort with **calibrated weighted Kemeny/MFAS** (igraph); (2) **calibrate** scores; (3) train on **full within-doc transitive closure** (drop `MAX_GAP`); (4) **eval as a partial order** (Kendall-τ).
- **Reasoning model:** distill rationales **into the CE** (think-free, no latency) as primary; Qwen3-0.6B think-free generative reasoner as the ceiling track. Naive CoT *hurts* pointwise — internalize it.
- Sub-1B ⇒ **distill first, short GRPO polish** with a **Kendall-τ ordering reward**. Add a **NEUTRAL** class so the order is a DAG, not a forced chain.
