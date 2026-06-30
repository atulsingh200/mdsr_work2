# Reasoning-Based Reranking with LLMs: State of the Art and What Is Feasible Sub-1B

*Technical research report. Downstream goal: given an anchor step, rerank candidate next-steps by **causal precedence**, and order multi-step workflows. Current team stack: bi-encoders (best `all-mpnet-base-v2`, `BGE-large`) + a `DeBERTa-v3-large` cross-encoder classifier.*

*Compiled 2026-06-30. Every non-obvious quantitative claim carries an inline source URL. Where a source could not be fully verified, this is stated explicitly.*

---

## Executive summary

- The 2024–2025 wave of "reasoning rerankers" applies the DeepSeek-R1 RL recipe — **GRPO with `<think>`/`<answer>` formatting and a verifiable relevance reward** — to LLM rerankers. The headline finding across the literature is that **RL plus a tiny amount of relevance-labeled data** (Rank-R1: ~18% of MS MARCO; REARANK: ~179 seed queries) matches or beats supervised fine-tuning and is competitive with GPT-4, especially on *reasoning-intensive* queries (the BRIGHT benchmark). Sources: [Rank-R1, arXiv:2503.06034](https://arxiv.org/abs/2503.06034); [REARANK, arXiv:2505.20046](https://arxiv.org/html/2505.20046v1).
- **Explicit chain-of-thought (CoT) at inference is the dominant cost driver** and is not always worth it. The most relevant result for our constraints is **TFRank** ([arXiv:2508.09539](https://arxiv.org/abs/2508.09539)): it trains *with* reasoning but uses a **"think-mode switch"** to emit only a pointwise score at inference (no reasoning chain). A **1.7B** TFRank reportedly competes with 7B reasoning rerankers on BRIGHT, with order-of-magnitude faster inference. This is the closest published analogue to a sub-1B, cross-encoder-style scorer.
- Independent work corroborates that CoT is often unhelpful for ranking ("Rethinking Reasoning in Document Ranking," [arXiv:2510.08985](https://arxiv.org/pdf/2510.08985)), reinforcing that a sub-1B model should **internalize** reasoning rather than emit it.
- **Paradigm matters for our downstream ordering step.** Pointwise produces independent, calibratable scores (O(N) calls) that feed naturally into a topological/tournament ordering; pairwise gives the strongest relative judgments but costs up to O(N²); listwise is cheap per-list but order-biased and context-limited; setwise is the efficiency-tuned middle. For workflow ordering from causal-precedence judgments, a **pairwise reasoning judge feeding a tournament/topological sort**, or a **calibrated pointwise scorer**, are the two viable shapes sub-1B.
- **Realistic sub-1B recommendation:** distill/RL a **Qwen3-0.6B/1.7B** into a **TFRank-style think-free pointwise scorer** for candidate scoring, and add a **lightweight pairwise causal-precedence head** (or pairwise prompt) whose outputs are aggregated by topological sort for workflow ordering. TFRank already ships a **0.6B checkpoint** ([JOHNNY-fans/TFRank](https://github.com/JOHNNY-fans/TFRank)), making sub-1B concretely demonstrated rather than speculative.

---

## 1. Rank-R1 — GRPO setwise reasoning reranker (arXiv:2503.06034)

**Source:** [abstract](https://arxiv.org/abs/2503.06034) · [full text v1](https://arxiv.org/html/2503.06034v1).

**Idea.** Take the **Setwise** prompt (present a query + a set of candidate documents, ask the model to pick the most relevant) and add a reasoning instruction, then train the reasoning ability purely with RL — using only **relevance labels, no reasoning supervision**.

- **Base models:** instruction-tuned **Qwen2.5 at 3B, 7B, and 14B** ([html v1](https://arxiv.org/html/2503.06034v1)).
- **RL algorithm:** **GRPO** (Group Relative Policy Optimization), adapting the DeepSeek-R1 recipe. Advantage is group-relative z-score, `A_i = (r_i − mean(r)) / std(r)`, over a group of sampled generations ([html v1](https://arxiv.org/html/2503.06034v1)).
- **Output format:** `<think> reasoning </think> <answer> answer </answer>`, where the answer is the **label of the most relevant document, in square brackets** ([html v1](https://arxiv.org/html/2503.06034v1)).
- **Reward (binary, verifiable):** reward = 1 **iff** the generation matches the required `<think>`/`<answer>` format **and** the answer label matches the ground-truth relevant document; otherwise 0 ([html v1](https://arxiv.org/html/2503.06034v1)).
- **Training data:** MS MARCO passage ranking. The key efficiency claim: GRPO is trained on **~18% of the ~400k examples** used for the SFT baseline ([html v1](https://arxiv.org/html/2503.06034v1)).
- **Results (NDCG@10), reported in the paper:**
  - TREC DL19/DL20 (in-domain): Rank-R1-7B GRPO **.727 / .685**; Rank-R1-14B GRPO **.714 / .691**; Setwise-7B SFT baseline **.738 / .692** — i.e., GRPO ≈ SFT at ~18% of the data.
  - BRIGHT (out-of-domain, reasoning-intensive): Rank-R1-14B GRPO **.205** avg vs RankGPT-4 zero-shot **.170** and Setwise-14B SFT **.167**.
  - (Source for all: [html v1](https://arxiv.org/html/2503.06034v1).)
- **Two settings:** *Zeroshot* (Setwise prompt, no training) and *GRPO* (trained). GRPO consistently improves over zero-shot, most strongly on complex queries ([html v1](https://arxiv.org/html/2503.06034v1)).

**Relevance to us.** Demonstrates the core recipe — setwise selection + `<think>/<answer>` + binary reward + GRPO — and that **RL needs little labeled data**. But it is a 3B–14B model and emits reasoning at inference (slow). The setwise "pick-the-most-relevant" objective is directly reusable for "pick the most likely next-step," and its binary reward maps cleanly to a causal-precedence label.

---

## 2. REARANK — listwise reasoning agent, ~179 samples (EMNLP 2025)

**Source:** [ACL Anthology 2025.emnlp-main.125](https://aclanthology.org/2025.emnlp-main.125/) ([PDF](https://aclanthology.org/2025.emnlp-main.125.pdf)) · arXiv mirror [2505.20046](https://arxiv.org/html/2505.20046v1). (The ACL PDF text layer was not machine-extractable; numbers below were taken from the arXiv HTML of the same paper and corroborated across two extraction passes — treat as arXiv-verified.)

- **Base model:** **Qwen2.5-7B-Instruct**, yielding **REARANK-7B** (HF: [`le723z/Rearank-7B`](https://huggingface.co/le723z/Rearank-7B)).
- **Paradigm:** **listwise** sliding-window reranking with **explicit reasoning before ranking**. Window size **20**, step **10**, so ~**10 LLM calls to rerank a top-100 list** ([arXiv html](https://arxiv.org/html/2505.20046v1)).
- **Output format:** `<think>…</think>` then `<answer>…</answer>` containing a permutation in bracket-identifier descending order, e.g. `[3] > [1] > [2]` ([arXiv html](https://arxiv.org/html/2505.20046v1)).
- **RL:** **GRPO** (DeepSeek-R1-style), 32 rollouts/step, batch 128, ~160 steps, 8×H100; group-relative advantage with a fixed reference policy for KL ([arXiv html](https://arxiv.org/html/2505.20046v1)). (Exact KL coefficient not stated in the source.)
- **Data efficiency:** only **179 annotated MS MARCO queries** as seed; a sampling/augmentation pipeline (~20 BM25 passages per query, ~50× repetition, filtering out queries with NDCG@10 < 0.1 or no relevant passage) expands to ~**12k training instances** ([arXiv html](https://arxiv.org/html/2505.20046v1)).
- **Reward (composite):** `r = 0.8·r_rank + 0.1·r_format1 + 0.1·r_format2`, where `r_rank` is a min-max-normalized NDCG@10 *improvement* `(NDCG_rerank − NDCG_init)/(NDCG_optimal − NDCG_init)`, and the two format terms check tag presence and `[X] > [Y]` formatting ([arXiv html](https://arxiv.org/html/2505.20046v1)).
- **Results (NDCG@10):** TREC DL19/DL20 **74.16 / 70.00** (vs RankGPT-4 75.59 / 70.56, RankZephyr-7B 73.90 / 70.60, Rank-R1-7B 72.70 / 68.50); BEIR avg **54.59** (GPT-4 55.84); **BRIGHT avg 17.7 — surpassing GPT-4's 16.8** ([arXiv html](https://arxiv.org/html/2505.20046v1)). Reported aggregate gains over the RankQwen2.5-7B baseline: **+6.5% in-domain, +4.5% out-of-domain** ([arXiv html](https://arxiv.org/html/2505.20046v1)).

**Relevance to us.** Proves a **listwise reasoning reranker can match GPT-4 from ~179 labeled queries**, and that an NDCG-improvement reward works well. Directly applicable to **workflow ordering** (it outputs a full permutation, which is exactly a step ordering). The catch: 7B and reasoning-at-inference; a listwise permutation also doesn't yield per-pair causal scores, only a global order. For multi-step workflow ordering, REARANK's permutation output is conceptually the right shape, but it would need shrinking and a precedence-aware reward.

---

## 3. TFRank — "think-free" pointwise ranking, 1.7B (arXiv:2508.09539) — *most relevant to us*

**Source:** [abstract](https://arxiv.org/abs/2508.09539) · [full text v1](https://arxiv.org/html/2508.09539v1) · [code & checkpoints](https://github.com/JOHNNY-fans/TFRank) · [secondary summary](https://www.emergentmind.com/topics/tfrank).

**Core thesis.** Reasoning-intensive rankers usually need large LLMs **and** explicit CoT at inference → high cost/latency. TFRank instead **trains with reasoning but switches reasoning off at inference**, delivering a precise pointwise relevance score with **no reasoning chain emitted**, so it can run small and fast ([abstract](https://arxiv.org/abs/2508.09539)).

**Base models.** Built on the **Qwen3 series — checkpoints at 0.6B, 1.7B, 4B, 8B** — plus a `Qwen2.5-7B-Instruct` variant; **seven checkpoints released** on Hugging Face, SFT and GRPO variants ([repo](https://github.com/JOHNNY-fans/TFRank)). The headline efficiency claim centers on the **1.7B** model.

**Think-mode switch.** Inference modes are toggled (Qwen3's `/think` vs `/no_think`): during **training** the model generates explicit reasoning chains (`/think`); at **inference** it runs `/no_think` and outputs **only the relevance score**, "delivering precise relevance scores … without generating any reasoning chains" ([abstract](https://arxiv.org/abs/2508.09539); [html v1](https://arxiv.org/html/2508.09539v1)). The repo exposes this as a flag at ranker initialization ([repo](https://github.com/JOHNNY-fans/TFRank)).

**Scoring scheme (pointwise, five-level).** A **5-level ordinal scale (0–4)** with fine-grained supervision. At inference, the model exposes only the **top-20 tokens by logit**; the relevant level tokens are read off these logits to compute a continuous score, and if the target tokens are absent from the top-20 a neutral fallback of **0.5** is assigned ([html v1](https://arxiv.org/html/2508.09539v1)). The repo's examples show responses like `yes(4)` / `no(1)`, i.e. an ordinal verbalization underlying a normalized 0–1 score ([repo](https://github.com/JOHNNY-fans/TFRank)). This is effectively an **expected-value-over-levels** pointwise score — the same role our DeBERTa cross-encoder plays, but reasoning-distilled.

**Training data.** CoT data distilled from **MS MARCO, M3 data, and DeepSeek-R1** (the repo also lists **Rank1** as a source) ([html v1](https://arxiv.org/html/2508.09539v1); [repo](https://github.com/JOHNNY-fans/TFRank)). Reported splits: **`ms_sub_binary` ≈ 195,335 queries / 385,791 samples** and **`ms_sub_finegrained` ≈ 7,068 queries** with fine-grained annotations spanning **pointwise, pairwise, and listwise** formats ([html v1](https://arxiv.org/html/2508.09539v1)).

**Training objective.** **Multi-task supervised fine-tuning** combining pointwise, pairwise, and listwise tasks plus CoT distillation, under a standard autoregressive LM loss `L_SFT = −log P(T|C)`; a GRPO-trained variant is also released ([secondary summary](https://www.emergentmind.com/topics/tfrank); [repo](https://github.com/JOHNNY-fans/TFRank)). (No separate explicit listwise/KL loss term was confirmed in the accessible text beyond the multi-task framing.)

**Results & efficiency.**
- **BRIGHT (reasoning-intensive):** TFRank-1.7B achieves average NDCG@10 "on par with or exceeding large baselines such as Rank1-7B and REARANK-7B" using **~4× fewer parameters**; TFRank-8B shows a statistically significant edge over REARANK-7B (paper reports `p ≪ 0.01`) ([html v1](https://arxiv.org/html/2508.09539v1); [secondary summary](https://www.emergentmind.com/topics/tfrank)).
- **BEIR:** TFRank-8B reaches **NDCG@10 ≈ 43.2** (general retrieval), described as matching/surpassing SOTA ([secondary summary](https://www.emergentmind.com/topics/tfrank)).
- **Latency:** think-free pointwise output yields an "order-of-magnitude increase in inference speed" vs CoT rerankers; the paper reports **NDCG@10 vs queries-per-hour** curves (specific QPH numbers are in the figures, not extractable as text here) ([abstract](https://arxiv.org/abs/2508.09539); [secondary summary](https://www.emergentmind.com/topics/tfrank)).

> **Caveat on exact numbers:** the precise per-dataset BRIGHT/BEIR NDCG table and the QPH throughput values live in the paper's figures/tables and could not be transcribed verbatim from the accessible HTML. The 43.2 BEIR figure comes from a secondary summary and should be confirmed against the PDF before being quoted as authoritative.

**Why this is the template for us.** TFRank is the only line of work that (a) targets **sub-1B and ~1.7B** explicitly, (b) produces a **pointwise, cross-encoder-style calibrated score** (drop-in for our DeBERTa classifier), and (c) keeps the **reasoning gains while paying zero reasoning cost at inference**. Its multi-task supervision already includes pairwise and listwise signals, which is exactly what a causal-precedence + workflow-ordering task needs.

---

## 4. Baselines: RankGPT, RankZephyr, Setwise, InsertRank

### RankGPT — listwise permutation + distillation (EMNLP 2023, arXiv:2304.09542)
[Source](https://arxiv.org/abs/2304.09542). Listwise **sliding window** (size **20**, step **10**) over top-100; each passage gets an identifier `[1] [2] …` and the model outputs a permutation `[2] > [3] > [1] …` (ordering, not scores). **Permutation distillation:** 10K MS MARCO queries × 20 BM25 candidates, ChatGPT teacher, student trained with a **RankNet pairwise loss**. Results (NDCG@10): RankGPT-gpt-3.5 DL19/DL20 **65.80 / 62.91**, BEIR avg **49.37**; RankGPT-gpt-4 **75.59 / 70.56**, BEIR avg **53.68**; a distilled **DeBERTa-v3-large (435M)** student reaches BEIR avg **53.03** and reportedly beats the prior monoT5-3B SOTA by **+1.67 NDCG** on BEIR ([arXiv:2304.09542](https://arxiv.org/abs/2304.09542)). *Note the DeBERTa-large-from-permutation-distillation result is directly relevant — it's the same backbone the team already uses.*

### RankZephyr — open 7B listwise distilled from GPT-4 (arXiv:2312.02724)
[Source](https://arxiv.org/abs/2312.02724). Base **Zephyr-β 7B (on Mistral-7B)**, listwise sliding window (20/10). **Two-stage distillation:** Stage 1 = ~100K MS MARCO queries with RankGPT-3.5 permutations; Stage 2 = up to **5K queries** with RankGPT-4 permutations. With SPLADE++ first stage, NDCG@10 DL19/DL20 = **0.7816 / 0.8159**, **matching/exceeding RankGPT-4 (0.7464 / 0.7076)** and robust to input order ([arXiv:2312.02724](https://arxiv.org/abs/2312.02724)).

### Setwise — efficiency-tuned selection paradigm (SIGIR 2024, arXiv:2310.09497)
[Source](https://arxiv.org/abs/2310.09497). Prompts the LLM to **select the single most relevant of a set of c+1 documents** per call, then plugs into **heapsort/bubblesort**, reducing sort complexity from `O(k·log₂N)` → `O(k·log_c N)` (heapsort) and `O(k·N)` → `O(k·N/(c−1))` (bubblesort). On Flan-T5-large / TREC DL19: setwise.heapsort uses **125.4 LLM inferences, 8.0 s/query**, vs pairwise.heapsort **230.3 inferences, 16.1 s** — roughly halving cost at comparable effectiveness (setwise.heapsort NDCG@10 **0.670** vs pairwise.heapsort **0.657**) ([arXiv:2310.09497](https://arxiv.org/abs/2310.09497)). This is the prompting unit that Rank-R1 builds its RL on.

### InsertRank — inject BM25 scores into the listwise prompt (arXiv:2506.14086)
[Source](https://arxiv.org/html/2506.14086v1). Listwise; **appends each candidate's BM25 lexical score to the prompt** (`… BM25 score: [score]`) and orders candidates by BM25 to ground LLM reasoning and curb "overthinking"/concept drift on reasoning-intensive queries. **No fine-tuning.** Evaluated zero-shot on GPT-4o, Gemini 2.0/2.5 Flash, Deepseek-R1, over **BRIGHT** (reasoning retrieval) and **R2MED** (medical), with BM25 as first stage. BM25 injection gives **+0.003 to +0.048 NDCG@10** on BRIGHT (largest: Gemini 2.5 Flash .293→.341, ~16.3% relative); **Deepseek-R1 + InsertRank ≈ 37.5 NDCG@10** on BRIGHT, the paper's headline ([arXiv:2506.14086](https://arxiv.org/html/2506.14086v1)). **Takeaway for us:** a cheap lexical/structural side-signal injected into the prompt measurably helps reasoning reranking — analogous to injecting workflow-position or BM25-style co-occurrence priors into a causal reranker.

---

## 5. Survey: "Large Language Models for Reranking: A Survey" (TechRxiv 176300630)

**Bibliographic (verified via Crossref — authoritative):** Yinxin Zhou, Qin Luo, Bin Feng, Bang Wang; posted **13 Nov 2025**, CC BY 4.0; DOI **10.36227/techrxiv.176300630.01740917/v1** — [DOI](https://doi.org/10.36227/techrxiv.176300630.01740917/v1) · [landing](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176300630.01740917/v1). **Caveat:** the survey body is Cloudflare-gated and was not directly readable; the abstract and metadata below are authoritative (Crossref), and taxonomy detail is from indexed snippets cross-checked against the primary papers it cites — fine-grained per-method placements are inferred and marked accordingly.

**Top-level taxonomy (two patterns, from the verbatim abstract):**
1. **Using LLMs for reranking** (prompting, no weight updates) → split into **one-stage** and **multi-stage** pipelines.
2. **Training LLMs for reranking** → **supervised fine-tuning, reinforcement learning, knowledge distillation**.

The **pointwise / pairwise / listwise / setwise** distinction is the prompting-paradigm axis inside the one-stage branch, distinguished by **how many documents enter each prompt** and the resulting **cost↔effectiveness** profile (snippet-verified):
- **Pointwise** — one query–doc pair, independent score (relevance-generation or query-generation). "High efficiency, poor effectiveness." O(N) calls; needs **score calibration**.
- **Pairwise** — two docs, "which is more relevant?"; aggregated via sorting. "Superior effectiveness but high computational overhead"; up to O(N²) (reduced by sorting variants).
- **Listwise** — whole list in, **permutation out** using global signals; few calls but **context-limited and position-biased**.
- **Setwise** — a set in, **select most relevant**; "reduces the number of LLM inferences and prompt token consumption," tunable by set size, robust to initial-retrieval variation.

**Reasoning rerankers** sit mostly under "training LLMs": **RL-based** (Rank-R1, GRPO/DeepSeek-R1 recipe — modifies the Setwise prompt to "reason first, then predict the label," matching SFT with ~18% data) and **CoT/test-time-compute** (Rank1, [arXiv:2502.18418](https://arxiv.org/abs/2502.18418), emits CoT before judging). The survey frames its **four challenge dimensions as efficiency, robustness, adaptability, and interpretability** (verbatim abstract), positioning reasoning traces as an interpretability gain.

---

## 6. Pointwise vs pairwise vs listwise vs setwise: tradeoffs for a sub-1B causal reranker

| Axis | Pointwise | Pairwise | Listwise | Setwise |
|---|---|---|---|---|
| LLM calls (N candidates) | O(N) | up to O(N²); O(N log N) with sort | O(N/window) (few) | O(N log_c N) with heapsort |
| Output | Independent score per item | Relative "A > B" per pair | Full permutation | "most relevant of set" |
| Effectiveness | Lowest | Highest | High | ≈ pairwise, cheaper |
| Score calibration | Needs calibration; **scores are directly comparable & reusable** | Relative only; needs aggregation | Order only, no scores | Selection only, no scores |
| Position/order bias | None | Low | **High** (mitigated by shuffled-aggregation) | Low–moderate |
| Fit for sub-1B reasoning | **Best** (1 forward pass, can be think-free) | Costly at inference but strongest signal | Long context hurts a small model | Good balance |
| Fit for workflow ordering | Score → sort; needs precedence framing | **Pairwise precedence → topological/tournament sort** (natural fit) | Permutation = ordering directly | Iterative "next step" selection |

Sources for the cost/effectiveness characterizations: [Setwise (arXiv:2310.09497)](https://arxiv.org/abs/2310.09497); survey abstract/snippets ([DOI](https://doi.org/10.36227/techrxiv.176300630.01740917/v1)); [Rank-R1](https://arxiv.org/html/2503.06034v1).

**Implications for the causal-precedence + workflow-ordering task:**

- **Producing scores that feed a workflow-ordering step.** Causal precedence is inherently a **relation between two steps** (does A enable B?). That maps most cleanly to a **pairwise** judgment. A pairwise score matrix `P(A→B)` can be turned into a global order by **topological sort** of the thresholded DAG, or — when the matrix has cycles/noise — by a **tournament ranking** (e.g., Copeland / sort by win-count) that is robust to inconsistent pairwise votes. This is the standard pairwise-aggregation route the survey and Setwise describe, repurposed from "more relevant" to "causally precedes."
- **Calibration favors pointwise for candidate filtering.** For the first job — rerank candidate next-steps for a fixed anchor — a **calibrated pointwise score** (TFRank's expected-value-over-levels) is directly comparable across candidates and cheap (one pass each), making it the right tool to shortlist before any pairwise work.
- **Setwise as an inference-time ordering operator.** "Pick the most relevant next step from a set," applied repeatedly (Setwise + heapsort), naturally produces an ordering while keeping calls at O(N log_c N) — attractive when N is small (typical workflows are short).
- **Listwise** is the most direct "emit the whole ordering" approach (REARANK), but a sub-1B model handling a long multi-candidate context with strong position bias is the riskiest option; if used, apply **shuffled-order aggregation** to de-bias.
- **Reasoning at inference is optional and often skippable.** TFRank's think-free result and the independent "CoT falls short for ranking" finding ([arXiv:2510.08985](https://arxiv.org/pdf/2510.08985)) both argue a small model should **internalize** reasoning (train with CoT, infer without), which is essential to hit acceptable latency sub-1B.

---

## What is realistic for a sub-1B causal reranker

**Verdict: a sub-1B reasoning-distilled reranker is feasible and demonstrated.** TFRank ships a **Qwen3-0.6B checkpoint** and reports a **1.7B model competitive with 7B reasoning rerankers** on BRIGHT ([repo](https://github.com/JOHNNY-fans/TFRank); [arXiv:2508.09539](https://arxiv.org/abs/2508.09539)). That is the existence proof; below is a concrete, implementation-oriented plan.

**Recommended architecture (two heads, one small backbone):**

1. **Think-free pointwise causal scorer (replaces/augments the DeBERTa classifier).**
   - Backbone: **Qwen3-0.6B or 1.7B**, TFRank-style. Train with `/think` CoT on causal-precedence labels (does candidate step plausibly *follow* the anchor?), infer with `/no_think` emitting a 5-level ordinal → continuous score via top-k level-token logits (TFRank's mechanism). This is a near drop-in for the current cross-encoder, but reasoning-distilled and calibrated. Sources: [TFRank html](https://arxiv.org/html/2508.09539v1).
   - Use it to **rerank candidate next-steps** for a given anchor (the primary task) — one forward pass per candidate, fast enough for online use.

2. **Pairwise causal-precedence judge → topological/tournament ordering (for full-workflow ordering).**
   - Either a pairwise prompt/head on the same backbone ("does step A causally precede step B?") producing `P(A→B)`, aggregated by **topological sort** (or Copeland tournament when noisy). This is the natural shape for ordering and avoids the position bias of long listwise contexts.
   - Restrict pairwise calls to the **shortlist** from head (1), keeping O(k²) with small k.

**Training recipe (data-efficient, RL-optional):**
- **Distillation first (cheapest, most reliable sub-1B):** generate CoT + ordinal labels from a strong teacher (a large reasoning model) on your AEP-causal workflow data, then SFT the small model multi-task (pointwise + pairwise + listwise), exactly as TFRank does ([TFRank html](https://arxiv.org/html/2508.09539v1)). RankGPT's result that a **435M DeBERTa distilled from permutations beat monoT5-3B** ([arXiv:2304.09542](https://arxiv.org/abs/2304.09542)) shows distillation alone goes a long way at <1B.
- **Optional GRPO polish:** if labels are scarce, add a GRPO stage with a **verifiable reward** — Rank-R1's binary "correct-next-step" reward for the pointwise/setwise head, or REARANK's **NDCG-improvement reward** for the ordering head — both shown to work from **tiny label sets** (Rank-R1 ~18% data; REARANK ~179 seed queries) ([Rank-R1](https://arxiv.org/html/2503.06034v1); [REARANK](https://arxiv.org/html/2505.20046v1)). Note GRPO at 7B used 8×H100; sub-1B GRPO is far cheaper but still heavier than SFT, so treat it as a second phase.
- **Inject a cheap side-signal (InsertRank lesson):** add the bi-encoder similarity (your `all-mpnet`/`BGE-large`) or a co-occurrence/temporal prior into the prompt/features, mirroring InsertRank's BM25 injection, which gave measurable gains for free ([arXiv:2506.14086](https://arxiv.org/html/2506.14086v1)).

**Efficiency expectation.** Keep reasoning **train-time only** (think-free at inference). TFRank reports order-of-magnitude speedups from this alone ([arXiv:2508.09539](https://arxiv.org/abs/2508.09539)); a 0.6–1.7B think-free pointwise scorer is in the same latency class as the current DeBERTa-v3-large cross-encoder (~435M) while adding reasoning-distilled accuracy.

**Honest limitations / open risks.**
- No published reranker *specifically* targets **causal precedence**; the above adapts relevance-reranking recipes. The precedence reward/label design is novel work the team must build and validate.
- TFRank's exact sub-1B (0.6B) BRIGHT numbers and throughput were not transcribable from the accessible text (figures only) — confirm against the PDF before committing to a 0.6B vs 1.7B choice.
- Listwise/setwise position bias and pairwise cycle-handling (topological sort on a non-DAG) are real engineering risks for the ordering step; tournament aggregation mitigates but does not eliminate them.

---

## References

- Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning — https://arxiv.org/abs/2503.06034 · full text https://arxiv.org/html/2503.06034v1
- REARANK: Reasoning Re-ranking Agent via Reinforcement Learning (EMNLP 2025) — https://aclanthology.org/2025.emnlp-main.125/ · PDF https://aclanthology.org/2025.emnlp-main.125.pdf · arXiv https://arxiv.org/abs/2505.20046 (html https://arxiv.org/html/2505.20046v1) · model https://huggingface.co/le723z/Rearank-7B
- TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking — https://arxiv.org/abs/2508.09539 · full text https://arxiv.org/html/2508.09539v1 · code https://github.com/JOHNNY-fans/TFRank · summary https://www.emergentmind.com/topics/tfrank
- RankGPT: Is ChatGPT Good at Search? (EMNLP 2023) — https://arxiv.org/abs/2304.09542
- RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! — https://arxiv.org/abs/2312.02724
- Setwise: A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with LLMs (SIGIR 2024) — https://arxiv.org/abs/2310.09497
- InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking — https://arxiv.org/abs/2506.14086 (html https://arxiv.org/html/2506.14086v1)
- Rank1: Test-Time Compute for Reranking — https://arxiv.org/abs/2502.18418
- Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short — https://arxiv.org/pdf/2510.08985
- Large Language Models for Reranking: A Survey (Zhou, Luo, Feng, Wang; TechRxiv, 13 Nov 2025) — https://doi.org/10.36227/techrxiv.176300630.01740917/v1 (body Cloudflare-gated; metadata via Crossref https://api.crossref.org/works/10.36227/techrxiv.176300630.01740917/v1)
