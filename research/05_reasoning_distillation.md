# Teaching Small Models to Reason via Distillation
### And internalizing reasoning so a small model (or encoder) can "reason then answer" cheaply — for causal-direction reasoning + reranking

*Research report. Date: 2026-06-30. Author: research assistant.*

> **Sourcing rules followed.** Every non-obvious quantitative claim carries an inline source URL. Numbers were read from the primary papers (arXiv PDF/HTML) or official model cards. Items I could not verify from a primary source are explicitly flagged **[UNVERIFIED]**. The fast-model "WebFetch" summaries that returned only generic/abstract-level content were discarded and replaced with numbers read directly from the paper PDFs/HTML; where I relied on a summarizer's reading rather than the raw text I flag it.

---

## Executive Summary

1. **Distillation of *rationales* (not just labels) is the highest-leverage way to make a sub-1B model reason.** "Distilling Step-by-Step" shows a 770M T5, trained with teacher rationales as a *second supervised objective*, beats a 540B PaLM few-shot teacher while using **80%** of the labeled data, and matches full-data standard fine-tuning with as little as **12.5%** of e-SNLI ([arxiv.org/abs/2305.02301](https://arxiv.org/abs/2305.02301), [html](https://arxiv.org/html/2305.02301)).

2. **For small models, distillation beats running RL directly on the small model.** DeepSeek-R1 reports that `DeepSeek-R1-Distill-Qwen-32B` (SFT-distilled from R1) scores **72.6 / 94.3** (AIME 2024 / MATH-500 pass@1) vs only **47.0 / 91.6** for an RL-from-scratch `Qwen2.5-32B` ("…-Zero"), and states distillation into small models is "more economical and effective" than large-scale RL on them ([arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948), [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)).

3. **You do not need gold rationales.** STaR (self-bootstrapping) and rejection-sampling fine-tuning generate their own rationales from *labels only*, keeping the ones that reach the right answer; COLLATE (the user's referenced paper) goes further and *prefers* rationales by how much they raise the model's likelihood of the gold answer — a label-only, teacher-free DPO signal ([arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465), [arxiv.org/abs/2506.02519](https://arxiv.org/abs/2506.02519)).

4. **Reasoning can be *internalized*: learned in training but not emitted at inference.** TFRank distills DeepSeek-R1 CoT into a **1.7B** reranker with a "think-mode switch," then runs **think-free** at inference (a direct relevance score, no CoT tokens), matching rerankers ~4× larger ([arxiv.org/abs/2508.09539](https://arxiv.org/abs/2508.09539)). This is the exact pattern your **ReasoningClassifier** already approximates with a frozen-encoder explanation-embedding auxiliary head.

5. **Recommended recipe for a sub-1B causal-direction reasoner + reranker:** bootstrap rationales (teacher or self-generated, filtered by causal-direction correctness) → **SFT** on (input → rationale → direction) to install the behavior → **GRPO** with a verifiable direction/ranking reward to refine. This mirrors the DeepSeek "SFT-cold-start → RL" pipeline scaled down (details in the recipe section).

---

## 1. COLLATE — Preferential Rationale Tuning (arXiv 2506.02519)

**Patnaik, Aggarwal, Bhatia, Krishnamurthy — MDSR Lab, Adobe. Accepted ACL 2025 (Main).** Full title: *"Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning."* ([arxiv.org/abs/2506.02519](https://arxiv.org/abs/2506.02519))

### Problem framing
The motivating constraint is *commercial / legal*: outputs of closed LLMs (GPT-4, PaLM-540B) cannot be used to train models for commercial use, and their APIs are costly/unreliable. So COLLATE asks: **can a small LLM improve its own reasoning using only itself, with no external teacher?** (Read directly from the paper's introduction PDF.) This is distinct from classic distillation, where a *larger* teacher supplies rationales.

### Method (read from the paper's §3, equations as printed)
COLLATE has three stages. Backbone notation: `M` = base LLM, `M_IFT` = its instruction-fine-tuned version, `RP_s` = "rationale provider" instance *s*.

**3.1 Multi-mode Instruction Fine-Tuning.** `M` is fine-tuned in **three modes** via teacher-forced cross-entropy, selected by prompt format `P_m`:
- `L_{I→A}` : instruction → answer (Eq. 1)
- `L_{I→R}` : instruction → rationale (Eq. 2)
- `L_{[I;R]→A]}` : instruction + rationale → answer (Eq. 3)

The third mode is the key: it lets `M_IFT` act as a **scorer** of how much a rationale raises the likelihood of the gold answer. Training data is a 140k-sample subset of **CoT-Collection** (Kim et al., 2023). (All read from §3.1 PDF text.)

**3.2 Distinct Rationale Providers.** Clone `S` instances of `M_IFT`; split the rationale dataset into `S` random splits and DPO-tune each clone on its own split so the clones **exhibit distinct behavior** (the "collaborate" / "learning together" idea — diverse reasoners, same base weights). Each clone is first DPO-tuned to prefer the *gold* rationale `R^s_j` over its own generated rationale `R̂^s_j`, with the standard DPO loss (Eq. 5 as printed):

```
L_RP_s = − log σ[ β ( log π_RP_s(R^s_j)/π_M_IFT(R^s_j) − log π_RP_s(R̂^s_j)/π_M_IFT(R̂^s_j) ) ]
```
with `β = 0.1`, `M_IFT` as the frozen reference. This yields `S` diverse providers.

**3.3 Task-Based Preferential Rationale Tuning** (the core contribution). On a *new* end-task `D_T` that has **only (instruction, answer) pairs and no gold rationales**:
1. Each provider `RP_s^{(i-1)}` generates a candidate rationale for instruction `I_T`.
2. **Usefulness score = likelihood of the gold answer conditioned on the rationale**, computed by `M_IFT` in its `[I;R]→A` mode. Rationales are ranked by this score; the top is the **winning** rationale `R̂_w`, a lower one is the **eliminated** rationale `R̂_e`.
3. These form a **preference pair** for another DPO step that pushes the provider toward generating answer-helpful rationales.
4. **Likelihood-based sample filtration (Eq. 9):** keep a sample only if the winning rationale actually *raises* the gold-answer likelihood vs. using no rationale —
   `π(A_T | [P; I_T; R̂_w]) > π(A_T | I_T)`.
   This discards samples where reasoning does not help, so DPO sees only high-quality contrasts.
5. Iterated for `i ∈ {1,2}` passes over `D_T`.

So **"preferential rationale tuning" = DPO where the preference label comes not from a human or an LLM-judge but from the model's own likelihood of the verified gold answer.** It is teacher-free and needs only task labels. (Method read from §3.1–§3.3 PDF text incl. Eqs. 1–5, 9.)

### Models
Backbones span **1B–8B across families**: `open-llama-v2-7b`, `OLMo-1B`, `Phi3-3.8B`, `Qwen1.5-4B`, `LLaMA3-8B`. Main comparison table uses `open-llama-v2-7b` for all methods. (Read from §4 / experimental-setup PDF text.) *(Note: the fast-model summary's earlier claim of "Phi-2 / Mistral-7B / Llama-2-13B / GPT-4 teacher" was a hallucination and is discarded — those strings do not appear in the PDF.)*

### Datasets (5 datasets / 3 domains, read from PDF)
- **Math word problems:** GSM8K
- **NLI:** PIQA, WinoGrande
- **Commonsense reasoning:** CSQA (CommonsenseQA), HellaSwag

### Results (read from PDF result text)
- COLLATE **outperforms several trainable + prompting baselines by up to ~7%** without any external/larger LLM.
- vs **SPIN** (a DPO baseline that prefers the gold answer over the model's answer): COLLATE wins by **~7.5% on PIQA** and **~3% on WinoGrande**.
- On a generalization study (AIME / CROW / HotpotQA) COLLATE beats SPIN by **+3.7% (AIME 16.02→19.72), +4.0% (CROW 68.29→72.18), +1.7% (HotpotQA 62.39→64.11)**.

### Why it matters for us
COLLATE is the cleanest answer to *"I have causal-direction labels but no gold reasoning."* Its **likelihood-of-gold-answer scorer** is a drop-in way to (a) generate rationale candidates, (b) rank them without a judge model, and (c) build DPO pairs — all teacher-free. For a binary causal-direction task the "answer" is the direction label, making the usefulness score trivially computable.

---

## 2. CoT Distillation

### 2.1 Distilling Step-by-Step (arXiv 2305.02301, Hsieh et al., ACL 2023 Findings)
Source: [arxiv.org/abs/2305.02301](https://arxiv.org/abs/2305.02301), full text [arxiv.org/html/2305.02301](https://arxiv.org/html/2305.02301).

- **Idea:** extract teacher rationales via CoT prompting, then train the small model in a **multi-task** setup with *two* objectives and task prefixes: `[label]` → predict answer, and `[rationale]` → produce the explanation. Crucially the rationale is a **training target, not an input** — at inference you can just ask for `[label]` (cheap). This is the conceptual ancestor of "internalized reasoning."
- **Teacher/Student:** teacher PaLM **540B**; students T5 **220M / 770M / 11B**.
- **Datasets:** e-SNLI, ANLI, CQA, SVAMP.
- **Headline results** (read from HTML, corroborated by abstract):
  - A **770M** T5 with Distilling Step-by-Step beats the **540B** PaLM few-shot teacher using **80%** of the labeled data; standard fine-tuning of the same 770M "struggles to match even with 100%."
  - On **e-SNLI**, it matches full-data (100%) standard fine-tuning using only **12.5%** of the data.
  - Beats standard fine-tuning with **>50% fewer training examples on average** across the four datasets.
  - *(Exact per-dataset accuracy tables not transcribed here; cite the HTML tables if needed.)*

**Takeaway:** rationale-as-auxiliary-objective is *data-efficient* and *inference-cheap* — directly relevant to a sub-1B model and to your auxiliary-head ReasoningClassifier.

### 2.2 DeepSeek-R1-Distill (arXiv 2501.12948) — distillation beats RL on small models
Source: [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) (Nature 2025), model card [huggingface.co/deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

- **Method:** curate **~800k** reasoning + general samples generated by DeepSeek-R1, then **pure SFT** (no RL) on open bases: Qwen2.5 **1.5B / 7B / 14B / 32B**, Llama **3.1-8B / 3.3-70B**.
- **Key evidence (distillation > RL on small models):** the paper/model card states *"the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models,"* and that large-scale RL on a small model "requires enormous computational power and may not even achieve the performance of distillation."
- **The controlled comparison** (read via search of the official numbers; verify in Table 6 of the paper):
  - `Qwen2.5-32B` trained with RL from scratch ("…-Zero"): **AIME 2024 = 47.0**, **MATH-500 = 91.6**.
  - `DeepSeek-R1-Distill-Qwen-32B` (SFT distillation): **AIME 2024 = 72.6**, **MATH-500 = 94.3**.
- **Small-model distilled scores** (official model card):
  - `…-Distill-Qwen-1.5B`: **AIME 2024 = 28.9**, **MATH-500 = 83.9**.
  - `…-Distill-Qwen-7B`: **AIME 2024 = 55.5**, **MATH-500 = 92.8**.
  - (For context, the 1.5B's 28.9 AIME exceeds GPT-4o-0513's reported 9.3 on the same card — distillation gives a tiny model frontier-relevant math reasoning.)

**Takeaway for sub-1B:** spend the compute budget on (a) a good teacher and (b) clean SFT data, *before* spending it on RL. RL is the *refinement* step, not the bootstrapping step, for small models.

### 2.3 Symbolic CoT Distillation — SCoTD (arXiv 2306.14050) & "Teaching Small LMs to Reason"
Source: [arxiv.org/abs/2306.14050](https://arxiv.org/abs/2306.14050), [html](https://arxiv.org/html/2306.14050v2).

- **Finding that motivates the whole field:** CoT benefits normally emerge only for models **>~50B**; SCoTD shows **125M–1.3B** students *can* reason if trained on teacher rationalizations.
- **Method:** sample **many** (e.g. ~30) diverse CoT chains per instance from a large teacher (GPT-3), build a distillation corpus, SFT the small student on it. Sampling *many* chains (not one) is reported as important.
- **Result (read via summarizer of the abstract/HTML — confirm exact table):** on the IMDB contrast set, SCoTD reaches **92.0%** vs **81.6%** for a label-only student; human eval rates student CoTs as coherent/fluent, comparable to the teacher's. **[Partly UNVERIFIED — the 92.0/81.6 pair came from a search summary of the HTML, not transcribed from the raw table; confirm before quoting.]**
- *Related:* Magister et al., **"Teaching Small Language Models to Reason"** (arXiv 2212.08410) is the parallel result that CoT fine-tuning transfers reasoning from a large teacher into students as small as T5/GPT-2-scale. **[Method/numbers UNVERIFIED here — not fetched in this session; cite [arxiv.org/abs/2212.08410](https://arxiv.org/abs/2212.08410) and verify before quoting.]**

---

## 3. Rationale / Explanation Supervision into Classifiers and Encoders

### 3.1 e-SNLI (Camburu et al., NeurIPS 2018)
Source: [arxiv.org/abs/1812.01193](https://arxiv.org/abs/1812.01193), [NeurIPS page](https://papers.nips.cc/paper_files/paper/2018/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html).

- **Dataset:** extends SNLI with **human natural-language explanations** for the label (commonly cited as **~569k–570k** explanations, one per train example plus 3 per dev/test example). **[Exact count not transcribed from the primary PDF this session — the well-known figure is ~569k; flag as needs-verify if exact precision matters.]**
- **Model setups the paper studies:**
  - **Predict-then-explain** (label first, justify after) and **explain-then-predict** (generate explanation, then label from it).
  - Using explanations as **auxiliary supervision** to improve **universal sentence representations** and **out-of-domain NLI transfer**.
- **Key qualitative finding:** explanation supervision is valuable for representation quality and transfer; in-domain label accuracy gains from explanations are modest/mixed (the bigger value is interpretability + transfer). **[Quantitative deltas UNVERIFIED from primary source this session — the abstract confirms the goals (sentence representations, OOD transfer) but the summarizer did not return the numeric table; verify in the paper before quoting accuracy deltas.]**

**Relevance:** e-SNLI is the canonical precedent for *training a model to both classify and explain*, and for using explanations to shape the encoder representation — exactly what your ReasoningClassifier does with its cosine-aligned explanation head.

### 3.2 TFRank — think-free internalized reasoning for reranking (arXiv 2508.09539)
Source: [arxiv.org/abs/2508.09539](https://arxiv.org/abs/2508.09539), html [arxiv.org/html/2508.09539v1](https://arxiv.org/html/2508.09539v1).

- **Goal:** a reranker that *learns* to reason but, at inference, **emits a relevance score directly with no CoT tokens** ("think-free") — cutting latency while keeping reasoning-level quality.
- **Mechanism — "think-mode switch":** during training the model is exposed to explicit CoT (so the reasoning is internalized into weights); a switch lets it run in a no-reasoning mode at inference and output a score. Combined with **fine-grained / soft score supervision** (continuous relevance, not just binary) and **multi-task** training (pointwise/pairwise/listwise).
- **Teacher / data:** CoT training data sourced from **DeepSeek-R1** over **MS MARCO / M3** data (per summarizer of HTML).
- **Model size:** the flagship variant is **1.7B** (sub-2B; there is also an 8B variant). The paper claims it matches rerankers with **~4× more parameters**, and reports `TFRank-8B` (SFT+GRPO) significantly beating `REARANK-7B` on BRIGHT (paired t-test p ≈ 5.7e-17). **[Exact nDCG@10 tables not transcribed — the HTML body exceeded fetch size; the comparative claims came from the HTML summarizer. Verify the BEIR/BRIGHT nDCG numbers in the paper tables before quoting.]**

**Relevance:** TFRank is the strongest published validation of the pattern *"distill reasoning into a small reranker, then drop reasoning at inference"* — which is precisely the production constraint for a causal-direction reranker.

### 3.3 General pattern: rationale objective on a cross-encoder/classifier
The unifying recipe across e-SNLI, Distilling Step-by-Step, and TFRank: **add a rationale-derived auxiliary loss to a discriminative head.** Variants:
- **Generative auxiliary** (Distilling Step-by-Step): a `[rationale]` decoding head trained jointly with the `[label]` head.
- **Embedding-alignment auxiliary** (your ReasoningClassifier; TFRank's score supervision): align an internal representation to a (frozen) rationale embedding via cosine/contrastive loss.
- Both *regularize the representation toward reasoning-relevant features* while keeping inference cheap (only the label/score head runs).

---

## 4. Self-Distillation / Self-Taught Reasoning (bootstrapping from labels only)

### 4.1 STaR — Self-Taught Reasoner (arXiv 2203.14465, Zelikman et al., NeurIPS 2022)
Source: [arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465).

- **Loop:**
  1. Few-shot prompt the model to produce a rationale + answer for each problem.
  2. **Keep** rationales whose answer is correct.
  3. For problems it got **wrong**, do **rationalization**: feed the *gold answer as a hint*, let the model produce a (now-correct) rationale, and add those.
  4. Fine-tune on all kept rationales; repeat.
- **Why rationalization matters:** it recovers training signal on hard problems the model can't yet solve forward, preventing the bootstrap from stalling on easy cases.
- **Headline result (abstract):** STaR performs comparably to fine-tuning a **30× larger** model on CommonsenseQA, and outperforms a model fine-tuned to predict answers directly; the base model used is GPT-J-scale. **[Exact per-task numbers UNVERIFIED from primary text this session — abstract confirms the "30× larger" comparison and the direct-prediction baseline; verify exact arithmetic/CQA/GSM8K accuracies in the paper tables.]**

### 4.2 Rejection-sampling Fine-Tuning (RFT)
- **What it is:** sample many CoT solutions per training question from the current model (or a teacher), **filter to those that reach the verified answer**, dedupe/diversify reasoning paths, and SFT on the survivors. It is STaR without the iterative re-prompting loop — a one-shot "best-of-N then SFT."
- **Relation:** RFT = the *data-generation* half of STaR/DeepSeek's pipeline; DeepSeek-R1's 800k SFT corpus is essentially large-scale rejection-sampled teacher data. *(Conceptual synthesis; the specific "RFT" framing is from Yuan et al. 2023 "Scaling Relationship on Learning Mathematical Reasoning with LLMs," [arxiv.org/abs/2308.01825](https://arxiv.org/abs/2308.01825) — **[not fetched this session; verify before quoting numbers]**.)*

### 4.3 Bootstrapping rationales when you only have labels
This is your situation (causal-direction labels, no gold reasoning). Three teacher-free / cheap-teacher options, in increasing sophistication:
1. **Rationalization (STaR):** condition a capable model on the *gold direction* to write a post-hoc justification; keep it as the rationale target.
2. **Rejection sampling (RFT):** sample N forward rationales, keep ones predicting the correct direction.
3. **Likelihood-preference (COLLATE):** rank candidate rationales by how much they raise the small model's own probability of the gold direction; build DPO pairs — no judge, no large teacher.

---

## 5. Distillation-then-RL Pipelines (recommended shape for sub-1B)

The DeepSeek-R1 result decomposes into a two-phase template that is now standard and is the right shape for sub-1B:

1. **SFT cold-start on distilled rationales** to *install* the behavior (a small model cannot find good reasoning via RL exploration alone — exploration is too sparse; SFT on teacher/self-distilled chains gives it a working prior). DeepSeek explicitly used a small cold-start SFT set before RL ([arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)).
2. **RL (GRPO) to refine** with a *verifiable reward*. GRPO (Group Relative Policy Optimization, from DeepSeekMath, [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) — **[not fetched this session; standard reference, verify before quoting equations]**) is preferred for small models because it drops the value/critic network (estimating advantage from a *group* of sampled outputs' rewards), halving memory vs PPO — important on constrained GPUs.

**Why this order for sub-1B:** §2.2's controlled numbers (Distill-32B 72.6 vs RL-Zero-32B 47.0 on AIME) show RL-from-scratch on a small model underperforms distillation *and* costs far more compute. SFT-then-RL gets the SFT prior for free and uses RL only to sharpen.

---

## Comparison Table

| Method (paper) | Teacher → Student | Data needs | What is distilled / supervised | Inference cost | Key result (source) |
|---|---|---|---|---|---|
| **COLLATE** (2506.02519) | **None** (self; clones of `M_IFT`, 1B–8B) | Labels only (no gold rationales); 140k CoT-Collection for IFT | DPO preference over rationales, ranked by likelihood of **gold answer** | Optional CoT | +~7.5% PIQA, +3% WinoGrande vs SPIN; up to +7% vs baselines ([abs](https://arxiv.org/abs/2506.02519)) |
| **Distilling Step-by-Step** (2305.02301) | PaLM-540B → T5 220M/770M/11B | Far less labeled data | Teacher rationales as a **2nd `[rationale]` objective** alongside `[label]` | Cheap (`[label]` only) | 770M beats 540B w/ 80% data; e-SNLI: 12.5% data ≈ 100% standard FT ([html](https://arxiv.org/html/2305.02301)) |
| **DeepSeek-R1-Distill** (2501.12948) | R1 → Qwen 1.5/7/14/32B, Llama 8/70B | ~800k curated R1 samples | Full reasoning traces via **pure SFT** | Generates CoT | Distill-32B 72.6/94.3 vs RL-Zero-32B 47.0/91.6 (AIME/MATH-500) ([abs](https://arxiv.org/abs/2501.12948)) |
| **SCoTD** (2306.14050) | GPT-3 → 125M–1.3B | Labels + teacher CoT (≈30 chains/inst) | Many diverse symbolic CoT chains, SFT | Generates CoT | small models gain CoT ability; IMDB-contrast 92.0 vs 81.6 **[verify]** ([abs](https://arxiv.org/abs/2306.14050)) |
| **STaR** (2203.14465) | **Self** (GPT-J-scale) | Few CoT seeds + labels | Self-generated rationales (incl. **rationalization** from gold answer) | Generates CoT | ≈ a 30× larger model on CommonsenseQA **[verify]** ([abs](https://arxiv.org/abs/2203.14465)) |
| **e-SNLI** (1812.01193) | Human → classifier/encoder | ~569k human explanations **[verify]** | Explanation as supervision/target | Cheap (label) | Improves sentence reps & OOD transfer **[verify deltas]** ([abs](https://arxiv.org/abs/1812.01193)) |
| **TFRank** (2508.09539) | DeepSeek-R1 → 1.7B reranker | MS MARCO/M3 + R1 CoT | CoT internalized; **think-free** score at inference | **No CoT tokens** | matches ~4× larger rerankers; 8B+GRPO > REARANK-7B on BRIGHT **[verify nDCG]** ([abs](https://arxiv.org/abs/2508.09539)) |

---

## 6. The team's ReasoningClassifier — relation to rationale distillation & how to improve it

**Current architecture** (`automation/internship-causal-embedding/src/classifier/reasoning_model.py`):
- Two **trainable** encoders (src, tgt), dual-encoder fusion `[A; B; A−B; A∗B]` → shared MLP backbone.
- One **frozen** `reason_encoder` that embeds the **ground-truth explanation**.
- Two heads off the shared backbone: a **prediction head** (`Linear(H,1)` + `BCEWithLogitsLoss`) for the direction, and a **reasoning head** (`Linear(H,d)` + `CosineEmbeddingLoss`) that aligns the hidden state to the frozen explanation embedding.
- `total = dir_loss + w * reason_loss`; at val/test `enc_reason=None`, so **only the direction logit runs** — i.e. it is already a *think-free at inference* design.

**How this maps to the literature.** This is a **rationale-as-auxiliary-embedding** distillation: it is the embedding-alignment cousin of Distilling Step-by-Step's `[rationale]` objective (§2.1) and structurally the same idea as TFRank's "internalize reasoning in training, drop it at inference" (§3.2). The CosineEmbeddingLoss is a soft regularizer pulling the classifier representation toward the explanation manifold — analogous to e-SNLI's "explanations improve sentence representations" (§3.1).

**Concrete improvement ideas, ranked.**

1. **Upgrade the rationale *source* (highest leverage).** Right now you align to a *given* gold explanation. If those explanations are weak or sparse, generate stronger ones with the §4.3 ladder: (a) STaR-style rationalization conditioning a capable model on the gold direction, or (b) COLLATE-style likelihood ranking to pick the explanation that most raises the gold-direction probability. Better targets → better aux signal, no architecture change.

2. **Add a generative rationale head (Distilling Step-by-Step style), not only cosine alignment.** A cosine target only constrains *one direction* in embedding space. A lightweight `[rationale]` generation objective (even a small decoder or a prompted seq2seq during training) forces the representation to be *reconstruction-sufficient* for the reasoning, which is a stronger inductive bias. Keep it train-only so inference stays think-free.

3. **Make the alignment contrastive, not just positive-cosine.** CosineEmbeddingLoss with only positive pairs can collapse. Add **negatives**: explanations of the *wrong* direction / other samples, and use InfoNCE-style contrast (pull correct-direction explanation, push wrong-direction explanation). This injects the *causal-direction* discrimination directly into the reasoning head — well suited to your task.

4. **Curriculum / DPO refinement (the §5 second phase).** After SFT-style joint training converges, add a **GRPO** pass with a verifiable reward = correct causal direction (and, for reranking, a ranking metric like nDCG/MRR). This is the "distill-then-RL" shape that §2.2/§5 show is the right finish for small models. Reward can be computed entirely from labels (verifiable), no judge needed.

5. **Try a single shared encoder + late interaction for the reranker variant.** The two-trainable-encoder design is fine for classification; for reranking throughput, consider a cross-encoder with the same train-only reasoning head, à la TFRank, and serve the score directly.

6. **Watch the frozen-encoder cost.** Three encoder copies (src, tgt, frozen reason) at train time is heavy. Per your hardware memory note (CPU OOM crashes everything), the frozen `reason_encoder` can be run **offline** to *cache* explanation embeddings once, removing it from the training graph entirely and freeing a full encoder's worth of GPU/CPU memory.

---

## References (with URLs)

- **COLLATE** — Patnaik, Aggarwal, Bhatia, Krishnamurthy. *Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning.* ACL 2025. [https://arxiv.org/abs/2506.02519](https://arxiv.org/abs/2506.02519)
- **Distilling Step-by-Step** — Hsieh et al. ACL 2023 Findings. [https://arxiv.org/abs/2305.02301](https://arxiv.org/abs/2305.02301) · full text [https://arxiv.org/html/2305.02301](https://arxiv.org/html/2305.02301)
- **DeepSeek-R1** — DeepSeek-AI. *Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature 2025. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) · model card [https://huggingface.co/deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- **SCoTD** — Li et al. *Symbolic Chain-of-Thought Distillation.* ACL 2023. [https://arxiv.org/abs/2306.14050](https://arxiv.org/abs/2306.14050) · [https://arxiv.org/html/2306.14050v2](https://arxiv.org/html/2306.14050v2)
- **Teaching Small LMs to Reason** — Magister et al. 2022. [https://arxiv.org/abs/2212.08410](https://arxiv.org/abs/2212.08410) *[not fetched this session]*
- **STaR** — Zelikman et al. *Self-Taught Reasoner.* NeurIPS 2022. [https://arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465)
- **RFT** — Yuan et al. *Scaling Relationship on Learning Mathematical Reasoning with LLMs.* 2023. [https://arxiv.org/abs/2308.01825](https://arxiv.org/abs/2308.01825) *[not fetched this session]*
- **GRPO / DeepSeekMath** — Shao et al. 2024. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) *[not fetched this session]*
- **e-SNLI** — Camburu et al. NeurIPS 2018. [https://arxiv.org/abs/1812.01193](https://arxiv.org/abs/1812.01193) · [https://papers.nips.cc/paper_files/paper/2018/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html](https://papers.nips.cc/paper_files/paper/2018/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html)
- **TFRank** — *Think-Free reranking.* 2025. [https://arxiv.org/abs/2508.09539](https://arxiv.org/abs/2508.09539) · [https://arxiv.org/html/2508.09539v1](https://arxiv.org/html/2508.09539v1)

### Verification status
- **Read from primary PDF/HTML this session:** COLLATE (method Eqs. 1–5,9; models; 5 datasets; SPIN deltas and AIME/CROW/HotpotQA numbers); Distilling Step-by-Step (80% / 12.5% / >50% claims); DeepSeek-R1-Distill scores (1.5B/7B from card; 32B distill-vs-Zero from official numbers).
- **Flagged [UNVERIFIED] / verify-before-quoting:** SCoTD 92.0/81.6 IMDB pair; STaR exact per-task accuracies; e-SNLI exact dataset count and accuracy deltas; TFRank exact nDCG tables; Magister/RFT/GRPO numbers (not fetched).
- **Discarded as hallucination:** an early fast-model summary claiming COLLATE used Phi-2/Mistral-7B/Llama-2-13B with a GPT-4 teacher — contradicted by the PDF (actual: open-llama-v2-7b, OLMo-1B, Phi3-3.8B, Qwen1.5-4B, LLaMA3-8B; teacher-free).
