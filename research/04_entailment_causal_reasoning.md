# Entailment, NLI, and Causal Reasoning for Directional Causal Precedence

**Audience:** A team building a model that predicts directional causal precedence between documentation steps ("does step A causally precede / enable step B?") and wants to inject explicit reasoning into the prediction.

**Date:** 2026-06-29

**Citation policy:** Every non-obvious quantitative claim carries an inline source URL. Numbers that could not be confirmed against a primary source in this research pass are explicitly flagged as **[UNVERIFIED]**. No numbers are fabricated; where a source did not state a figure, that absence is noted rather than filled in.

---

## Executive Summary

1. **NLI / textual entailment** is the well-studied 3-way classification of a (premise, hypothesis) pair into entailment / contradiction / neutral (https://nlp.stanford.edu/projects/snli/). It is *inherently pairwise and directional* — "premise entails hypothesis" is not symmetric — which makes it a natural template for "step A causally precedes step B."

2. **DeBERTa-v3 cross-encoders are the strong, cheap NLI workhorse.** A DeBERTa-v3-large NLI checkpoint reaches **0.912 / 0.908** matched/mismatched accuracy on MNLI and **0.702** average on adversarial ANLI (https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli). The cross-encoder lets premise and hypothesis tokens attend to each other, which is exactly the interaction a directional-precedence judgment needs.

3. **Explicit reasoning measurably helps entailment**, especially the *implied* (multi-hop / world-knowledge) cases that surface NLI models get near-chance on. The INLI work reports models trained on standard NLI corpora score only **~50%** on implied entailment, while a model trained on implication-aware data reaches **88.5%** on those cases (https://arxiv.org/html/2501.07719v1). Reasoning structures (entailment trees, atomic-fact decomposition, neuro-symbolic CoT validation) make the *why* inspectable and improve the *what*.

4. **Causal direction is the consistently hard part of causal NLP.** Even GPT-4 is near-random at inferring causation from correlation (F1 **29.08**, https://arxiv.org/html/2306.05836v3) and "fails at all direct causation recognition with parents and children" — i.e., it cannot reliably tell A→B from B→A. Supervised span models on annotated corpora reach only **0.79–0.86 F1** on direction prediction (https://ar5iv.labs.arxiv.org/html/2103.13606). This argues for *explicit asymmetric supervision*, not generic semantic similarity.

5. **A verifiable label is a clean training signal.** Because the directional/entailment label is discrete and ground-truthed, it slots directly into Reinforcement Learning from Verifiable Rewards (RLVR; https://arxiv.org/abs/2411.15124) and DeepSeek-R1-style accuracy rewards (https://arxiv.org/html/2501.12948v1) as a 0/1 reward, while the reasoning trace stays free-form. For a sub-1B model, **distillation of teacher chains-of-thought filtered by the verifiable label** is the empirically favored route over running RL directly on a small base.

---

## 1. NLI / RTE Fundamentals and DeBERTa-v3 NLI Models

### 1.1 The task

Natural Language Inference (NLI), historically Recognizing Textual Entailment (RTE), determines the inference relation between two ordered texts — a **premise** and a **hypothesis** — as one of **entailment** (hypothesis follows), **contradiction** (hypothesis incompatible), or **neutral** (undetermined) (https://nlp.stanford.edu/projects/snli/). The directionality is intrinsic: "P entails H" does not imply "H entails P."

**Foundational corpora:**

- **SNLI** (Bowman et al., 2015, *A large annotated corpus for learning natural language inference*, EMNLP 2015, https://arxiv.org/abs/1508.05326): ~**570k** human-written English sentence pairs, balanced across the three labels, "approximately two orders of magnitude larger than all other resources of its type" (https://nlp.stanford.edu/projects/snli/).
- **MultiNLI / MNLI** (Williams et al., 2018, *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference*, NAACL-HLT 2018, https://aclanthology.org/N18-1101/, arXiv https://arxiv.org/abs/1704.05426): ~**433k** examples across **ten genres** of written and spoken English, "a substantially more difficult task" than SNLI. MNLI defines the standard **matched** (in-domain) vs **mismatched** (cross-domain) evaluation split — the source of the `mnli-m` / `mnli-mm` metrics below.

> Note: the explicit "entailment / contradiction / neutral" 3-label wording is taken from the Stanford SNLI project page (same authors) rather than verbatim from the paper abstracts; it is standard and consistent across the literature.

### 1.2 ANLI — Adversarial NLI (why benchmarks got harder)

ANLI (Nie et al., 2020, *Adversarial NLI: A New Benchmark for Natural Language Understanding*, ACL 2020, https://aclanthology.org/2020.acl-main.441/, arXiv https://arxiv.org/abs/1910.14599) was collected by an "iterative, adversarial human-and-model-in-the-loop procedure": annotators write hypotheses that **fool the current best model**, and only verified model-fooling examples are kept. Because examples sit precisely where strong models fail, accuracy is far lower than on SNLI/MNLI. Round structure and train sizes (https://huggingface.co/datasets/facebook/anli):

- **R1** (target model BERT-Large): 16,946 train / 1,000 dev / 1,000 test
- **R2** (target RoBERTa): 45,460 train / 1,000 dev / 1,000 test
- **R3** (target RoBERTa, wider context sources): 100,459 train / 1,200 dev / 1,200 test

ANLI is the reason a "good" NLI score must be reported on adversarial data, not just MNLI.

### 1.3 Why DeBERTa-v3 is a strong NLI cross-encoder

DeBERTa-v3 (He, Gao, Chen, 2021, *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*, arXiv https://arxiv.org/abs/2111.09543; ICLR 2023) combines three ideas:

1. **Disentangled attention** (from DeBERTa): content and position are represented separately, with self-attention computed across content-to-content, content-to-position, and position-to-content terms.
2. **ELECTRA-style Replaced Token Detection (RTD)**: a more sample-efficient pretraining objective than masked language modeling — a discriminator predicts which tokens a generator replaced.
3. **Gradient-Disentangled Embedding Sharing (GDES)**: resolves the "tug-of-war" in vanilla ELECTRA where the generator and discriminator losses pull shared token embeddings in different directions, improving both efficiency and quality.

Reported headline: DeBERTaV3-large reaches a **91.37% average GLUE** score (https://arxiv.org/abs/2111.09543).

For NLI specifically, the model is fine-tuned on the concatenated pair (`premise [SEP] hypothesis`) so that **every premise token can attend to every hypothesis token** — full cross-attention. This token-level interaction is what pairwise inference needs, and it is why DeBERTa-v3 NLI cross-encoders top the public leaderboards.

### 1.4 DeBERTa-v3 NLI checkpoints — reported accuracy

**MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli** (https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) — fine-tuned on **885,242** NLI pairs from MultiNLI, Fever-NLI, ANLI, LingNLI, and WANLI. Reported accuracy from the model card's metrics table:

| Eval set | Accuracy |
|---|---|
| MNLI matched (`mnli_test_m`) | **0.912** |
| MNLI mismatched (`mnli_test_mm`) | **0.908** |
| ANLI test (avg over rounds) | **0.702** |
| ANLI test R3 | **0.64** |
| LingNLI | **0.87** |
| WANLI | **0.77** |

> **[UNVERIFIED]** No separate FEVER-NLI *test* accuracy is published on the card (only dev was referenced).

**cross-encoder/nli-deberta-v3-large** (https://huggingface.co/cross-encoder/nli-deberta-v3-large) — base `microsoft/deberta-v3-large`, trained on SNLI + MultiNLI. Output label order is `['contradiction', 'entailment', 'neutral']` (note: differs from many other NLI models — important for index mapping). Reported: **SNLI-test 92.20**, **MNLI mismatched 90.49**.

> **[UNVERIFIED]** MNLI *matched* accuracy is not reported separately on this card — do not cite an mnli-m figure for this checkpoint.

### 1.5 Cross-encoder vs bi-encoder

Per the Sentence-Transformers docs (https://www.sbert.net/examples/applications/cross-encoder/README.html): a **bi-encoder** embeds each sentence independently (good for retrieval/search at scale); a **cross-encoder** passes both sentences jointly through the Transformer to produce a single pair score and "does not produce a sentence embedding." Cross-encoders "achieve better performances than Bi-Encoders" but do not scale to large-scale comparison (the docs cite ~65 hours to cross-encode 10k sentence pairs vs ~5 seconds for a bi-encoder). **NLI is inherently pairwise** and benefits from the cross-attention; that is why the strong NLI checkpoints are cross-encoders. *Implication for the team:* a directional-precedence judgment over a candidate step pair is a cross-encoder task; a bi-encoder is appropriate only for first-stage candidate retrieval.

---

## 2. Reasoning Over Entailment: CoT, Explanations, and Implicit Entailment

### 2.1 Natural-language explanations: e-SNLI

e-SNLI (Camburu, Rocktäschel, Lukasiewicz, Blunsom, 2018, *e-SNLI: Natural Language Inference with Natural Language Explanations*, NeurIPS 2018, https://arxiv.org/abs/1812.01193) "extend[s] the Stanford Natural Language Inference dataset with an additional layer of human-annotated natural language explanations of the entailment relations." Each example gets a free-form human explanation plus highlighted rationale words. Splits (https://huggingface.co/datasets/esnli/esnli): **549,367** train / **9,842** validation / **9,824** test. It is used both to *train* models that "output [explanations] at test time" and to *evaluate* generated justifications against the human references.

> **[UNVERIFIED]** Secondary summaries cite "543,950 of 570,152 SNLI examples explained" and "6,325 workers"; these were not confirmable from the abstract or dataset card.

### 2.2 Chain-of-thought as a general method

CoT prompting (Wei et al., 2022, *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022, https://arxiv.org/abs/2201.11903): "generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning." Headline: a 540B model with eight CoT exemplars set SOTA on GSM8K.

> **Scope caveat:** Wei et al.'s reported benchmarks are arithmetic, commonsense, and symbolic — **not NLI**. Treat it as the general mechanism, not NLI-specific evidence. The NLI-/entailment-structured evidence is in 2.3–2.6.

### 2.3 Entailment-as-reasoning and entailment trees

- **Entailment as Few-Shot Learner (EFL)** (Wang et al., 2021, https://arxiv.org/abs/2104.14690): reformulate arbitrary NLP tasks *as* textual entailment and fine-tune from as few as 8 examples. *(Reported figures — "improves few-shot SOTA by 12%", competitive with models 500× larger — come from search snippets and are **[UNVERIFIED]** against the PDF.)*
- **EntailmentBank** (Dalvi et al., 2021, *Explaining Answers with Entailment Trees*, EMNLP 2021, https://arxiv.org/abs/2104.08661): **1,840** expert-annotated multistep entailment trees, where a hypothesis is proven via a tree of entailment steps over supporting sentences, at three difficulty levels (all relevant facts / + distractors / full corpus). This is the template for "prove a conclusion via a chain of intermediate entailments" — directly analogous to proving "A enables B" through intermediate steps.
- **Selection-Inference** (Creswell, Shanahan, Higgins, 2022, https://arxiv.org/abs/2205.09712) and **Faithful Reasoning** (https://arxiv.org/pdf/2208.14271): alternate *selection* (pick relevant facts) and *inference* (derive next step) to produce an interpretable, auditable reasoning trace.

### 2.4 VeriCoT — neuro-symbolic CoT validation (arXiv 2511.04662, VERIFIED)

VeriCoT (Feng, Weir, Bostrom, Bayless, Cassel, Chaudhary, Kiesl-Reiter, Rangwala, 2025, *VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks*, https://arxiv.org/abs/2511.04662, HTML https://arxiv.org/html/2511.04662v1) addresses that "even when [LLMs] reach correct answers, the underlying reasoning may be flawed." Method (verbatim from abstract):

- "formalizes each CoT reasoning step into first-order logic"
- "identifies premises that ground the argument in source context, commonsense knowledge, or prior reasoning steps"
- "the symbolic representation enables automated solvers to verify logical validity while the NL premises allow humans and systems to identify ungrounded or fallacious reasoning steps"

Evaluated on **ProofWriter, LegalBench, and BioASQ**, it "effectively identifies flawed reasoning, and serves as a strong predictor of final answer correctness."

> **[UNVERIFIED]** No specific accuracy/F1 numbers appear on the abstract page; the full-text HTML tables would carry them.

*Relevance:* VeriCoT is a template for *validating* a model's "A precedes B because…" trace — every step grounded in source documentation, commonsense, or a prior step, with logical consistency checked. The "grounded vs ungrounded premise" distinction maps cleanly to "is this precedence claim supported by the doc text or hallucinated?"

### 2.5 Atomic-SNLI — atomic fact decomposition (arXiv 2601.06528, VERIFIED)

Atomic-SNLI (Huang, 2026, *Atomic-SNLI: Fine-Grained Natural Language Inference through Atomic Fact Decomposition*, https://arxiv.org/abs/2601.06528, HTML https://arxiv.org/html/2601.06528) argues current NLI "provides black-box decisions that lack explanatory power." It decomposes hypotheses into atomic facts but shows the naive assumption — "a hypothesis is entailed only when all its atomic facts are entailed" — "fails in practice due to models' poor performance on fine-grained reasoning," with models "substantially worse on atomic level inference compared to sentence level." Atomic-SNLI is a dataset built by decomposing SNLI and enriching with curated atomic examples "through linguistically informed generation strategies"; models fine-tuned on it "achieve significant improvements in atomic reasoning capabilities while maintaining strong sentence level performance," enabling "transparent, explainable results at the fact level."

> **[UNVERIFIED]** The abstract states qualitative gains only — no accuracy numbers or dataset size were on the abstract page.

*Relevance:* documentation steps are multi-clause; decomposing "step A" and "step B" into atomic prerequisites/effects lets the model reason about which atomic effect of A satisfies which atomic precondition of B.

### 2.6 "Implied" vs "explicit" entailment — the most directly relevant distinction

INLI / *Entailed Between the Lines* (Havaldar, Alvari, Palowitch, Hosseini, Buthpitiya, Fabrikant, 2025, https://arxiv.org/abs/2501.07719, HTML https://arxiv.org/html/2501.07719v1) draws the distinction the team should care about:

- **Explicit entailment**: "follows directly from the text's lexical semantics (e.g. via synonymy and paraphrasing) and syntax."
- **Implied entailment**: "requires some sort of an additional cognitive step, such as logical reasoning, world knowledge, conversational pragmatics, or figurative language."

The **Implied NLI (INLI)** dataset has **10,000 premises with 40,000 hypotheses** (four per premise: explicit-entailment, implied-entailment, neutral, contradiction). Reported results (from the paper page):

- T5-XXL trained on INLI: **92.4%** overall accuracy; **88.5%** on implied-entailment cases.
- Models trained on SNLI / MNLI / ANLI / WANLI: only **~50%** (≈ chance) on implied entailments.
- Human agreement: Fleiss's Kappa **0.711**; majority agreement **93.5%**.

**This is the crux for causal precedence:** "step A enables step B" is almost always an *implied* relation requiring world knowledge ("you must authenticate before you can publish"), not surface lexical overlap. Standard NLI models are near-chance on exactly this regime — which is the argument for injecting reasoning rather than relying on embedding similarity.

---

## 3. Causal Direction and Causal Reasoning in NLP

### 3.1 Commonsense causal plausibility: COPA and e-CARE

- **COPA** (Roemmele, Bejan, Gordon, 2011, *Choice of Plausible Alternatives*, http://commonsensereasoning.org/2011/papers/Roemmele.pdf): **1,000** forced-choice questions (split 500 dev / 500 test). Each item gives a premise and two alternatives for *either the cause or the effect*; the model picks the more plausible. The **forward (effect) vs backward (cause)** framing — "What was the cause?" vs "What happened as a result?" — is the defining directional feature. COPA is part of SuperGLUE (Wang et al., 2019, https://arxiv.org/abs/1905.00537); the shared-task version is SemEval-2012 Task 7 (https://aclanthology.org/S12-1052/).
  > **[UNVERIFIED]** Often-cited "~65% SemEval-era / ~85% SuperGLUE-era" COPA accuracies were not confirmable from primary text.
- **e-CARE** (Du, Ding, Xiong, Liu, Qin, 2022, *e-CARE: a New Dataset for Exploring Explainable Causal Reasoning*, ACL 2022, https://aclanthology.org/2022.acl-long.33/, arXiv https://arxiv.org/abs/2205.05849): "over 21K causal reasoning questions," each paired with a **natural-language conceptual explanation** of the causal fact (the layer "absent in existing causal reasoning resources"). Best model **ALBERT 73.86%** vs **human 92.00%** on the causal-reasoning task. Crucially, **explanations transfer**: pretraining on e-CARE explanations lifts **COPA 70.4 → 75.4**, BECauSE 76.8 → 81.0, EventStoryLine F1 66.5 → 68.1. (Split counts 14,928 / 2,132 / 4,264 are from the GitHub README, https://github.com/Waste-Wood/e-CARE.)

  *Takeaway:* training a model to *explain* causality, not just label it, improves downstream causal tasks — direct support for the team's "inject reasoning" hypothesis.

### 3.2 Formal causal inference: Corr2Cause and CLADDER

- **Corr2Cause** (Jin et al., 2023, *Can Large Language Models Infer Causation from Correlation?*, ICLR 2024, https://arxiv.org/abs/2306.05836, HTML https://arxiv.org/html/2306.05836v3): "first benchmark dataset to test the pure causal inference skills of LLMs," **>200K** samples, 17 LLMs. Headline: LLMs "achieve almost close to random performance." Table 4: **GPT-4 F1 = 29.08** (P 20.92 / R 47.66 / Acc 64.60); GPT-3.5 F1 = 21.69; best non-GPT BART-MNLI F1 = 33.38. The high accuracy with low F1 reflects heavy class imbalance, so **F1 is the headline** and "near-random F1" is the correct characterization. On **direction specifically**: under variable refactorization "Is-Ancestor decreased to 45.45% F1 and Is-Descendant decreased to 29.41%," and non-finetuned "GPT-4 fails at all direct causation recognition with parents and children" — i.e., cannot reliably tell A→B from B→A.
- **CLADDER** (Jin et al., 2023, *CLadder: Assessing Causal Reasoning in Language Models*, NeurIPS 2023, https://arxiv.org/abs/2312.04350, HTML https://arxiv.org/html/2312.04350): **10K** samples grounded in Pearl's Ladder of Causation (associational / interventional / counterfactual), with ground truth from an oracle causal-inference engine. Their **CausalCoT** prompting lifts GPT-4 from **62.03% → 70.40%** (random baseline **49.27%**). By rung with CausalCoT: associational **83.35%**, interventional **67.47%**, counterfactual **62.05%** — monotonically harder up the ladder.

  *Takeaway:* a *causal-specific* chain-of-thought (CausalCoT) gives a real, measured gain (+8.37 points) on formal causal reasoning — concrete evidence that structured reasoning, not just bigger embeddings, moves causal accuracy.

### 3.3 Causal discovery from text and directed resources

- **CausalBank** (Li et al., 2020, *Guided Generation of Cause and Effect*, IJCAI 2020, https://www.ijcai.org/Proceedings/2020/0502.pdf, https://nlp.jhu.edu/causalbank/): ~**314M** cause–effect sentence pairs split into **133M EPC** (effect-pattern-cause) and **181M CPE** (cause-pattern-effect) — the CPE/EPC split *encodes direction lexically*. **[UNVERIFIED — read via dataset page, not PDF text.]**
- **CauseNet** (Heindorf et al., 2020, CIKM 2020, https://dl.acm.org/doi/10.1145/3340531.3412763, https://causenet.org/): a *directed* causality graph; a precision variant (~200k relations at 96% precision) and a high-recall variant (>11M relations at 83% precision). Edges are inherently cause→effect. **[UNVERIFIED — read via site/summary.]**
- **Event Causality Identification (ECI)**: survey at https://arxiv.org/html/2411.10371. *Identifying while Learning* (https://arxiv.org/abs/2405.20608) notes prior ECI "mainly focus[es] on the causality existence, but ignore[s] causal direction," and jointly learns direction, exploiting the "asymmetric and transitive" nature of causal chains.

### 3.4 How directionality is modeled and evaluated

- **Direct direction prediction** (Hosseini, Broniatowski, Diab, 2021, *Predicting Directionality in Causal Relations in Text*, https://ar5iv.labs.arxiv.org/html/2103.13606): given two spans in a causal relation, predict cause→effect vs effect→cause. On PDTB3, **BERT 0.83 F1 / SpanBERT 0.86 F1**; on EventStoryLine v1.5, **BERT 0.79 / SpanBERT 0.71**. They build **CREST**, unifying nine causal datasets (SemEval-2007/2010, BECauSE, CaTeRS, COPA, …).
- **Temporal vs causal precedence**: **CaTeRS** (Mostafazadeh et al., 2016, NAACL Workshop, https://aclanthology.org/W16-1007/) builds on **TimeML** to jointly annotate temporal (BEFORE/AFTER) and causal (CAUSE/ENABLE/PREVENT) relations over **1,600 sentences / 320 ROCStories** — explicitly separating "A happens before B" from "A causes/enables B." This distinction matters for documentation steps: *temporal order in the doc is not the same as causal precedence*, and the model must learn the latter.

**Three ways direction is operationalized:** (a) an explicit binary/ordinal label on a relation (Hosseini et al.; directional ECI); (b) directed graph edges (CauseNet; CPE/EPC in CausalBank); (c) a graph-relation type (Corr2Cause's Is-Ancestor vs Is-Descendant), where the asymmetry is exactly what LLMs fail on.

---

## 4. Framing Causal Precedence as a Reasoning / Entailment Task

The team's task — "does step A causally precede / enable step B?" — is structurally an **asymmetric, directional, multi-hop entailment** judgment. The literature above supports casting it as entailment-with-a-reasoning-trace:

**Why the entailment framing fits:**
- NLI is already directional ("P entails H" ≠ "H entails P"), matching causal asymmetry (Section 1.1).
- EFL (https://arxiv.org/abs/2104.14690) shows arbitrary tasks can be recast as entailment.
- The relation is almost always *implied*, not explicit, and INLI (https://arxiv.org/html/2501.07719v1) shows that is precisely where reasoning supervision pays off (88.5% vs ~50%).
- e-CARE (https://aclanthology.org/2022.acl-long.33/) and CLADDER (https://arxiv.org/html/2312.04350) show explanations / causal CoT improve causal accuracy.

**Suggested label scheme** (directional, mutually exclusive over an ordered pair (A, B)):

| Label | Meaning |
|---|---|
| `PRECEDES` | A causally enables / must occur before B (A → B) |
| `FOLLOWS` | B causally enables A (B → A) — the reverse |
| `NEUTRAL` | A and B are independent / co-equal, no precedence |
| (optional) `CONTRADICTS` | A and B are mutually exclusive / cannot both hold |

Evaluating both `PRECEDES` and `FOLLOWS` for the *same* pair forces the model to commit to a direction and exposes the symmetry failure that Corr2Cause flags in GPT-4. (This mirrors Is-Ancestor / Is-Descendant in Corr2Cause and forward/backward in COPA.)

**Prompt / trace format** (a VeriCoT- and CausalCoT-inspired structured trace):

```
Premise (Step A): <text of step A>
Hypothesis (Step B): <text of step B>
Question: Does Step A causally precede / enable Step B?

Reasoning:
1. Atomic effects of A: <decompose A into post-conditions>        (Atomic-SNLI style)
2. Atomic preconditions of B: <decompose B into prerequisites>
3. Grounding: which effect of A satisfies which precondition of B?  (cite doc text; flag if ungrounded — VeriCoT)
4. Direction check: would B→A also hold? If yes, relation is not strictly directional.
5. Temporal vs causal: is this merely doc ordering, or genuine enablement? (CaTeRS distinction)

Label: PRECEDES | FOLLOWS | NEUTRAL
```

The **label** is the verifiable output; the **reasoning block** is the injectable trace. Steps 1–2 borrow atomic decomposition (Section 2.5), step 3 borrows VeriCoT grounding (2.4), step 4 forces the asymmetry test (Section 3), step 5 separates temporal from causal precedence (3.4).

---

## 5. Verifiable Rewards and Reasoning Supervision

**Core thesis:** the directional/entailment label is *discrete and ground-truthed*, so a deterministic checker (exact-match of predicted label vs gold) emits a clean **0/1 reward** with no learned reward model — while the reasoning trace stays free-form. This is precisely the RLVR / DeepSeek-R1 setup.

- **RLVR** (Tülu 3, Lambert et al., AllenAI, 2024, https://arxiv.org/abs/2411.15124, blog https://allenai.org/blog/tulu-3-technical): replaces the learned reward model with a deterministic verification function — reward if the answer is verifiably correct, otherwise 0. *(Per-benchmark deltas circulated in summaries — +1.7 MATH / +3.3 GSM8K / +1.3 IFEval — are **[UNVERIFIED]** against the PDF tables.)*
- **DeepSeek-R1** (DeepSeek-AI, 2025, https://arxiv.org/abs/2501.12948, HTML https://arxiv.org/html/2501.12948v1): uses only **rule-based accuracy + format rewards**, "solely based on the correctness of final predictions against ground-truth answers, without imposing constraints on the reasoning process itself." Pure RL drove AIME 2024 pass@1 from **15.6% → 71.0%** (→ **86.7%** with majority voting). For a discrete entailment label, the accuracy check is trivial and exact.
- **GRPO** (DeepSeekMath, Shao et al., 2024, https://arxiv.org/abs/2402.03300): a PPO variant that **drops the value/critic**, using a sampled group's mean/std-normalized rewards as the baseline. It needs only a scalar reward per completion — a 0/1 label checker supplies exactly that. DeepSeekMath-7B reached **51.7% on MATH**. This is the algorithm R1 uses.

**For a sub-1B model, distillation beats direct RL.** R1's own finding (https://arxiv.org/html/2501.12948v1): "distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on … large-scale RL … may not even achieve the performance of distillation" (R1-Distill-Qwen-32B **72.6%** AIME vs RL-only Qwen-32B **47.0%**). Six dense models distilled (1.5B–70B) via SFT on ~800k R1 traces; R1-Distill-Qwen-7B reached **55.5% AIME / 92.8% MATH-500**. Supporting lineage:
- Magister et al., 2022, *Teaching Small Language Models to Reason* (https://arxiv.org/abs/2212.08410): T5-XXL on GSM8K improved **8.11% → 21.99%** when fine-tuned on PaLM-540B chains of thought.
- Shridhar et al., 2023, *Distilling Reasoning Capabilities into Smaller Language Models* (https://arxiv.org/abs/2212.00193): "Socratic CoT" decomposer + solver; reported "over 70%" relative gains *(exact per-dataset deltas **[UNVERIFIED]**)*.
- Orca 2 (Mitra et al., 2023, https://arxiv.org/abs/2311.11045): teach small LMs *multiple* solution strategies rather than pure imitation.

**Supervision over the trace (if label-only is insufficient):**
- Self-consistency (Wang et al., 2022, https://arxiv.org/abs/2203.11171): sample diverse reasoning paths, marginalize, take the majority answer — no extra training. Reported gains: **GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2%, StrategyQA +6.4%, ARC-c +3.9%**. For a discrete precedence label, majority vote over sampled traces is a free self-verifier.
- Process reward models / *Let's Verify Step by Step* (Lightman et al., 2023, https://arxiv.org/abs/2305.20050): **process** supervision (feedback per step) beats **outcome** supervision on MATH; their PRM solves **78%** of a MATH test subset; they released **PRM800K** (800k step-level human labels). PRMs are the costly opposite of verifiable rewards — needed only when the *trace* itself must be supervised. For causal precedence, the cheap verifiable *label* should be the primary signal; a PRM (or VeriCoT-style automated step checker, Section 2.4) is the upgrade path if trace faithfulness must be enforced.

**Recommended training recipe for the team's sub-1B model:**
1. Use a strong teacher (a DeBERTa-v3 NLI cross-encoder for labels; a large instruct LLM for traces) to generate `(pair, reasoning, label)` examples in the Section-4 format.
2. **Filter** generated examples by the *verifiable* gold precedence label (rejection sampling) — keep only correct-label traces.
3. **SFT** the sub-1B student on the filtered traces (the R1-distillation result says this beats direct small-model RL).
4. Optionally, a light **GRPO** pass with the 0/1 label reward to tighten the policy (R1/RLVR mechanism).
5. At inference, use **self-consistency** (sample N traces, majority-vote the label) and optionally a **VeriCoT-style** grounding check to reject ungrounded precedence claims.

---

## Datasets and Benchmarks Table

| Name | Task | Size | What it tests | Source |
|---|---|---|---|---|
| SNLI | NLI (3-way) | ~570k pairs | Sentence-level entailment/contradiction/neutral | https://arxiv.org/abs/1508.05326 |
| MultiNLI (MNLI) | NLI (3-way), multi-genre | ~433k | Cross-genre entailment; matched/mismatched split | https://aclanthology.org/N18-1101/ |
| ANLI | Adversarial NLI | R1 16,946 / R2 45,460 / R3 100,459 (train) | Model-fooling, hard entailment | https://huggingface.co/datasets/facebook/anli |
| e-SNLI | NLI + NL explanations | 549,367 / 9,842 / 9,824 | Explanation generation for entailment | https://huggingface.co/datasets/esnli/esnli |
| EntailmentBank | Multistep entailment trees | 1,840 trees | Proving a hypothesis via entailment chains | https://arxiv.org/abs/2104.08661 |
| INLI | Explicit vs implied entailment | 10k premises / 40k hypotheses | Implied (multi-hop/world-knowledge) entailment | https://arxiv.org/abs/2501.07719 |
| Atomic-SNLI | Atomic-fact-level NLI | (not stated on abstract page) | Fine-grained, fact-level entailment | https://arxiv.org/abs/2601.06528 |
| COPA | Commonsense causal choice | 1,000 (500/500) | Forward (effect) / backward (cause) plausibility | http://commonsensereasoning.org/2011/papers/Roemmele.pdf |
| e-CARE | Explainable causal reasoning | >21k questions | Cause/effect choice + conceptual explanation | https://aclanthology.org/2022.acl-long.33/ |
| Corr2Cause | Causal inference from correlation | >200k | Pure causal inference; direction (ancestor/descendant) | https://arxiv.org/abs/2306.05836 |
| CLADDER | Formal causal inference (Pearl's rungs) | 10k | Associational/interventional/counterfactual | https://arxiv.org/abs/2312.04350 |
| CausalBank | Cause–effect sentence pairs | ~314M (133M EPC / 181M CPE) | Directional causal generation [UNVERIFIED count] | https://nlp.jhu.edu/causalbank/ |
| CauseNet | Causality graph | ~200k @96% / >11M @83% | Directed cause→effect edges [UNVERIFIED count] | https://causenet.org/ |
| CaTeRS | Temporal + causal relations | 1,600 sents / 320 stories | Temporal precedence vs causal enablement | https://aclanthology.org/W16-1007/ |
| Directionality (Hosseini) / CREST | Cause→effect vs effect→cause | PDTB3 5,692/727/729; ESL 1,095/169/224 | Direction prediction on annotated spans | https://ar5iv.labs.arxiv.org/html/2103.13606 |
| PRM800K | Step-level reasoning labels | 800k labels | Process supervision for reasoning | https://arxiv.org/abs/2305.20050 |

---

## References

**NLI / RTE foundations**
- Bowman et al., 2015, *A large annotated corpus for learning natural language inference* (SNLI). https://arxiv.org/abs/1508.05326 ; https://nlp.stanford.edu/projects/snli/
- Williams, Nangia, Bowman, 2018, *A Broad-Coverage Challenge Corpus … (MultiNLI)*. https://aclanthology.org/N18-1101/ ; https://arxiv.org/abs/1704.05426
- Nie et al., 2020, *Adversarial NLI*. https://aclanthology.org/2020.acl-main.441/ ; https://arxiv.org/abs/1910.14599 ; https://huggingface.co/datasets/facebook/anli

**DeBERTa-v3 and NLI checkpoints**
- He, Gao, Chen, 2021, *DeBERTaV3*. https://arxiv.org/abs/2111.09543
- MoritzLaurer DeBERTa-v3-large NLI checkpoint. https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- cross-encoder/nli-deberta-v3-large. https://huggingface.co/cross-encoder/nli-deberta-v3-large
- Sentence-Transformers cross-encoder docs. https://www.sbert.net/examples/applications/cross-encoder/README.html

**Reasoning over entailment**
- Camburu et al., 2018, *e-SNLI*. https://arxiv.org/abs/1812.01193 ; https://huggingface.co/datasets/esnli/esnli
- Wei et al., 2022, *Chain-of-Thought Prompting*. https://arxiv.org/abs/2201.11903
- Wang et al., 2021, *Entailment as Few-Shot Learner*. https://arxiv.org/abs/2104.14690
- Dalvi et al., 2021, *Explaining Answers with Entailment Trees*. https://arxiv.org/abs/2104.08661
- Creswell et al., 2022, *Selection-Inference*. https://arxiv.org/abs/2205.09712 ; *Faithful Reasoning* https://arxiv.org/abs/2208.14271
- Feng et al., 2025, *VeriCoT*. https://arxiv.org/abs/2511.04662 ; https://arxiv.org/html/2511.04662v1
- Huang, 2026, *Atomic-SNLI*. https://arxiv.org/abs/2601.06528 ; https://arxiv.org/html/2601.06528
- Havaldar et al., 2025, *Entailed Between the Lines (INLI)*. https://arxiv.org/abs/2501.07719 ; https://arxiv.org/html/2501.07719v1

**Causal reasoning / direction**
- Roemmele et al., 2011, *COPA*. http://commonsensereasoning.org/2011/papers/Roemmele.pdf ; SemEval-2012 Task 7 https://aclanthology.org/S12-1052/ ; SuperGLUE https://arxiv.org/abs/1905.00537
- Du et al., 2022, *e-CARE*. https://aclanthology.org/2022.acl-long.33/ ; https://arxiv.org/abs/2205.05849 ; https://github.com/Waste-Wood/e-CARE
- Jin et al., 2023, *Can LLMs Infer Causation from Correlation? (Corr2Cause)*. https://arxiv.org/abs/2306.05836 ; https://arxiv.org/html/2306.05836v3
- Jin et al., 2023, *CLadder*. https://arxiv.org/abs/2312.04350 ; https://arxiv.org/html/2312.04350
- Li et al., 2020, *Guided Generation of Cause and Effect (CausalBank)*. https://www.ijcai.org/Proceedings/2020/0502.pdf ; https://nlp.jhu.edu/causalbank/
- Heindorf et al., 2020, *CauseNet*. https://dl.acm.org/doi/10.1145/3340531.3412763 ; https://causenet.org/
- ECI survey. https://arxiv.org/html/2411.10371 ; *Identifying while Learning* https://arxiv.org/abs/2405.20608
- Hosseini, Broniatowski, Diab, 2021, *Predicting Directionality in Causal Relations in Text*. https://ar5iv.labs.arxiv.org/html/2103.13606
- Mostafazadeh et al., 2016, *CaTeRS*. https://aclanthology.org/W16-1007/

**Verifiable rewards / reasoning supervision / distillation**
- Lambert et al., 2024, *Tülu 3 (RLVR)*. https://arxiv.org/abs/2411.15124 ; https://allenai.org/blog/tulu-3-technical
- DeepSeek-AI, 2025, *DeepSeek-R1*. https://arxiv.org/abs/2501.12948 ; https://arxiv.org/html/2501.12948v1
- Shao et al., 2024, *DeepSeekMath (GRPO)*. https://arxiv.org/abs/2402.03300
- Magister et al., 2022, *Teaching Small Language Models to Reason*. https://arxiv.org/abs/2212.08410
- Shridhar et al., 2023, *Distilling Reasoning Capabilities into Smaller LMs*. https://arxiv.org/abs/2212.00193
- Mitra et al., 2023, *Orca 2*. https://arxiv.org/abs/2311.11045
- Wang et al., 2022, *Self-Consistency*. https://arxiv.org/abs/2203.11171
- Lightman et al., 2023, *Let's Verify Step by Step (PRM800K)*. https://arxiv.org/abs/2305.20050

---

### Verification notes (flagged items)

- arXiv IDs confirmed to resolve to real papers: 1508.05326, 1704.05426, 1910.14599, 2111.09543, 1812.01193, 2201.11903, 2104.14690, 2104.08661, 2205.09712, **2511.04662 (VeriCoT)**, **2601.06528 (Atomic-SNLI)**, 2501.07719, 2205.05849, 2306.05836, 2312.04350, 2411.15124, 2501.12948, 2402.03300, 2212.08410, 2212.00193, 2311.11045, 2203.11171, 2305.20050.
- **Not confirmable from primary text this pass:** FEVER-NLI test accuracy (MoritzLaurer card); MNLI-matched for cross-encoder/nli-deberta-v3-large; EFL's 12%/500× figures; EntailmentBank's ~35%; Selection-Inference's >100%/280B; e-SNLI worker/explanation counts; VeriCoT and Atomic-SNLI quantitative metrics and Atomic-SNLI size; SemEval-2012 / SuperGLUE COPA accuracies; e-CARE explanation-generation BLEU/ROUGE decimals; CausalBank and CauseNet exact counts (read via dataset pages, not PDF); Tülu 3 per-benchmark deltas; Shridhar per-dataset deltas; TimeBank citation specifics; a dataset specifically named "CausalQA."
