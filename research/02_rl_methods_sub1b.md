# RL and Training Methods for Reasoning in Sub-1B Language Models

**Scope:** Methods that improve *reasoning* in small (sub-1B-parameter) LLMs, evaluated against a concrete downstream use case: a **causal-direction reasoning model** that predicts whether `text_1` causally precedes `text_2`, then reranks/orders workflow steps. Training budget: **~100K–260K labeled pairs, 4×A100**.

**Date:** 2026-06-30. Every non-obvious quantitative claim is cited inline with a source URL. Where a claim could not be verified against a primary source, this is stated explicitly.

---

## Executive Summary

1. **GRPO is the dominant RL algorithm for reasoning** because it drops PPO's value/critic network and computes advantages from the *relative* reward inside a group of sampled completions, halving memory and removing a hard-to-train component ([DeepSeekMath, arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)).

2. **Vanilla GRPO has two well-documented optimization biases** — response-length normalization and per-question std normalization — that inflate response length and destabilize training. **Dr. GRPO** removes both ([Understanding R1-Zero-Like Training, arXiv:2503.20783](https://arxiv.org/pdf/2503.20783)); **DAPO** adds Clip-Higher (fixes entropy collapse), dynamic sampling, token-level loss, and overlong-reward shaping ([DAPO, arXiv:2503.14476](https://arxiv.org/html/2503.14476v1)).

3. **For the sub-1B regime, the strongest published evidence says: distillation (SFT on reasoning traces from a stronger model) first, then optionally a *short* RL polish.** DeepSeek explicitly found that distilling a large model into small dense models beats running large-scale RL directly on the small model ([DeepSeek-R1 model card, HF](https://huggingface.co/deepseek-ai/DeepSeek-R1)). Pure RL "from scratch" on a weak small base is compute-hungry and may never reach distillation quality.

4. **"SFT memorizes, RL generalizes" is real but conditional**: RL with outcome reward generalizes out-of-distribution better than SFT, *but SFT is still needed first* to stabilize output format before RL pays off ([arXiv:2501.17161](https://arxiv.org/abs/2501.17161)).

5. **Recommendation for the causal-ordering task:** Treat it primarily as a **verifiable-reward** problem (the gold direction/order is known), do **SFT on reasoning traces → GRPO/Dr.GRPO with a format+correctness reward** (and an NDCG/Kendall-tau style reward for the ranking variant). On 4×A100 with a sub-1B model, **full fine-tune is feasible**, but LoRA-GRPO (the "Tina" recipe) is a very cheap, competitive alternative ([Tina, arXiv:2504.15777](https://arxiv.org/abs/2504.15777)).

---

## 1. GRPO (Group Relative Policy Optimization)

### Core mechanism
GRPO was introduced in **DeepSeekMath** ([arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)) and scaled in **DeepSeek-R1** ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)). It is a PPO variant designed for LLM reasoning.

- **No critic.** Standard PPO is actor-critic: it trains a separate value network the same size as the policy to estimate a per-token baseline. GRPO **removes the value model entirely**, which avoids learning a value function from an LM backbone and saves the memory of a second full-size network ([DeepSeekMath, arXiv:2402.03300](https://arxiv.org/pdf/2402.03300); [Cameron Wolfe, GRPO](https://cameronrwolfe.substack.com/p/grpo)).
- **Group sampling.** For each prompt `q`, sample a *group* of `G` completions `{o_1...o_G}` from the current (or old) policy.
- **Group-relative advantage.** Score each completion with a reward `r_i`. The advantage is the within-group standardized reward:

  `A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)`

  Every token in completion `o_i` gets this same scalar advantage ([DeepSeekMath, arXiv:2402.03300](https://arxiv.org/pdf/2402.03300); [HF LLM course ch.12](https://huggingface.co/learn/llm-course/chapter12/3b)). The group mean acts as the baseline that the critic used to provide in PPO.
- **Clipped surrogate + KL.** GRPO keeps PPO's clipped importance-sampling objective, and adds a **KL penalty to a frozen reference policy** (the SFT model) directly in the loss to prevent drift ([DeepSeekMath, arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)).

### Reward design
DeepSeek-R1-Zero used **rule-based, verifiable rewards**: accuracy reward (is the final answer correct, checked by a parser/checker) + format reward (did it put reasoning inside `<think>...</think>` and the answer in the required place), deliberately avoiding a learned reward model to prevent reward hacking ([DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).

### Data needs
Prompts with **verifiable answers** (math, code, or any task with a checkable label). No human preference labels required — this is "RL from Verifiable Rewards" (RLVR). The downstream causal-direction task fits this perfectly: the gold causal direction / step order is the verifier.

### Compute
Cheaper than PPO per step (no critic forward/backward, no critic optimizer state), but you pay for **`G` generations per prompt** — generation (rollout) dominates wall-clock. Effective batch ≈ `prompts × G`.

### Pros / Cons
- **Pros:** no critic (less memory, fewer moving parts), strong on verifiable reasoning, simple reward.
- **Cons:** rollout cost scales with `G`; vanilla form has length/entropy pathologies (see §2); high variance when group rewards are degenerate (all-correct or all-wrong groups give zero advantage).

---

## 2. GRPO Variants and Fixes (2025–2026)

### Dr. GRPO — "GRPO Done Right"
From **Understanding R1-Zero-Like Training: A Critical Perspective** ([arXiv:2503.20783](https://arxiv.org/pdf/2503.20783); [code: sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)). Identifies **two biases** in vanilla GRPO:

1. **Response-length normalization** — dividing each sequence loss by its token count biases updates toward longer responses; observed empirically as response length climbing even after reward plateaus, *especially for wrong answers* ([Dr. GRPO summary, EmergentMind](https://www.emergentmind.com/topics/dr-grpo)).
2. **Question-level std normalization** — dividing the advantage by the per-question reward std over-weights easy/hard questions and adds variance.

**Dr. GRPO fix:** (a) normalize the summed sequence loss by a **fixed constant** instead of per-sequence length; (b) **remove the std term** from the advantage denominator (use `r_i − mean` only). Result: fixes the bias while keeping reasoning accuracy and **reduces average incorrect-response length by ~38%** (better token efficiency) ([arXiv:2503.20783](https://arxiv.org/pdf/2503.20783)). *Caveat:* a community note argues the Dr.GRPO formulation can still implicitly weight by completion length in some setups and may need explicit counteracting ([MAD GRPO blog, HF](https://huggingface.co/blog/telcom/mad-grpo)) — treat as a secondary, not fully settled, claim.

### DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization
ByteDance Seed + Tsinghua, fully open-source ([arXiv:2503.14476](https://arxiv.org/html/2503.14476v1); [site](https://dapo-sia.github.io/); [code](https://github.com/BytedTsinghua-SIA/DAPO)). Four techniques on top of GRPO:

1. **Clip-Higher** — decouple the PPO clip range into `ε_low=0.2`, `ε_high=0.28`, giving low-probability tokens more room to grow. **Fixes entropy collapse** (premature determinism / loss of exploration) ([arXiv:2503.14476](https://arxiv.org/html/2503.14476v1)).
2. **Dynamic Sampling** — discard prompts whose group is all-correct or all-wrong (advantage = 0, zero gradient); oversample to keep batches full of informative `0 < accuracy < 1` prompts. Speeds convergence.
3. **Token-Level Policy Gradient Loss** — normalize over total tokens in the batch rather than averaging per sequence, so long sequences aren't under-weighted; reduces unhealthy entropy growth and repetition.
4. **Overlong Reward Shaping** — soft, graduated penalty for over-length/truncated responses (a "punishment cache" zone) instead of a harsh binary penalty, reducing reward noise.

DAPO also **removes the KL term**, uses rule-based reward, and reaches **AIME-2024 = 50** with Qwen2.5-32B base, beating DeepSeek-R1-Zero-Qwen-32B (47) with **50% of the training steps** ([arXiv:2503.14476](https://arxiv.org/html/2503.14476v1)).

### S-GRPO — Serial-Group Decaying-Reward Policy Optimization
([arXiv:2505.07686](https://arxiv.org/abs/2505.07686)). Targets **overthinking / CoT redundancy**, not raw accuracy. Instead of sampling parallel completions, it samples **one** reasoning path and serially picks multiple *exit positions* along it; correct early exits get **higher decaying reward** than later ones. This teaches the model to *stop thinking early when confident*. Reported **35.4%–61.1% sequence-length reduction with +0.72% to +6.08% accuracy** across GSM8K/AIME24/AMC23/MATH-500/GPQA ([arXiv:2505.07686](https://arxiv.org/abs/2505.07686)). Relevant if inference latency on the ordering task matters.

### Summary of what each fixes
| Problem in vanilla GRPO | Fix | Source |
|---|---|---|
| Response-length inflation (length bias) | Dr. GRPO (constant-norm loss); DAPO token-level loss | [2503.20783](https://arxiv.org/pdf/2503.20783), [2503.14476](https://arxiv.org/html/2503.14476v1) |
| Per-question std variance | Dr. GRPO (drop std) | [2503.20783](https://arxiv.org/pdf/2503.20783) |
| Entropy collapse (no exploration) | DAPO Clip-Higher | [2503.14476](https://arxiv.org/html/2503.14476v1) |
| Zero-gradient (all-right/all-wrong groups) | DAPO dynamic sampling | [2503.14476](https://arxiv.org/html/2503.14476v1) |
| Overthinking / verbose CoT | S-GRPO decaying-reward early exit | [2505.07686](https://arxiv.org/abs/2505.07686) |

---

## 3. PPO vs DPO vs GRPO for Reasoning

| Method | Needs reward model? | Needs critic/value net? | Data | Best for |
|---|---|---|---|---|
| **PPO** | Yes (or rule reward) | **Yes** (separate value net) | prompts + reward signal | Stable optimization with a learned baseline; classic RLHF |
| **DPO** | **No** (implicit) | No | **pairwise preferences** (chosen/rejected) | Cheap preference/style/rationale tuning; no RL loop |
| **GRPO** | No (rule/verifiable) | **No** (group baseline) | prompts + **verifiable** reward | Verifiable reasoning (math/code/ordering) |

**Key distinctions and when to use each:**
- **PPO** is actor-critic; the critic provides a lower-variance, state-dependent baseline but is unstable and needs a warm-up. PPO has been shown to **outperform DPO** on reasoning/coding/safety (avg +1.3/+2.9/+2.3 points) in a controlled study ([Unpacking DPO and PPO, arXiv:2406.09279](https://arxiv.org/html/2406.09279v1)). Choose PPO when you want a learned value baseline and can afford the second network.
- **DPO** skips the RL loop and reward model entirely, directly optimizing a classification-style loss on **(chosen, rejected) pairs**. Ideal for **preference or rationale tuning** when you have pairwise labels and want simplicity/stability. It does *not* explore (no sampling/reward), so it cannot discover new reasoning behaviors the way RL can.
- **GRPO** is the middle ground for **verifiable reasoning**: no reward model, no critic, learns from self-generated samples scored by a checker. A 2025/2026 comparative study finds GRPO converges faster and reaches a higher reward ceiling than PPO on complex multi-hop reasoning, while PPO can edge it on raw accuracy in simpler settings due to its value baseline ([Comparative Analysis of PPO/GRPO/DAPO, arXiv:2512.07611](https://arxiv.org/html/2512.07611v1)). *(Note: this comparative paper is recent; treat its exact deltas as indicative.)*

**For the causal-ordering task:** Your labels are *directional/ordering ground truth*, i.e. **verifiable**, not subjective preferences. That points to **GRPO/Dr.GRPO** for the RL stage. DPO is a reasonable, cheaper alternative if you instead frame data as preference pairs (correct order ≻ corrupted order) and want to skip sampling.

---

## 4. The Key Question — Does Pure RL Work Sub-1B, or Is Distillation-Then-RL Required?

**Verdict (evidence-based): For sub-1B reasoning, do distillation-style SFT first; pure from-scratch RL on a weak small base is not reliable and is compute-inefficient. A short RL polish *after* SFT adds generalization.**

Supporting evidence:

- **DeepSeek-R1's own conclusion.** DeepSeek states distilling reasoning patterns from a larger model into smaller dense models yields **"better performance compared to the reasoning patterns discovered through RL on small models."** They directly compared a distilled 32B model to **DeepSeek-R1-Zero-Qwen-32B** (RL applied directly to the base) and concluded distillation wins, while small-model large-scale RL **"requires enormous computational power and may not even achieve the performance of distillation"** ([DeepSeek-R1 model card, HF](https://huggingface.co/deepseek-ai/DeepSeek-R1); [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)). The distilled models were SFT'd on **800K samples** curated from R1 (≈600K reasoning + 200K non-reasoning).

- **"RL for Reasoning in Small LLMs: What Works and What Doesn't" (Open-RS).** Starting from the **already-distilled DeepSeek-R1-Distill-Qwen-1.5B**, GRPO on **7,000 math samples**, 4×A40 (48GB), ~24h, **~$42** lifted AMC23 63%→80% and AIME24 to 46.7% (beating o1-preview's 44.6%) ([arXiv:2503.16219](https://arxiv.org/html/2503.16219v2); [code](https://github.com/knoveleng/open-rs)). **Crucial nuance:** the base was *already distilled*; RL was a cheap polish, not from-scratch. They also report **degradation after ~150–200 steps, KL instability, length-limit truncation (3,584–4,096 tokens), and multilingual drift** — i.e., small-model RL is fragile ([arXiv:2503.16219](https://arxiv.org/html/2503.16219v2)).

- **Tina: Tiny Reasoning Models via LoRA** ([arXiv:2504.15777](https://arxiv.org/abs/2504.15777)). GRPO via **LoRA** on DeepSeek-R1-Distill-Qwen-1.5B: **>20% reasoning gain, 43.33% Pass@1 on AIME24, ~$9** to reproduce the best checkpoint (≈260× cost reduction). Again: distilled base + cheap RL adapter.

- **"SFT Memorizes, RL Generalizes"** ([arXiv:2501.17161](https://arxiv.org/abs/2501.17161), ICML'25). RL with outcome reward generalizes to unseen rule/visual variants better than SFT (which memorizes), **but the paper explicitly states SFT remains necessary first to stabilize output format so that RL can work.** This is the canonical justification for the **SFT→RL** order.

**Bottom line for sub-1B:** A weak <1B base typically lacks the latent reasoning for RL to "discover" much on its own (the all-wrong groups give zero gradient). Seed it with SFT on high-quality reasoning traces (distillation), *then* a short, well-regularized GRPO/Dr.GRPO run to generalize and sharpen. Budget RL to **tens-to-low-hundreds of steps** and watch for the instability the Open-RS authors documented.

---

## 5. Reward Design for Verifiable / Structured Tasks

### General RLVR pattern (format + correctness)
DeepSeek-R1's recipe — and the de-facto standard — is a **sum of independent reward terms** ([DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948); [TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer)):

- **Format reward** — e.g., `+1` if the output matches the required schema (reasoning in `<think>`, answer as a parseable token like `BEFORE`/`AFTER` or a permutation), else `0`. Stabilizes parsing.
- **Correctness reward** — `+1` if the verifier says the answer matches gold, else `0` (or `−1`). Checked by a deterministic function, no reward model.

In TRL you pass a **list of reward functions**; the total reward is their (optionally weighted) sum ([TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer); [grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)).

### For the causal-direction (binary) task
- Correctness: `+1` if predicted direction == gold (`text_1 → text_2` vs `text_2 → text_1`), else `0`.
- Format: `+0.1`–`+0.5` for emitting the exact answer tag.
- Optional: small **length / overthinking penalty** (cf. S-GRPO / DAPO overlong shaping) to avoid runaway CoT on a binary decision.

### For ranking / ordering (reranking workflow steps)
This is an *ordering* output, so use an **order-sensitive verifiable reward** rather than 0/1:
- **NDCG-style or rank-based reward** is the established choice for RL rerankers: **REARANK** and **Rank-R1** train listwise LLM rerankers with **GRPO using NDCG-normalized group rewards** ([REARANK, arXiv:2505.20046](https://arxiv.org/pdf/2505.20046)). **Rank-GRPO** generalizes GRPO to treat each *rank position* as the unit of advantage estimation for rank-style verifiable outputs ([Rank-GRPO, arXiv:2510.20150](https://arxiv.org/pdf/2510.20150)).
- For step *ordering* specifically, a natural verifiable reward is **Kendall's tau or Spearman correlation** between the predicted permutation and the gold workflow order (continuous in [−1, 1], dense gradient signal), optionally plus a `+1` exact-permutation bonus. (This is a standard rank-correlation metric; the *application* as a GRPO reward is an engineering choice consistent with the NDCG-reward precedent above rather than a single cited result.)
- Keep a **format reward** so the model emits a parseable permutation/ranked list.

**Design principle:** dense, order-aware rewards (NDCG/Kendall-tau) give more gradient signal than sparse exact-match, which matters with small models where many groups would otherwise be all-wrong (zero advantage). Combine with **DAPO dynamic sampling** to drop degenerate groups.

---

## 6. Practical Recipe, Libraries, and Compute

### Libraries
- **TRL `GRPOTrainer`** — easiest entry; native support for **custom reward functions (list)**, LoRA/PEFT, and vLLM-backed generation ([TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer); [source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)). Best for a single 4×A100 node and sub-1B models.
- **verl (HybridFlow)** — ByteDance's high-throughput RL framework; its **HybridEngine** swaps weights in-place between FSDP training and vLLM inference on the same GPUs, giving **1.5×–20× throughput** over OpenRLHF v0.2.5 on a 16×A100 benchmark; supports GRPO/PPO/DAPO ([verl repo](https://github.com/verl-project/verl); [verl GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)). Use if you scale up or want DAPO out of the box.
- **OpenRLHF** — Ray + vLLM, separates Actor/Reward/Reference/Critic across GPUs, scales to 70B+; supports PPO/DAPO/REINFORCE++ ([OpenRLHF repo](https://github.com/OpenRLHF/OpenRLHF)). More than needed for sub-1B but battle-tested.

### LoRA vs full fine-tune
- For sub-1B on 4×A100 (80GB), **full fine-tune is entirely feasible** and removes adapter-merge complexity.
- **LoRA-GRPO is the proven cheap path**: Tina got SOTA-competitive 1.5B reasoning for **~$9** with LoRA ([arXiv:2504.15777](https://arxiv.org/abs/2504.15777)). Recommended hyperparameters seen in practice: **rank r=16, alpha=16, dropout=0.05** ([TRL/Unsloth GRPO examples](https://huggingface.co/docs/trl/grpo_trainer)). Start with LoRA to iterate cheaply; switch to full FT only if LoRA plateaus.

### Typical GRPO hyperparameters (TRL-style, adapt to your task)
From TRL docs/examples ([TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer)):
- `learning_rate` ≈ **5e-6** (RL stage; SFT stage higher, e.g. 1e-5).
- **`num_generations` (group size G) = 4–16.** G=2–3 lacks reward diversity; 4–16 balances signal vs cost; bigger G = better learning, more rollout cost.
- `per_device_train_batch_size` small (1–4) with gradient accumulation.
- `max_prompt_length` / `max_completion_length` sized to your task — for binary causal decisions a few hundred tokens suffice; for CoT ordering allow more, but **do not under-size** it (Open-RS truncation lesson, [arXiv:2503.16219](https://arxiv.org/html/2503.16219v2)).
- KL penalty `β`: keep a modest KL to the SFT reference for stability (or drop it à la DAPO once stable, [arXiv:2503.14476](https://arxiv.org/html/2503.14476v1)).
- Apply **Dr. GRPO** loss normalization (constant, no std) to avoid length bias ([arXiv:2503.20783](https://arxiv.org/pdf/2503.20783)).

### Compute / time on A100-class hardware
- **Reference point:** GRPO on a distilled 1.5B model, 7K samples, 1 epoch ≈ **24h on 4×A40 (48GB)**; on 4×A100 (80GB) this is comparable-to-faster ([Open-RS, arXiv:2503.16219](https://arxiv.org/html/2503.16219v2)).
- **Rollout dominates**: use **vLLM-backed generation** (TRL/verl support it) to keep GPUs busy. Effective compute ≈ `steps × prompts × G × completion_length`.
- For sub-1B with 100K–260K pairs, expect SFT to be a few hours and a *short* RL polish (tens–low-hundreds of steps) to be the cost-effective regime — going long invites the documented instability.

---

## Method Comparison Table

| Method | Type | Critic | Reward source | Key idea | Fixes / Strength | Source |
|---|---|---|---|---|---|---|
| **PPO** | On-policy RL | Yes | reward model or rule | Clipped surrogate + learned value baseline | Low-variance baseline; stable | [2406.09279](https://arxiv.org/html/2406.09279v1) |
| **DPO** | Offline preference | No | implicit (pairs) | Direct loss on chosen≻rejected | No RL loop, cheap, stable | [2406.09279](https://arxiv.org/html/2406.09279v1) |
| **GRPO** | On-policy RL | **No** | rule/verifiable | Group-relative advantage `(r−mean)/std` | No critic; strong on verifiable reasoning | [2402.03300](https://arxiv.org/pdf/2402.03300) |
| **Dr. GRPO** | GRPO fix | No | verifiable | Constant-norm loss, drop std | −38% wrong-answer length; fixes length bias | [2503.20783](https://arxiv.org/pdf/2503.20783) |
| **DAPO** | GRPO fix | No | rule, no KL | Clip-Higher + dyn. sampling + token loss + overlong shaping | Fixes entropy collapse; 50% fewer steps | [2503.14476](https://arxiv.org/html/2503.14476v1) |
| **S-GRPO** | GRPO fix | No | decaying serial reward | Reward early exit on a single path | −35–61% length, +acc; less overthinking | [2505.07686](https://arxiv.org/abs/2505.07686) |
| **Rank-GRPO / REARANK** | GRPO for ranking | No | NDCG / rank | Rank-level advantage / NDCG reward | Listwise reranking with reasoning | [2510.20150](https://arxiv.org/pdf/2510.20150), [2505.20046](https://arxiv.org/pdf/2505.20046) |

---

## Recommendation for Sub-1B Causal-Direction + Ordering Reasoning

**Pipeline (SFT → GRPO), on 4×A100 with a sub-1B base:**

1. **Pick a distilled or reasoning-capable sub-1B base** (e.g., a small Qwen/Llama distill). Do **not** start RL from a raw weak base — DeepSeek's own result says small-model from-scratch RL underperforms distillation and is compute-hungry ([DeepSeek-R1, HF card](https://huggingface.co/deepseek-ai/DeepSeek-R1)).

2. **SFT (distillation) stage.** Generate or use reasoning traces for your causal-direction and ordering pairs (have a strong model produce `<think>` rationales + the gold answer), SFT the sub-1B model on them. This stabilizes format and seeds reasoning — the prerequisite that makes RL effective ([SFT Memorizes RL Generalizes, arXiv:2501.17161](https://arxiv.org/abs/2501.17161)).

3. **RL stage — GRPO with Dr.GRPO fixes** (TRL `GRPOTrainer`, vLLM rollouts, LoRA r=16 first):
   - **Causal-direction reward** = correctness (`+1` if predicted direction == gold) + format (`+0.1–0.5`).
   - **Ordering/rerank reward** = NDCG or Kendall-tau between predicted permutation and gold order (dense), + exact-permutation bonus + format. Precedent: GRPO + NDCG-normalized rewards for listwise rerankers ([REARANK, arXiv:2505.20046](https://arxiv.org/pdf/2505.20046); [Rank-GRPO, arXiv:2510.20150](https://arxiv.org/pdf/2510.20150)).
   - Use **Dr.GRPO normalization** (no length bias) and **DAPO dynamic sampling** (drop all-right/all-wrong groups — important because dense rank rewards still produce degenerate easy/hard groups).
   - `num_generations` 8 (try 4–16), lr 5e-6, modest KL to the SFT reference, keep it **short (tens–low-hundreds of steps)** and monitor for KL blow-up / length drift ([Open-RS, arXiv:2503.16219](https://arxiv.org/html/2503.16219v2)).

4. **If latency matters**, add an S-GRPO-style early-exit / overthinking penalty so the model doesn't over-reason a binary decision ([arXiv:2505.07686](https://arxiv.org/abs/2505.07686)).

5. **Cheaper alternative / ablation:** frame the data as preference pairs (gold order ≻ corrupted order) and run **DPO** — no sampling, very stable, good when you want to skip the RL loop. Use it as a baseline against GRPO.

**Why this and not pure RL:** Your dataset (100K–260K pairs) is large enough for a strong SFT/distillation stage, and the labels are verifiable — the ideal setup for SFT→GRPO. The literature is consistent that, at sub-1B scale, the distillation seed is what makes the subsequent (cheap) RL pay off, and that long unguided RL on small models is unstable and may never match the distilled ceiling.

---

## References

- DeepSeekMath / GRPO origin — arXiv:2402.03300: https://arxiv.org/pdf/2402.03300
- DeepSeek-R1 — arXiv:2501.12948: https://arxiv.org/abs/2501.12948
- DeepSeek-R1 model card (distillation vs RL conclusion, 800K SFT) — https://huggingface.co/deepseek-ai/DeepSeek-R1
- Understanding R1-Zero-Like Training / **Dr. GRPO** — arXiv:2503.20783: https://arxiv.org/pdf/2503.20783 ; code: https://github.com/sail-sg/understand-r1-zero
- Dr. GRPO summary — https://www.emergentmind.com/topics/dr-grpo
- MAD GRPO critique of Dr.GRPO (secondary, unsettled) — https://huggingface.co/blog/telcom/mad-grpo
- **DAPO** — arXiv:2503.14476: https://arxiv.org/html/2503.14476v1 ; site: https://dapo-sia.github.io/ ; code: https://github.com/BytedTsinghua-SIA/DAPO
- **S-GRPO** (early exit) — arXiv:2505.07686: https://arxiv.org/abs/2505.07686
- RL for Reasoning in Small LLMs / **Open-RS** — arXiv:2503.16219: https://arxiv.org/html/2503.16219v2 ; code: https://github.com/knoveleng/open-rs
- **Tina: Tiny Reasoning Models via LoRA** — arXiv:2504.15777: https://arxiv.org/abs/2504.15777 ; code: https://github.com/shangshang-wang/Tina
- **SFT Memorizes, RL Generalizes** — arXiv:2501.17161: https://arxiv.org/abs/2501.17161 ; site: https://tianzhechu.com/SFTvsRL/
- Unpacking DPO and PPO — arXiv:2406.09279: https://arxiv.org/html/2406.09279v1
- Comparative Analysis of PPO/GRPO/DAPO (recent, treat deltas as indicative) — arXiv:2512.07611: https://arxiv.org/html/2512.07611v1
- REARANK (GRPO + NDCG reranking) — arXiv:2505.20046: https://arxiv.org/pdf/2505.20046
- Rank-GRPO — arXiv:2510.20150: https://arxiv.org/pdf/2510.20150
- TRL GRPOTrainer docs — https://huggingface.co/docs/trl/grpo_trainer ; source: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
- verl (HybridFlow) — repo: https://github.com/verl-project/verl ; GRPO docs: https://verl.readthedocs.io/en/latest/algo/grpo.html
- OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
- GRPO explainer (Cameron Wolfe) — https://cameronrwolfe.substack.com/p/grpo
- HF LLM course, GRPO chapter — https://huggingface.co/learn/llm-course/chapter12/3b

**Verification notes / caveats:**
- The exact AIME/MATH per-model numbers for DeepSeek-R1-Distill-Qwen-1.5B were not re-fetched from the R1 PDF in this pass; the distillation-vs-RL *conclusion* is confirmed via the R1 model card and corroborating sources, but treat specific 1.5B benchmark digits as needing a direct PDF check before quoting.
- The Kendall-tau-as-GRPO-reward suggestion (§5) is an engineering recommendation consistent with the NDCG-reward precedent, not a single cited experimental result.
- arXiv:2512.07611 (PPO/GRPO/DAPO comparison) is recent; its exact point deltas are indicative.
