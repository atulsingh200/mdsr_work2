# Feasibility Study: Latent / Recursive Reasoning Models (HRM, TRM) for AEP/AJO Causal Workflow Ordering

**Date:** 2026-06-30  **Scope:** feasibility / analysis only — no heavy training in this session.

---

## 0. TL;DR verdict

**Worth pursuing — but narrowly, as a *learned rank-aggregation head* (an "ordering solver"), not as a
replacement for the cross-encoder, and not as the primary plan.**

A TRM-style tiny recursive network (5–7M params) sitting **on top of a frozen cross-encoder** is the
natural fit: the cross-encoder produces a calibrated `N×N` pairwise relation tensor, and the recursive
reasoner iteratively refines a global ordering / soft-permutation in latent state — structurally the same
move TRM/HRM make on Sudoku and ARC grids. This directly attacks the **aggregation** stage (the documented
0/30 collapse), as a *learned* alternative/complement to MFAS/Kemeny.

But three hard caveats keep this in **Track-B / research** status, not Track-A:

1. The **ARC Prize team's own ablation** shows HRM's gains come from the **outer refinement loop + heavy
   per-task augmentation + transductive `puzzle_id` memorization**, *not* the fancy hierarchy — and the
   setup is essentially **transductive memorization with near-zero cross-task transfer**. Our deployment is
   the opposite: we want **inductive** generalization to unseen workflows.
2. **Data regime mismatch.** HRM/TRM need ~1000 examples *per puzzle distribution* with 300+ augmentations
   and deep supervision. We have 100K+ *pairs* but **few full-workflow gold orders** (≈30 in the AJO test).
   The thing the recursive model would learn to do (global ordering) is exactly the thing we have least
   labeled data for.
3. The survey's existing Tier-1 fix (**calibrated weighted MFAS/Kemeny**) already converts the
   cross-encoder's ~0.96 pairwise signal into **all-workflows-correct** in the reference implementation,
   at ~1 day of work and zero training. A learned solver has to *beat a solved problem* — its value is only
   in the regime where the pairwise signal is **decent-but-noisy and structurally cyclic**, and where we want
   the model to **jointly reason** over steps rather than aggregate independent pairwise calls.

So: **build Tier-1 (MFAS/Kemeny) first; prototype the TRM-orderer as a research bet** on closing the gap
between "noisy pairwise" and "correct global order" with a single jointly-trained module, and as a
**listwise upgrade path** (it subsumes the Tier-3 "listwise / global ordering" idea).

---

## 1. The project problem (recap, grounded)

Sub-1B model over AEP / AJO docs+tutorials doing three tasks built on one pairwise primitive
`P(text_1 causally precedes text_2)`:

1. **Pairwise causal direction** (binary/3-way).
2. **Follow-up reranking** (MRR / Recall@K).
3. **Workflow ordering** — order a 3–12 step workflow (the real goal; **currently 0/30**).

Decisive repo facts (from `research/01`, `research/08`, `sections/11`):

- Cross-encoder (DeBERTa-v3-large) orders **21/30**; bi-encoder **0/30**.
- On the *real* scored AJO workflows, the **bi-encoder pairwise signal is ~0.41** (worse than chance after
  removing input-order leakage) -> **no aggregator can save it**. The **cross-encoder is ~0.96** -> **every**
  aggregator (MFAS / Kemeny / Bradley-Terry) then orders all evaluated workflows correctly.
- **Naive topological sort collapses on a single cycle**; noisy pairwise scores always produce cycles.
- Data is **adjacent-only, within-doc positional, forced total order, no NEUTRAL** -> distribution shift +
  forced edges between parallel steps.
- Hardware 4xA100 + ~1TB CPU; **CPU OOM crashes the box**; no heavy training this session.

The survey's recommended system is: **CE + calibrate + MFAS/Kemeny (Tier 1) -> distilled rationale head /
Qwen3-0.6B think-free (Tier 2) -> tau-reward GRPO + listwise (Tier 3)**.

---

## 2. The methods (accurate summaries from primary sources)

### 2.1 HRM — Hierarchical Reasoning Model (Wang et al., Sapient Intelligence, 2025; arXiv:2506.21734)

- **Architecture:** two coupled recurrent modules — a **high-level (H)** module for slow abstract planning
  and a **low-level (L)** module for fast detailed computation; the L-module iterates to a local equilibrium
  conditioned on the current H-state, then H updates ("**hierarchical convergence**"). One forward pass
  performs many internal reasoning steps with no explicit step supervision.
- **Size / data:** **~27M params**, trained on **~1000 examples** per task, **no pre-training, no CoT data**.
- **Training tricks:** **deep supervision** (repeated forward segments with carried state, loss each
  segment); a **1-step gradient approximation** (treat the converged inner state as a fixed point and
  backprop ~one step, a la implicit/DEQ models) to avoid BPTT through the whole unroll; **ACT / Q-learning
  halting** to decide how many segments to run.
- **Results (paper):** near-perfect on **Sudoku-Extreme** and **Maze-Hard**; competitive on **ARC-AGI-1**
  (~40%) and **ARC-AGI-2** (~5%) — beating much larger LLMs at a tiny parameter count.

### 2.2 The ARC Prize team's ablation of HRM — *be honest about this* (arcprize.org/blog/hrm-analysis)

The ARC Prize team re-implemented and ablated HRM. Findings (these reshape the whole story):

- **The hierarchy barely matters.** A plain same-size Transformer comes **within ~5pp** of HRM. The
  "planner/worker" H-L split is not where the performance lives.
- **The outer refinement loop is the real driver.** Going from 1 -> 2 outer loops gives **+13pp**; 1 -> 8
  loops **roughly doubles** Public-Eval performance (e.g. ~18.6% -> ~35.5% -> ~38-39%).
- **Training-time refinement >> inference-time refinement.** Models *trained* with 16 refinement steps then
  run with 1 loop still gain **>15pp** vs. models trained with 1 — the loop teaches a better solver, it's
  not just test-time compute.
- **Augmentation is essential but cheap.** **~300** augmentations reach near-max (the paper's 1000 are
  overkill); even ~30 come within ~4%.
- **It is transductive memorization.** Training on **only the 400 eval tasks** still reaches **31% pass@2**
  vs 41% — i.e. cross-task transfer is small; performance is largely **memorizing solutions for the specific
  eval tasks**. The **`puzzle_id` embedding** means the model **only works on puzzle ids seen in training**;
  the underlying "program" stays implicit. The authors call this a **major limitation**.

**Implication for us:** the *useful, transferable* nugget of HRM is **"train a recurrent solver with an
outer refinement loop + augmentation under deep supervision,"** not the biological hierarchy and not the
transductive puzzle-id setup. Any design we adopt must avoid the memorization trap by being **inductive over
workflow content** (use real step embeddings, never a per-workflow id embedding).

### 2.3 TRM — Tiny Recursive Model ("Less is More", Jolicoeur-Martineau, Samsung SAIL, 2025; arXiv:2510.04871)

TRM is HRM stripped to its working core and made stronger:

- **One tiny network, 2 layers, ~5-7M params** — recurses over a triple **(x, y, z)**: embedded question
  `x`, current answer `y`, latent `z`.
- **Recursion:** `latent_recursion` runs, by default, `n=6` inner steps of `z <- net(x+y+z)` then one
  `y <- net(y+z)`; `deep_recursion` runs `T=3` such blocks (`T-1` grad-free to pre-converge, then 1 with
  gradients), repeated under **deep supervision** for up to **16** steps, detaching state between steps.
- **Simplifications vs HRM:** **drops** the biological two-timescale story **and** the fixed-point /
  Implicit-Function-Theorem justification — TRM **backprops through the full recursion** (cleaner, and the
  ablation shows the **1-step gradient approximation actually hurts**: 87.4% -> 56.5% on Sudoku-Extreme).
  Replaces HRM's dual ACT loss with a **single binary halting loss** (one forward pass instead of two).
  Uses **EMA** (0.999) to stay stable on tiny data (-7.5pp without it). For small fixed grids it **drops
  self-attention** in favor of an **MLP-mixer** token-mix (`[L,L]` linear is cheap when `L <= D`), which
  *helps* on 9x9 Sudoku (with self-attention: 74.7% vs 87.4% MLP) but the attention variant is better on
  large/long grids and on ARC.
- **Results:** **Sudoku-Extreme 87.4%** (HRM 55.0%), **Maze-Hard 85.3%** (HRM 74.5%), **ARC-AGI-1 44.6%**
  (HRM 40.3%), **ARC-AGI-2 7.8%** (HRM 5.0%) — beating Gemini 2.5 Pro / o3-mini on these grids with
  <0.01% of their parameters.
- **Why it generalizes from tiny data:** small net + deep recursion + deep supervision + EMA controls
  overfitting; recursion supplies *effective depth* (~42) without parameters. **Caveat:** this is on the
  **same transductive, per-puzzle, augmentation-heavy** regime as HRM — the ARC-team critique of the
  *setup* still applies.

### 2.4 The broader latent / iterative-reasoning family (context for the survey)

| Method | Idea | Relevance here |
|---|---|---|
| **Adaptive Computation Time** (Graves 2016, arXiv:1603.08983) | per-step halting probability -> variable "ponder" steps in an RNN | the ancestor of HRM/TRM's ACT halting |
| **Universal Transformers** (Dehghani et al. 2019, arXiv:1807.03819) | weight-tied recurrent-in-depth Transformer + per-token ACT halting | recurrence-in-depth = "think longer with same params" |
| **PonderNet** (Banino et al. 2021, arXiv:2107.05407) | reformulates halting as a differentiable probabilistic model (unbiased gradients) | principled halting for a learned orderer |
| **Deep Equilibrium Models (DEQ)** (Bai, Kolter, Koltun 2019, arXiv:1909.01377) | output = fixed point of a weight-tied map; backprop via implicit function theorem at the equilibrium | the theory HRM borrowed (and TRM discarded) |
| **Recurrent-depth / "Huginn"** (Geiping et al. 2025, arXiv:2502.05171) | 3.5B LM with a core block unrolled arbitrarily many times at test time -> latent CoT, scalable test-time compute | latent reasoning at LM scale; "think-free" tie-in |
| **Coconut** (Hao et al. 2024, arXiv:2412.06769) | feed the LM's last hidden state back as the next "continuous thought" — reasoning in latent space; can encode a BFS frontier | the token-CoT vs latent-CoT contrast |
| **Quiet-STaR** (Zelikman et al. 2024, arXiv:2403.09629) | learn token-level internal rationales that improve next-token prediction | bridges to the survey's "think-free reasoning" theme |
| **Survey on Latent Reasoning** (Zhu et al. 2025, arXiv:2507.06203) | taxonomy: activation recurrence, hidden-state propagation, internalized CoT | the umbrella the section sits under |

**Token-CoT vs latent reasoning:** explicit CoT spends tokens and is interpretable but brittle and slow;
latent reasoning keeps the chain in continuous hidden state (more bandwidth, no decoding cost, *but*
opaque). HRM/TRM are the extreme "puzzle-solver" end of latent reasoning: **no language at all** — pure
iterative refinement of a structured answer tensor. This is precisely why they fit *ordering* (a structured
combinatorial answer) better than they fit *free-text* reasoning.

---

## 3. Mapping our three tasks onto the HRM/TRM paradigm

### 3.1 Workflow ordering = solving a small permutation/CSP puzzle  (the real fit)

This is the key insight. A workflow of `N` steps with a pairwise precedence signal is **structurally a
Sudoku/ARC-like constraint-satisfaction puzzle**:

- **The "grid" / puzzle** = the `N x N` pairwise relation tensor `R` (calibrated `P(A->B)`, `P(B->A)`,
  `P(neutral)`) plus the `N x d` step embeddings `E`.
- **The "answer"** = a global ordering, represented as a **soft precedence / permutation matrix**
  `Y in R^{N x N}` (or a Plackett-Luce score vector).
- **The "solve"** = iteratively refine `Y` so it is **maximally consistent** with the confident pairwise
  constraints while **resolving cycles** — exactly what a Kemeny / minimum-feedback-arc-set objective
  encodes, and exactly the kind of "make the grid globally consistent" task TRM excels at.

So a TRM-style recurrent reasoner can take `(E, R)` as the puzzle and **learn** the aggregation: refine a
latent scratchpad `z`, then the ordering `y`, for `n` inner steps x deep-supervision steps, halting via ACT
on easy (short) workflows. This is a **learned alternative/complement to MFAS/Kemeny** — and unlike those
hand-written solvers it can (a) use the **step content** (`E`), not just the scalar scores, to break ties
the pairwise model was unsure about, and (b) be trained end-to-end with a **Kendall-tau-aware listwise
loss**, subsuming the Tier-3 "listwise / global ordering" idea.

**Why latent recursion specifically helps the 0/30:** the failure is that `O(N^2)` independent pairwise
decisions compound and form cycles. A recurrent solver reasons about the order **jointly and iteratively**:
each refinement pass can *propagate* a confident edge (A->B, B->C => nudge A->C) and *down-weight* a
low-confidence contradicting edge — i.e. it performs soft transitive closure + cycle repair in latent space,
which is what MFAS does combinatorially.

### 3.2 Pairwise causal direction = tiny 2-token reasoning

We could feed the two step embeddings `(e_A, e_B)` (or the CE's joint representation) into a tiny recursive
head that iterates a few steps and emits `PRECEDES / FOLLOWS / NEUTRAL`. This is a **plausible but low-value**
use: the cross-encoder already gets ~0.96 here; a recursive head over *frozen* embeddings is closer to the
*bi-encoder* regime (independent embeddings) that scored 0.41. **Latent recursion does not add the
cross-attention that is the actual source of the CE's strength.** So for the pairwise primitive, keep the
cross-encoder; don't replace it with a recursive head over pooled embeddings.

### 3.3 Follow-up reranking = pointwise/listwise scoring

A recursive reasoner could re-score a small candidate set jointly (listwise), iterating over the
anchor + candidates. This overlaps heavily with listwise cross-encoders (Set-Encoder) the survey already
covers; the recursive angle is a *minor* variant and not a clear win. Reranking is **not** the place to bet
on HRM/TRM.

**Conclusion of the mapping:** the *only* task where the HRM/TRM paradigm has a genuine structural
advantage is **(3) workflow ordering**, framed as a learned permutation/CSP solver over the CE relation
tensor.

---

## 4. Honest pros and cons

### 4.1 Pros (strongest arguments *for*)

1. **Structural fit is real.** Ordering = permutation/CSP over a small `N x N` grid — the exact shape
   TRM/HRM solve. It's a *learned* drop-in for the aggregation stage that can use step **content**, not
   just scores.
2. **Trivially under budget & cheap to run.** 5-27M params on top of a frozen encoder — orders of magnitude
   under 1B; fits one A100 easily; no CPU-OOM risk; fast to prototype.
3. **It is the principled listwise/global upgrade.** It subsumes the survey's Tier-3 "listwise / global
   ordering" idea (kills per-pair error compounding) with a concrete, tiny, trainable architecture, and
   gives a learnable home for a **Kendall-tau / Plackett-Luce** objective.
4. **Composes with everything we already recommend.** Encoder front-end can be the *winning* DeBERTa CE
   (frozen); calibration, NEUTRAL, transitive-closure data all still apply and *feed* it.

### 4.2 Cons (strongest arguments *against*)

1. **The ARC-team verdict undercuts the hype.** Most of HRM's reported magic is **outer-loop refinement +
   augmentation + transductive memorization**, not the architecture — and the setup **does not transfer
   across tasks**. We need inductive generalization to unseen AEP/AJO workflows, the regime HRM is weakest in.
2. **Worst-possible data regime for the thing it learns.** HRM/TRM want ~1000 examples + 300 augmentations
   **per distribution**, with **dense gold answers** (full Sudoku solutions). We have **~30 gold full
   orderings**. We can synthesize orderings from within-doc positional labels (and augment by shuffling
   presentation), but the *exact* signal the orderer needs (correct global orders for diverse real
   workflows) is our scarcest label. High overfitting / memorization risk.
3. **Puzzle-vs-language mismatch.** HRM/TRM operate on **fixed token grids**, not text. We *must* bolt on an
   encoder front-end (BGE / cross-encoder -> embeddings/relation tensor). That means the recursive model
   never sees raw text and **cannot recover the cross-attention** that makes the CE robust — it can only
   reason over whatever the encoder already distilled into `R` and `E`.
4. **It must beat an already-solved problem.** On the real CE signal (~0.96), **plain MFAS/Kemeny already
   orders every evaluated workflow correctly** at ~1 day and zero training. A learned solver only earns its
   keep in the **decent-but-noisy/cyclic** middle regime (e.g. a weaker/faster pairwise model, or harder
   unseen workflows, or when partial-order/NEUTRAL structure makes combinatorial MFAS ambiguous).

---

## 5. Comparison vs. the survey's existing recommendation

| Stage | Survey recommendation | Where TRM-orderer fits | Win? |
|---|---|---|---|
| Pairwise scorer | DeBERTa-v3 CE (+ distilled rationale; or Qwen3-0.6B think-free) | **keep as frozen front-end** | TRM does **not** win here (loses cross-attention) |
| Calibration | temperature scaling | feeds the relation tensor `R` | complementary |
| **Aggregation** | **calibrated weighted MFAS / Kemeny** | **TRM-orderer = learned aggregator** | **only here**; wins *iff* signal is noisy/cyclic or content helps tie-break |
| Listwise/global (Tier 3) | ListMLE / pointer / Set-Encoder | **TRM-orderer is a concrete realization** | TRM is a clean, tiny implementation of this idea |
| Reasoning (Tier 2) | distilled rationale head / generative think-free | orthogonal (TRM has no language) | not competing |
| RL polish (Tier 3) | tau-reward GRPO | could reward the TRM-orderer's order too | complementary |

**Verdict by stage:** latent recursive reasoning **does not** displace the cross-encoder (its strength is
cross-attention over text, which a frozen-embedding recursor can't reproduce) and **does not** beat
MFAS/Kemeny when the signal is already ~0.96. It **wins specifically as a *learned aggregator / listwise
orderer*** in the **noisy-but-decent, cyclic, partial-order** regime, and as the **concrete tiny
architecture** for the survey's otherwise-abstract Tier-3 "global ordering" recommendation.

---

## 6. Concrete proposal + minimal architecture

**Name:** *TRM-Orderer* — a tiny recursive learned aggregator.

```
Step texts --> [FROZEN encoder front-end]
                 - per-step embeddings  E  in R^{N x d}      (BGE / MiniLM / CE pooled)
                 - pairwise relation    R  in R^{N x N x 3}  (calibrated CE: P(->),P(<-),P(0))
                              |
                              v
        +------------ TRM-Orderer (~5-7M params, single 2-layer net) -----------+
        |  x = emb_proj(E) + rel_proj(R)          # "question" tokens, [N,H]      |
        |  y = x ; z = 0                          # answer-tokens, latent         |
        |  repeat (deep supervision, <=16):                                       |
        |     for T-1 grad-free, then 1 grad pass:                                |
        |        for n inner steps:  z <- net(x + y + z)                          |
        |        y <- net(y + z)                                                  |
        |     Y = y_head(y)            # precedence logits  [N,N]                  |
        |     h = halt(z)             # ACT stop prob       [1]                   |
        |     loss += KendallTau/PlackettLuce(Y, gold) + lambda*constraint(Y,R)   |
        |     detach(y,z)                                                         |
        +-------------------------------------------------------------------------+
                              |
                              v
            decode order = argsort(net_wins(Y))   # or Sinkhorn -> hard permutation
```

**Training (NOT run here):**
- **Deep supervision** with a **Kendall-tau-aware listwise loss** (ListMLE / Plackett-Luce on `Y`) +
  an auxiliary **constraint-consistency** term pulling `Y` toward confident `R` edges and *away* from
  low-confidence ones (this is the differentiable analogue of MFAS).
- **ACT halting** (PonderNet-style differentiable, or TRM's single binary loss) to spend fewer steps on
  short workflows.
- **Augmentation** (the ARC-team lesson): generate many orderings per workflow by **shuffling presentation
  order** (kills position leakage) and by **transitive-closure subsampling** of within-doc positional
  labels; aim for the regime where augmentation, not architecture, does the heavy lifting — but **never** a
  per-workflow id embedding (avoid HRM's transductive trap; stay inductive over `E`).
- **EMA** weights (0.999) for stability on scarce orderings.
- Encoder **frozen**; only the tiny recursor trains -> cheap, single-GPU, no OOM risk.

A runnable-shape PyTorch sketch is in `trm_orderer_sketch.py` (forward pass on random tensors; ~0.64M params
at the toy sizes used; numpy `--dry-run` fallback). **No training is launched.**

**Evaluation:** tie-aware Kendall-tau (tau_b/tau_c) + precedence-pair P/R (matching the survey's Tier-1
eval), and **head-to-head vs MFAS/Kemeny on the same `R`** — the only honest way to know if learning the
aggregation beats solving it.

**Risk register:**
- *High:* too few gold orderings -> overfit/memorize (mitigate: augmentation, EMA, freeze encoder, evaluate
  on held-out workflows / cross-doc).
- *Medium:* may simply tie MFAS/Kemeny when the signal is clean (then it adds complexity for nothing —
  decide on the noisy/partial-order regime).
- *Low:* compute / size (tiny).

---

## 7. Final verdict

- **Is it worth pursuing?** Yes, but as a **Track-B research bet**, after Tier-1 (MFAS/Kemeny) is in.
- **As what?** A **learned rank-aggregation / listwise ordering head** (*TRM-Orderer*) on top of the
  **frozen cross-encoder** — i.e. the concrete tiny implementation of the survey's Tier-3 "global ordering"
  idea, *competing with MFAS/Kemeny at the aggregation stage*, **not** replacing the cross-encoder and
  **not** used for the pairwise primitive or reranking.
- **At what risk?** Moderate, dominated by the **gold-ordering data scarcity** and the **ARC-team caveat**
  that the architecture may matter less than the refinement loop + augmentation. It is cheap to prototype and
  trivially within the sub-1B budget, so the downside is bounded.
- **Where it does NOT win:** the pairwise scorer (cross-attention beats frozen-embedding recursion), follow-up
  reranking (Set-Encoder territory), and the *already-solved* clean-signal aggregation case.

---

## References (real papers / arXiv IDs)

- Wang et al. **Hierarchical Reasoning Model.** arXiv:2506.21734 (2025). Sapient Intelligence.
- ARC Prize team. **The Hidden Drivers of HRM's Performance on ARC-AGI.** arcprize.org/blog/hrm-analysis (2025).
- Jolicoeur-Martineau. **Less is More: Recursive Reasoning with Tiny Networks (TRM).** arXiv:2510.04871 (2025). Samsung SAIL.
- Graves. **Adaptive Computation Time for Recurrent Neural Networks.** arXiv:1603.08983 (2016).
- Dehghani et al. **Universal Transformers.** arXiv:1807.03819 (ICLR 2019).
- Banino, Balaguer, Blundell. **PonderNet: Learning to Ponder.** arXiv:2107.05407 (2021).
- Bai, Kolter, Koltun. **Deep Equilibrium Models.** arXiv:1909.01377 (NeurIPS 2019).
- Geiping et al. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn).** arXiv:2502.05171 (2025).
- Hao et al. **Training Large Language Models to Reason in a Continuous Latent Space (Coconut).** arXiv:2412.06769 (2024).
- Zelikman et al. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking.** arXiv:2403.09629 (2024).
- Zhu et al. **A Survey on Latent Reasoning.** arXiv:2507.06203 (2025).
