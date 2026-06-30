# From Pairwise Causal Scores to a Global Workflow Order: Rank Aggregation, Cycle-Breaking, and Learning-to-Order

**Scope.** A model emits pairwise directional scores `P[i,j] = P(step_i causally precedes step_j)` for the steps of a workflow. We must aggregate these — noisy, possibly cyclic, possibly with genuinely-parallel (unordered) pairs — into a global order. The current pipeline (a tournament / topological sort in `order_events.py`) scores **0/30** on a 30-sample test (total failure), while a stronger pairwise model reaches **21/30**. This report surveys the optimal, robust ways to turn pairwise causal scores into a correct global order, with methods, tradeoffs, citations, and a recommended pipeline.

> **Citation discipline.** Every non-obvious claim carries an inline source URL. Numbers are reported only where verified against a primary source. Items that could not be verified verbatim are explicitly **[FLAGGED]**. No numbers are fabricated.

---

## Executive summary

1. **The current failure is structural, not a tuning problem.** A topological sort *requires a DAG*: on **any** directed cycle it cannot produce an order at all, and naively forcing acyclicity by dropping arbitrary edges yields wildly wrong orders. Noisy pairwise causal scores are almost always cyclic (A→B→C→A). A 0/30 vs 21/30 gap with a similar underlying pairwise signal is the classic signature of a brittle aggregation step, not a weak pairwise model. ([Feedback arc set — Wikipedia](https://en.wikipedia.org/wiki/Feedback_arc_set))

2. **The principled global objective is Minimum-Weight Feedback Arc Set (= Kemeny / minimum-violations ranking):** find the total order that disagrees with the *fewest (lowest-weight)* pairwise edges. It degrades gracefully — reproduces topological order exactly on a true DAG, and returns the best-supported order under noise/cycles. It is NP-hard in general but has a **PTAS on tournaments** (Kenyon-Mathieu & Schudy, STOC 2007) and excellent practical heuristics. ([Kemeny–Young — Wikipedia](https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method); [Kenyon-Mathieu & Schudy PDF](https://cs.brown.edu/~claire/Publis/kenyonschudy.pdf))

3. **If you want scores (not just an order), spectral / model-based aggregation is robust and cheap:** Rank Centrality (a random walk whose stationary distribution scores items) is *minimax-optimal* for top-K under standard noise (Negahban-Oh-Shah; Chen et al. 2019), and Bradley-Terry / Plackett-Luce MLE gives calibrated latent scores. Plain **win-counting (Borda/Copeland)** is provably near-optimal *with no parametric assumption* (Shah-Wainwright, JMLR 2017) — the most model-robust baseline. ([arXiv:1209.1688](https://arxiv.org/abs/1209.1688); [JMLR 18:199](https://www.jmlr.org/papers/v18/16-206.html))

4. **Cycles are signal, not just error.** HodgeRank decomposes the pairwise flow into a *gradient* (rankable) part plus *curl/harmonic* (cyclic, inconsistent) parts, giving both a global ranking **and** a certificate of how rankable the data is. ([arXiv:0811.1067](https://arxiv.org/abs/0811.1067))

5. **Parallel steps must be modeled as a partial order (DAG), not a chain.** Genuinely-unordered steps form an *antichain*; forcing them into a total order both misrepresents the workflow and creates spurious "errors." Evaluate with tie-aware metrics: Kendall's tau-b/tau-c, Goodman-Kruskal gamma, the Fagin et al. partial-ranking tau-distance, and precision/recall over precedence pairs. ([Kendall rank correlation — Wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient); [Fagin et al. 2006](https://epubs.siam.org/doi/abs/10.1137/05063088X))

6. **Per-pair accuracy and full-sequence accuracy are very different objectives.** A pipeline making O(n²) independent binary decisions compounds errors: one wrong high-leverage edge reorders a whole chain. End-to-end **listwise** objectives (ListNet, ListMLE = Plackett-Luce NLL) and **pointer/seq2seq** decoding condition each decision on global context and the partial order so far, which is why they beat pairwise-then-topo-sort in the sentence-ordering literature. ([Gong et al. 2016](https://arxiv.org/abs/1611.04953); [Xia et al., ICML 2008](https://icml.cc/Conferences/2008/papers/167.pdf))

7. **Calibrate the pairwise scores first.** Aggregators consume `P[i,j]` as a weight/probability, not just a sign. Overconfident scores distort weighted-FAS and Bradley-Terry objectives and break thresholding/abstention. Temperature scaling (one parameter, preserves arg-max) is the cheap, standard fix. ([Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599))

---

## 1. Rank aggregation from pairwise comparisons

**Setup.** `n` items (steps). Pairwise data is a matrix of scores/win-counts. Methods output either a global **score vector** `s ∈ ℝⁿ` (which induces an order) or a ranking directly. "BTL" = Bradley-Terry-Luce.

### 1.1 Bradley-Terry and Plackett-Luce (probabilistic, MLE; `choix`)

**Bradley-Terry (BT).** Each item has positive worth `pᵢ = exp(βᵢ)`; the model is `Pr(i beats j) = pᵢ/(pᵢ+pⱼ) = 1/(1+exp(−(βᵢ−βⱼ)))`. The log-odds of `i` beating `j` is exactly the score difference `βᵢ−βⱼ` — i.e. BT is logistic regression on latent-score differences, the same family as Elo. Original: Bradley & Terry, *Biometrika* 39(3/4):324-345, 1952 ([DOI](https://doi.org/10.1093/biomet/39.3-4.324) · [JSTOR](https://www.jstor.org/stable/2334029)); formulas at [Wikipedia](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model). The MLE is concave (after fixing scale), classically solved by the Zermelo/Ford minorization-maximization (MM) fixed point; the MLE is unique and finite **iff the comparison graph is strongly connected** (Ford's condition).

**Plackett-Luce (PL).** Generalizes BT from pairs to rankings via Luce's choice axiom: `P(ranking) = ∏ᵣ p_{σ(r)} / Σ_{k≥r} p_{σ(k)}` (a chain of "choose-best-from-remaining" softmaxes). BT is the 2-item case. Key inference result: the ML estimate of any Luce-axiom model is the **stationary distribution of a Markov chain**, which unifies spectral methods; **Luce Spectral Ranking (LSR)** approximates the MLE in one spectral step and **I-LSR converges to the MLE** (Maystre & Grossglauser, NeurIPS 2015, [proceedings](https://proceedings.neurips.cc/paper/2015/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html) · [PDF](http://papers.neurips.cc/paper/5681-fast-and-accurate-inference-of-plackettluce-models.pdf)). **[FLAGGED]** the paper's specific runtime/accuracy benchmark numbers were not verified from the abstract.

**`choix` library** ([docs](http://choix.lum.li/en/latest/) · [API](https://choix.lum.li/en/latest/api.html) · [GitHub](https://github.com/lucasmaystre/choix) · [PyPI](https://pypi.org/project/choix/)) implements "inference algorithms for models based on Luce's choice axiom"; the pairwise variant *is* Bradley-Terry, "closely related to the Elo rating system." Verified pairwise functions:
- `lsr_pairwise` / `lsr_pairwise_dense` — one-shot spectral approximation to the MLE.
- `ilsr_pairwise` / `ilsr_pairwise_dense` — **ML estimate via I-LSR** (iterative, converges to MLE).
- `mm_pairwise` — ML estimate via minorization-maximization.
- `opt_pairwise` — ML/MAP via `scipy.optimize`.
- `rank_centrality` — Negahban-Oh-Shah Rank Centrality (§1.3).
- `ep_pairwise` — approximate Bayesian inference (expectation propagation).
- `log_likelihood_pairwise` — likelihood evaluation.

**Complexity.** MM and I-LSR iterate; each iteration is `O(#comparisons)` (sparse). `opt_pairwise` is heavier per step.

**Cycles / robustness.** BT/PL are *parametric* and assume transitive latent scores. Noisy/inconsistent observations are absorbed into the best transitive fit — a Condorcet cycle in the raw data is simply fit to the closest transitive score vector. The flip side: the model is **misspecified** if true preferences are genuinely intransitive, and cannot represent a real cycle; estimates can then be biased. Requires a strongly connected comparison graph.

### 1.2 Elo rating

Each player has rating `Rᵢ`; expected score `Eᵢ = 1/(1+10^{(Rⱼ−Rᵢ)/400})`; update `Rᵢ ← Rᵢ + K(S−Eᵢ)`, `S∈{1,½,0}` (Arpad Elo, *The Rating of Chessplayers*, 1978; Elo↔BT connection at [Wikipedia](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)). Elo's win-probability is exactly the BT logistic with base 10, scale 400 — Elo is an **online stochastic-approximation (SGD-style)** estimator of the same latent-score model that BT estimates in batch. **Complexity:** `O(1)` per game, no matrix ops. **Cycles/robustness:** order-dependent and drifting; it tracks a moving estimate rather than resolving cycles; large `K` adapts fast but with more variance, small `K` is stable but slow. No batch-optimality guarantee. (FIDE K-factor values are administrative policy, not theory — **[FLAGGED]** as non-theoretical.)

### 1.3 Spectral ranking / Rank Centrality (Negahban, Oh, Shah)

Build a random walk on the comparison graph where the transition `i→j` is proportional to the fraction of times `j` beat `i` (normalized by max degree). The **Rank Centrality score is the stationary probability** `π = πP`, computed by power iteration. Items that lose often pass probability mass to their winners, so winners accumulate mass. Primary: Negahban, Oh, Shah, "Rank Centrality: Ranking from Pairwise Comparisons," [arXiv:1209.1688](https://arxiv.org/abs/1209.1688), *Operations Research* 65(1):266-287, 2017 ([INFORMS](https://pubsonline.informs.org/doi/10.1287/opre.2016.1534) · [MIT DSpace](https://dspace.mit.edu/handle/1721.1/111030)).

**Complexity.** `O(nnz(P))` per power iteration (`nnz` = compared pairs); far cheaper than full nonlinear MLE in practice.

**Guarantees.** Under BTL, when the comparison-graph Laplacian has a strictly positive spectral gap (e.g. each item compared to randomly chosen others), the sample dependence is **nearly order-optimal**, and empirically it "performs as well as the MLE for the BTL model." Sharper results (Chen, Fan, Ma, Wang, "Spectral Method and Regularized MLE Are Both Optimal for Top-K Ranking," *Annals of Statistics* 2019, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6785035/)): **top-K exact recovery** w.p. `≥ 1−O(n⁻⁵)` when `n²pL² ≳ n·log n / Δ_K²` (`p` = comparison probability, `L` = comparisons per pair, `Δ_K` = score gap between K-th and (K+1)-th items, bounded dynamic range), **matching the minimax lower bound** — so the spectral method is minimax-optimal for top-K. **[FLAGGED]:** the exact exponents (presence of square roots) on the `ℓ∞`/`ℓ2` error bounds were not cleanly extractable; the minimax-optimality and the top-K threshold were returned verbatim and are reliable.

**Cycles/robustness.** The stationary distribution is well-defined whenever the chain is irreducible (strongly connected graph); cycles do not break the method — they are absorbed into the walk. Robustness ⇔ spectral gap: a well-connected comparison design gives near-optimal sample complexity.

### 1.4 Least-squares / SVD ranking and HodgeRank

Primary: Jiang, Lim, Yao, Ye, "Statistical Ranking and Combinatorial Hodge Theory," [arXiv:0811.1067](https://arxiv.org/abs/0811.1067), *Math. Programming* 127(1):203-244, 2011 ([Springer](https://link.springer.com/article/10.1007/s10107-010-0419-x) · [author PDF](https://web.stanford.edu/~yyye/hodgeRank2011.pdf)).

Treat comparisons as a skew-symmetric **edge flow** `Y` (`Y_{ij}` = advantage of `i` over `j`). A global ranking exists iff `Y` is a **gradient**: `Y_{ij} = sᵢ − sⱼ`. The least-squares fit `min_s Σ w_{ij}(sᵢ−sⱼ−Y_{ij})²` is solved by a single **graph-Laplacian linear system** `Ls = div Y` — this is the "least-squares / SVD ranking." The **Hodge / Helmholtz decomposition** splits the flow space orthogonally into three parts:
1. **Gradient** — the part explained by a global score `s` (the rankable/consistent signal; the L2-optimal ranking is exactly this potential).
2. **Curl** — *locally* cyclic flow (inconsistency around triangles, e.g. `a>b>c>a`).
3. **Harmonic** — *locally acyclic but globally cyclic* flow (global inconsistency tied to graph topology).

The magnitudes of curl + harmonic quantify *how unrankable* the data is: large gradient fraction ⇒ a meaningful global ranking exists; large cyclic energy ⇒ inherently inconsistent. **Complexity:** one sparse weighted-Laplacian solve (polynomial), far cheaper than the NP-hard Kemeny optimization it contrasts against. **Cycles/robustness:** HodgeRank's signature strength — it does not merely tolerate cycles, it **measures and separates** them, returning a global ranking *plus* a certified inconsistency residual that tells you when to distrust the ranking.

### 1.5 Copeland's method and Borda count

Classical social-choice rules on the pairwise matrix; no probabilistic model.
- **Copeland:** decide a majority winner per pair; score = (#wins) − (#losses), ties counted as ½; rank by score. Satisfies the **Condorcet criterion** (a Condorcet winner is ranked first). ([Copeland's method](https://en.wikipedia.org/wiki/Copeland's_method); [Condorcet method](https://en.wikipedia.org/wiki/Condorcet_method)).
- **Borda:** sum points by rank position; generalizes to >2 items; avoids hard ties but can violate Condorcet.
- Both are `O(n²)` given the matrix. Copeland **does not resolve Condorcet cycles** — it can leave persistent ties that do not vanish as sample size grows (voting paradox).

**Why win-counting is the most model-robust baseline.** Shah & Wainwright, "Simple, Robust and Optimal Ranking from Pairwise Comparisons," [arXiv:1512.08949](https://arxiv.org/abs/1512.08949), JMLR 18(199):1-38, 2017 ([JMLR](https://www.jmlr.org/papers/v18/16-206.html)), analyze the **counting algorithm** (rank by number of pairwise wins — Borda/Copeland-style) and prove it is (verbatim): **(1) optimal** — "achieves the information-theoretic limits" for top-k and full ranking; **(2) robust** — guarantees "impose no conditions on the underlying matrix of pairwise-comparison probabilities," unlike BTL-only results, i.e. it works under general (non-parametric, possibly non-transitive) noise; **(3) efficient** — "speed-ups of several orders of magnitude." This is the headline reason to keep win-counting as a baseline when you doubt the BTL model.

---

## 2. Minimum Feedback Arc Set (MFAS) / Kemeny ranking

### 2.1 MFAS: ordering as breaking cycles
Given a digraph `G=(V,A)`, a **feedback arc set** is a set of arcs intersecting every directed cycle; removing it yields a DAG. The **minimum (weight) FAS** is equivalently the **linear vertex ordering minimizing the number (weight) of "backward" arcs** — arcs pointing from a later to an earlier vertex. This ordering↔cycle-breaking duality is exactly "rank the steps so you disagree with as few pairwise edges as possible." A **tournament** is a complete oriented graph (exactly one of `u→v`, `v→u` per pair); ranking lives on **(weighted) FAS on tournaments (FAST)**. ([Feedback arc set — Wikipedia](https://en.wikipedia.org/wiki/Feedback_arc_set))

**Complexity.** General FAS decision is one of Karp's original 21 NP-complete problems (Karp 1972, [PDF](https://www.cs.umd.edu/~gasarch/BLOGPAPERS/Karp.pdf); [list](https://en.wikipedia.org/wiki/Karp's_21_NP-complete_problems)). FAS stays **NP-hard even on tournaments** (Ailon-Charikar-Newman 2005 under randomized reductions; derandomized by Alon, "Ranking Tournaments," *SIAM J. Discrete Math.* 20(1):137-142, [PDF](https://www.cs.tau.ac.il/~nogaa/PDFS/paley.pdf); simpler reduction by Charbit-Thomassé-Yeo 2007, [PDF](https://perso.ens-lyon.fr/stephan.thomasse/liste/feedback.pdf); independent proof via Slater by Conitzer, AAAI-06, [PDF](https://www.cs.cmu.edu/~conitzer/slaterAAAI06.pdf)).

**Approximation.**
- General digraphs: best known `O(log n · log log n)` (attributed to Even-Naor-Schieber-Sudan 1998; ratio per [Wikipedia](https://en.wikipedia.org/wiki/Feedback_arc_set)) — **[FLAGGED]** attribution confirmed on Wikipedia, original paper not opened. A constant-factor approximation for general FAS is open.
- **Tournaments: a PTAS.** Kenyon-Mathieu & Schudy, "How to Rank with Few Errors," STOC 2007, pp. 95-103 ([ACM](https://dl.acm.org/doi/pdf/10.1145/1250790.1250806) · [author PDF](https://cs.brown.edu/~claire/Publis/kenyonschudy.pdf) · [journal version](https://cs.brown.edu/people/wschudy/papers/fast_journal.pdf)) give a `(1+ε)`-approximation in polynomial time for **weighted** FAST, which directly yields a **PTAS for Kemeny rank aggregation**. **[FLAGGED]:** the exact `n^{O(1/ε…)}` running-time expression was not extractable; the `(1+ε)`-PTAS claim and the Kemeny↔weighted-FAST reduction are corroborated by multiple sources. A simple combinatorial **5-approximation** ("order by number of wins") and LP-rounding 2.5-/2.127-approximations also exist ([improved LP bound](https://www.sciencedirect.com/science/article/abs/pii/S0304397524003852)).

### 2.2 Kemeny ranking
The **Kemeny consensus** minimizes the sum of **Kendall-tau distances** to the inputs — equivalently minimizes total pairwise disagreement ([Kemeny–Young — Wikipedia](https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method)). Build the **majority (weighted) tournament** (arc in the majority direction, weight = margin/fraction); the Kemeny optimum is exactly the **minimum-weight FAS / minimum-violations ranking** of that tournament — the reduction that makes the KMS PTAS apply. **NP-hardness:** Bartholdi-Tovey-Trick, *Soc. Choice Welf.* 6(2):157-165, 1989 ([Springer](https://link.springer.com/article/10.1007/BF00303169)); strengthened by Dwork, Kumar, Naor, Sivakumar, "Rank Aggregation Methods for the Web," WWW 2001 ([PDF](https://www.stat.uchicago.edu/~lekheng/meetings/mathofranking/ref/kumar.pdf) · [Weizmann](https://www.wisdom.weizmann.ac.il/~naor/PAPERS/rank_www10.html)) — **NP-hard even with only 4 input rankings**; survey: Hemaspaandra-Spakowski-Vogel, *TCS* 349(3):382-391, 2005 ([ACM](https://dl.acm.org/doi/10.1016/j.tcs.2005.08.031)). Statistically, weighted MFAS = Kemeny is the maximum-likelihood ranking under Mallows/Condorcet-style noise models, a justification topological sort lacks.

### 2.3 Practical algorithms (and what NetworkX gives you)
- **Eades-Lin-Smyth greedy (GR) heuristic.** Eades, Lin, Smyth, "A fast and effective heuristic for the feedback arc set problem," *Information Processing Letters* 47(6):319-323, 1993 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/002001909390079O)). Repeatedly peel off sinks (to the end) and sources (to the front); for the rest pick the vertex maximizing `outdeg − indeg`. Linear time `O(m)`; guarantee: acyclic subgraph of `≥ m/2 + n/6` arcs (FAS `≤ m/2 − n/6`).
- **Local search:** from a starting permutation (GR output or sort-by-wins) apply swaps/insertions/pivoting that reduce backward-arc weight. The practical complement to the (too-slow) PTAS.
- **ILP / exact:** binary order vars `x_{ij}` with triangle constraints `x_{ij}+x_{jk}+x_{ki} ≤ 2`, minimize backward-arc weight; branch-and-cut ([exact method](https://dl.acm.org/doi/fullHtml/10.1145/3446429)). Handles a few dozen items in favorable cases.

**NetworkX has NO `feedback_arc_set`.** (The library that does is **igraph**: `igraph.feedback_arc_set(..., algo="approx_eades"|"exact_ip")`, [docs](https://igraph.org/r/doc/feedback_arc_set.html).) NetworkX supplies only building blocks ([dag](https://networkx.org/documentation/stable/reference/algorithms/dag.html), [cycles](https://networkx.org/documentation/stable/reference/algorithms/cycles.html), [tournament](https://networkx.org/documentation/stable/reference/algorithms/tournament.html)):
- DAG: `topological_sort`, `topological_generations` (← stratifies a DAG into generations = **antichains / parallel-step layers**), `all_topological_sorts`, `lexicographical_topological_sort`, `is_directed_acyclic_graph`, `transitive_closure`, `transitive_reduction`, `dag_longest_path`.
- Cycles: `simple_cycles`, `find_cycle`, `cycle_basis`, `minimum_cycle_basis`.
- Tournament: `is_tournament`, `hamiltonian_path` (gives *a* ranking, **not** an MFAS-optimal one), `is_strongly_connected`, `tournament_matrix`.

So in Python you implement MFAS/Kemeny yourself (GR → local search, or ILP via PuLP/Gurobi) or call igraph; NetworkX handles topo-sort *after* you remove the FAS, plus cycle detection.

### 2.4 When MFAS beats naive topological sort
- **Topological sort requires a DAG**; on any cycle it fails outright. Real causal pairwise data is almost always cyclic, so `topological_sort` is simply inapplicable — this is the most likely cause of the 0/30 result.
- **MFAS degrades gracefully:** always returns a total order; on a true DAG it reproduces the topological order exactly (zero backward arcs); on noisy/cyclic input it returns the order disagreeing with the fewest/least-weight edges.
- **Robustness:** a single spurious back-edge breaks topo-sort (or, with arbitrary edge deletion, scrambles the order); MFAS isolates the minimum offending set so the result reflects the dominant signal. With edge weights = confidences, weighted MFAS = Kemeny = ML ranking.

---

## 3. Handling cycles and inconsistency in predicted pairwise graphs

A pairwise model making O(n²) independent decisions routinely produces **A→B→C→A**. Options, roughly in increasing sophistication:

1. **Don't threshold to a hard graph at all — keep weights.** Convert `P[i,j]` into a weighted tournament and solve **weighted MFAS / Kemeny** (§2). Cycles never need to be "fixed"; they are paid for in the objective. This is the single most important change vs. the current pipeline.
2. **Confidence-weighted edges.** Weight each arc by margin `|P[i,j] − 0.5|` or log-odds `log(P[i,j]/(1−P[i,j]))` (the natural BT/weighted-FAS weight). A low-confidence wrong edge then costs little, so it cannot flip the global order; a high-confidence edge is respected. Requires **calibrated** `P` (§7) for the weights to mean what they say.
3. **Spectral / model-based absorption.** Rank Centrality (§1.3), BT/PL MLE (§1.1), and HodgeRank's gradient (§1.4) all absorb cycles into a global score without an explicit cycle-breaking step — provided the comparison graph is strongly connected. HodgeRank additionally **quantifies** the cyclic (curl + harmonic) energy so you know how trustworthy the order is ([arXiv:0811.1067](https://arxiv.org/abs/0811.1067)).
4. **Thresholding / abstention for low-confidence pairs.** Keep an oriented edge only when `P` (or `1−P`) exceeds a threshold τ; **abstain** when `P ≈ 0.5`, leaving the pair *unordered*. With calibrated probabilities, τ corresponds to a fixed reliability level (Guo et al. 2017, §7). Abstained pairs become the antichains of §4 — genuinely-parallel steps left unordered rather than forced.
5. **Cycle diagnosis.** Use `networkx.simple_cycles` / `find_cycle` to surface 3-cycles for inspection; many real cycles are a single weak edge whose removal (lowest-weight arc on the cycle) is exactly the greedy MFAS move.

---

## 4. Partial orders and parallel steps

**Model the workflow as a DAG / strict partial order, not a chain.** A DAG induces a partial order (`x<y` iff a directed path exists). A **linear extension** = a total order consistent with the partial order = a **topological ordering**; a workflow generally has *many* valid linear extensions. An **antichain** is a set of pairwise-incomparable elements — exactly the **steps that may run in parallel**. **Dilworth's theorem:** the largest antichain (max parallelism) equals the minimum number of chains the poset decomposes into (Dilworth, *Annals of Math.* 51(1):161-166, 1950, [JSTOR](https://www.jstor.org/stable/1969503); [antichain](https://en.wikipedia.org/wiki/Antichain), [Dilworth](https://en.wikipedia.org/wiki/Dilworth%27s_theorem), [linear extension](https://en.wikipedia.org/wiki/Linear_extension)). `networkx.topological_generations` directly yields the layered antichain structure.

**Why this matters for the 0/30 vs 21/30 evaluation:** scoring against a single gold *chain* penalizes a prediction that merely chose a different valid linear extension of parallel steps. Evaluate against the partial order.

**Metrics with ties** (formulas verified at [Kendall rank correlation — Wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient); primary Kendall, *Biometrika* 30:81-93, 1938, [JSTOR](https://www.jstor.org/stable/2332226)). Over all `n₀ = n(n−1)/2` pairs with `n_c` concordant, `n_d` discordant:
- **tau-a** (no tie correction): `τ_a = (n_c − n_d)/n₀`.
- **tau-b** (tie-corrected denominator): `τ_b = (n_c − n_d)/√[(n₀−n₁)(n₀−n₂)]`, `n₁,n₂` = tie corrections in each variable; reaches ±1 only for square tables.
- **tau-c** (Stuart-Kendall, rectangular): `τ_c = 2(n_c − n_d)/[n²·(m−1)/m]`, `m = min(r,c)`.
- **Goodman-Kruskal gamma** (ties excluded entirely): `γ = (n_c − n_d)/(n_c + n_d)` (Goodman & Kruskal, *JASA* 49:732-764, 1954, [JSTOR](https://www.jstor.org/stable/2281536); [Wikipedia](https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_gamma)).

**Partial-ranking (bucket-order) distance.** Fagin, Kumar, Mahdian, Sivakumar, Vee, "Comparing Partial Rankings," *SIAM J. Discrete Math.* 20(3):628-648, 2006 ([SIAM](https://epubs.siam.org/doi/abs/10.1137/05063088X), DOI 10.1137/05063088X) define four metrics by generalizing Kendall-tau and Spearman-footrule to partial rankings (bucket orders = ties allowed) and prove them **equivalent up to constant factors**. The companion "Comparing Top k Lists," *SIAM J. Discrete Math.* 17(1):134-160, 2003 ([SIAM](https://epubs.siam.org/doi/10.1137/S0895480102412856)) introduces the **Kendall tau distance with penalty `p ∈ [0,1]` for tied pairs**: per pair, penalty `0` if both agree on order, `1` if they strictly disagree, `p` if tied in one ranking but ordered in the other (`p=½` is the neutral choice). **[FLAGGED]:** the `{0,1,p}` scheme is reproduced from the companion paper + secondary literature; the 2006 PDF body was not machine-extractable — confirm verbatim before quoting lemma numbers.

**Precision/recall over precedence pairs.** Represent a partial order as its set of precedence constraints `{(i,j): i must precede j}` and score predicted vs gold as pairwise classification: **precision** = correct predicted precedences / predicted precedences; **recall** = recovered gold precedences / gold precedences; **F1**. This credits any correct precedence regardless of which linear extension was chosen and does not penalize ordering of genuinely-parallel pairs. **[FLAGGED]:** standard practice (and equal to Kendall tau-distance counting) rather than one canonical citation.

---

## 5. Error compounding: why per-pair accuracy can be high while full-sequence accuracy collapses

A pairwise-then-assemble pipeline makes **O(n²) independent binary decisions** that must be reconciled into one permutation. Three compounding effects:

- **Leverage.** A single wrong but high-confidence edge can reorder an entire chain; full-sequence (exact-match / PMR) success requires *all* the load-bearing pairwise decisions to be right simultaneously. If per-pair accuracy is `a`, exact-sequence success is closer to `a^(#critical pairs)` than to `a` — which is precisely how 21/30 per-pair-ish strength can collapse to 0/30 exact-order.
- **Local context.** A pairwise classifier sees only two steps; it lacks the global workflow context that disambiguates ties and resolves cycles.
- **Exposure bias / inconsistency.** Independent decisions can be mutually inconsistent (cycles), and a brittle assembler (topo-sort) then fails outright. This is the sequence-prediction "exposure bias / error propagation" phenomenon: a model never trained on its own partial outputs compounds early mistakes ([scheduled sampling overview](https://www.researchgate.net/publication/278048447_Scheduled_Sampling_for_Sequence_Prediction_with_Recurrent_Neural_Networks); [exposure bias survey](https://www.emergentmind.com/topics/exposure-bias)).

**Mitigations:** (a) a **global objective** (weighted MFAS/Kemeny) that optimizes the whole order jointly rather than committing greedily; (b) **beam search** over the assembly instead of a single greedy pass; (c) **listwise / pointer decoding** (§6) that conditions each step on the partial order built so far. The sentence-ordering papers below were explicitly motivated by replacing pairwise pipelines to "alleviate the error propagation problem" (Gong et al., [arXiv:1611.04953](https://arxiv.org/abs/1611.04953)).

---

## 6. Learning-to-order end-to-end (sentence-ordering literature)

The NLP **sentence-ordering** task (predict the correct order of shuffled sentences) is the closest analog to step ordering and has directly studied "pairwise→topological sort" vs end-to-end approaches. Metrics: **Kendall's tau**, **Perfect Match Ratio (PMR)** (fraction with the *entire* order exactly correct — the strict analog of your 0/30 metric), and first/last-sentence accuracy. Datasets: NIPS/AAN/NSF/arXiv abstracts, SIND (visual storytelling), ROCStories.

**Pairwise-then-topological-sort (your current design).** Prabhumoye, Salakhutdinov, Black, "Topological Sort for Sentence Ordering," ACL 2020 ([ACL](https://aclanthology.org/2020.acl-main.248/) · [arXiv](https://arxiv.org/pdf/2005.00432) · [code](https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering)) reframe ordering as constraint solving: predict pairwise relative-order constraints with a BERT classifier, then **topological sort** the constraint graph (variants TSort / B-TSort). This is essentially your pipeline done well — note their BERT pairwise classifier is strong and they argue topo-sort over *robust* constraints is competitive and interpretable. The earlier Chen, Qiu, Huang, "Neural Sentence Ordering" ([arXiv:1607.06952](https://arxiv.org/abs/1607.06952)) uses a pairwise LSTM + **beam search** to assemble the order.

**End-to-end pointer / set-to-sequence (the error-compounding fix).** Gong et al., "End-to-End Neural Sentence Ordering Using Pointer Network" ([arXiv:1611.04953](https://arxiv.org/abs/1611.04953)) and Logeswaran, Lee, Radev (AAAI 2018, [arXiv:1611.02654](https://arxiv.org/abs/1611.02654)) encode all sentences jointly and **decode the permutation with a pointer network**, explicitly to "alleviate error propagation" and "utilize the whole contextual information." BERSON (Cui, Li, Zhang, EMNLP 2020, [ACL](https://aclanthology.org/2020.emnlp-main.511/)) adds a BERT-based Relational Pointer Decoder. As re-tabulated by Re-BART (EMNLP 2021, [arXiv:2104.07064](https://arxiv.org/abs/2104.07064)), BERSON's pointer decoder reports **higher τ and PMR than B-TSort** on shared datasets (e.g. NIPS PMR 48.01 vs 32.59, τ 0.85 vs 0.81; NSF PMR 23.07 vs 10.44) — consistent with the error-compounding thesis, though both use BERT so it is not a perfectly controlled ablation. **[FLAGGED]:** these values come from a re-tabulating third paper, not the originals; the exact 2005.00432 table was not cleanly extractable.

**Listwise losses (and their PL connection).**
- **ListNet** — Cao, Qin, Liu, Tsai, Li, "Learning to Rank: From Pairwise Approach to Listwise Approach," ICML 2007 ([MSR TR PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)): defines a distribution over permutations via the **Plackett-Luce / Luce** model and minimizes **cross-entropy** between predicted and gold (typically top-one) probabilities.
- **ListMLE** — Xia, Liu, Wang, Zhang, Li, "Listwise Approach to Learning to Rank: Theory and Algorithm," ICML 2008 ([PDF](https://icml.cc/Conferences/2008/papers/167.pdf)): the **negative log-likelihood of the gold permutation under Plackett-Luce** — i.e. **ListMLE = Plackett-Luce NLL** (verified). They prove the loss is continuous, differentiable, convex, and give a consistency condition.
- The autoregressive **pointer decoder** implements the *same* PL factorization operationally: each step is a softmax (pointer attention) choosing the next item from those remaining — end-to-end pointer decoding and ListMLE are the same generative story.

**Controlled loss comparison.** RankTxNet (Kumar et al., AAAI 2020, [arXiv:2001.00056](https://arxiv.org/abs/2001.00056)) scores each sentence and sorts, supporting pointwise/pairwise/**listwise (ListMLE)** losses in one architecture. **Verified** (their Table 2): listwise/pairwise beat pointwise, and ListMLE is best in most datasets — e.g. arXiv pairwise→ListMLE τ 0.7516→0.7666, PMR 41.28→43.44; AAN τ 0.7704→0.7748; NSF τ 0.5614→0.5798. This isolates the *objective* (global > local) from architecture.

**Takeaway for step ordering:** if you can train end-to-end, a pointer/seq2seq decoder or a listwise (Plackett-Luce / ListMLE) objective directly optimizes the global permutation and avoids the O(n²)-independent-decision compounding. If you must keep a pairwise reranker, replace topo-sort with weighted MFAS/Kemeny (§2) and/or beam search.

---

## 7. Calibration of pairwise scores

Aggregators consume `P[i,j]` as a **weight/probability**, not just a sign, so miscalibration directly distorts the global order.

- **Why it matters.** Bradley-Terry sets `logit Pr(i≻j) = βᵢ − βⱼ`; fitting latent strengths by MLE treats each `P` as a probability, so systematically over/under-confident inputs bias the recovered order ([BT model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)). Weighted MFAS/Kemeny weights each arc by its score, so an overconfident wrong edge can dominate the objective and flip the aggregate ([weighted FAS for ranking](https://arxiv.org/abs/2412.16181)). And only calibrated `P` makes a confidence threshold τ correspond to a fixed reliability level for thresholding/abstention (§3-4). **[FLAGGED]:** the end-to-end "calibrate before aggregating precedence" pipeline is a synthesis of these independently-sourced primitives, not a single primary result.

- **Methods.**
  - **Platt scaling** (Platt 1999, [PDF](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods)): fit a 2-parameter sigmoid `P(y=1|f) = 1/(1+exp(Af+B))` on held-out data by NLL. Numerical refinement: Lin, Lin, Weng, *Machine Learning* 68(3), 2007 ([DOI](https://dl.acm.org/doi/abs/10.1007/s10994-007-5018-6)).
  - **Isotonic regression** (Zadrozny & Elkan, KDD 2002, [ACM](https://dl.acm.org/doi/10.1145/775047.775151)): non-parametric monotone fit via pair-adjacent-violators; more flexible than Platt, more data-hungry / overfit-prone on small calibration sets.
  - **Temperature scaling** (Guo, Pleiss, Sun, Weinberger, ICML 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) · [PMLR](https://proceedings.mlr.press/v70/guo17a.html)): single scalar `T>0`, calibrated prob = `softmax(z/T)`, `T` chosen to minimize validation NLL. "A single-parameter variant of Platt scaling," "surprisingly effective"; modern nets are typically **overconfident**. Because it scales all logits equally, it **does not change the arg-max** — accuracy is unchanged, only confidences are fixed (co-author reference [blog](https://geoffpleiss.com/blog/nn_calibration.html)). This is the recommended cheap default.

- **Measuring calibration.** Perfectly calibrated: `P(Ŷ=Y | P̂=p) = p` ∀p. **Reliability diagram** plots bin accuracy vs confidence (diagonal = perfect; below = overconfident). **Expected Calibration Error**: `ECE = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)|` over `M` confidence bins; **MCE** = max bin gap (Guo et al. 2017; binned estimator from Naeini, Cooper, Hauskrecht, AAAI 2015, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9602)). `sklearn` provides `CalibratedClassifierCV` (Platt/isotonic) and `calibration_curve`.

---

## Comparison table

| Method | Output | Handles cycles? | Handles partial order? | Complexity | When to use |
|---|---|---|---|---|---|
| **Topological sort** (current) | total order | **No** — fails on any cycle | Yields *a* linear extension; doesn't model antichains | `O(V+E)` | Only when input graph is already a clean DAG |
| **Bradley-Terry / Plackett-Luce MLE** (`choix`) | latent scores | Absorbs (best transitive fit); cannot represent true cycle | Via score ties / abstention | iterative, `O(edges)`/iter | Want calibrated scores; trust BTL model; strongly-connected graph |
| **Elo** | online scores | Tracks, doesn't resolve; order-dependent | No (chain) | `O(1)`/comparison | Streaming/online updates; quick baseline |
| **Rank Centrality** (spectral) | stationary scores | Absorbs (well-defined on strongly-connected graph) | Via score ties | `O(edges)`/power iter | Want minimax-optimal scores cheaply under BTL; good graph connectivity |
| **HodgeRank** (LSQ/SVD) | scores + inconsistency residual | **Explicitly decomposes** gradient + curl + harmonic | Gradient gives scores; residual flags unrankable pairs | one sparse Laplacian solve | Want a ranking *plus* a certificate of how rankable the data is |
| **Copeland / Borda (win-counting)** | win-count scores | Copeland can leave persistent ties | Via tie scores | `O(n²)` | Most model-robust baseline; no parametric assumption (Shah-Wainwright) |
| **Weighted MFAS / Kemeny** | total order (min violations) | **Yes** — minimizes backward-arc weight | Bucket/partial-order variants exist | NP-hard; **PTAS on tournaments**; GR `O(m)` heuristic | The principled global order from noisy/cyclic weighted pairwise data |
| **Listwise (ListNet/ListMLE) / pointer decoder** | permutation (or scores→sort) | Joint objective avoids inconsistency | Can predict ties / abstain | training-dependent; decode `O(n²)` | When you can train end-to-end; best full-sequence accuracy |

---

## Recommended ordering pipeline for noisy causal pairwise scores (sub-1B reranker → global order)

A concrete, implementation-oriented pipeline that replaces the brittle topo-sort while reusing the existing pairwise reranker. Steps 1-4 are a drop-in fix; step 6 is the larger end-to-end upgrade.

**0. Keep the strong pairwise reranker.** The 21/30 model shows the pairwise signal is usable — the failure is in aggregation, not (primarily) the pairwise model.

**1. Calibrate the pairwise scores.** Fit **temperature scaling** (one scalar `T`, [Guo et al. 2017](https://arxiv.org/abs/1706.04599)) on a held-out split so `P[i,j]` is a trustworthy probability; verify with ECE / a reliability diagram (`sklearn.calibration`). This is cheap and does not change which step the model thinks comes first.

**2. Build a weighted tournament — do NOT threshold to a hard DAG.** For each ordered pair set arc weight to the calibrated log-odds `w_{ij} = log(P[i,j]/(1−P[i,j]))` (or margin `P[i,j]−0.5`). Keep both directions' net weight; this preserves confidence and is exactly the weight that makes weighted-FAS equal the ML (Kemeny) order.

**3. Aggregate with weighted Minimum-Violations / Kemeny ranking** ([Kemeny↔weighted-FAST](https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method); [KMS PTAS](https://cs.brown.edu/~claire/Publis/kenyonschudy.pdf)). Practical recipe: initialize by **sort-by-(weighted) wins** (Copeland/Borda — provably good and model-robust, [Shah-Wainwright JMLR 2017](https://www.jmlr.org/papers/v18/16-206.html)), then run **Eades-Lin-Smyth greedy** ([IPL 1993](https://www.sciencedirect.com/science/article/abs/pii/002001909390079O)) + **local search** (adjacent swaps / single-item reinsertion that reduce backward-arc weight). For ≤ ~30-40 steps, an **ILP** (`x_{ij}` order vars + triangle constraints, minimize backward weight, via PuLP/Gurobi) finds the exact optimum. Use **igraph's `feedback_arc_set`** if you prefer a library; NetworkX has no FAS function, only the building blocks.

**3b. (Optional, recommended) Run a spectral/score method in parallel** — `choix.rank_centrality` or `choix.ilsr_pairwise` ([choix](https://choix.lum.li/en/latest/api.html)) — to get continuous scores, sanity-check the MFAS order, and break ties. Run **HodgeRank** ([arXiv:0811.1067](https://arxiv.org/abs/0811.1067)) to obtain the curl+harmonic inconsistency energy: high cyclic energy means "this workflow has no clean global order" and you should expose parallelism rather than force a chain.

**4. Recover parallel structure (partial order).** Use **abstention**: for pairs with `P[i,j] ≈ 0.5` (within a calibrated band), leave the pair *unordered*. Compute `networkx.topological_generations` on the kept high-confidence edges to expose antichains (parallel-step layers). Output a **DAG / partial order**, not a forced chain.

**5. Evaluate as a partial order.** Replace exact-chain matching with: **precision/recall/F1 over precedence pairs**, **Kendall tau-b/tau-c** (tie-aware), and the **Fagin et al. partial-ranking tau-distance with penalty `p`** ([SIAM 2006](https://epubs.siam.org/doi/abs/10.1137/05063088X)) so a correct-but-different linear extension and a correctly-abstained parallel pair are not counted as failures. This alone may move the 0/30 metric substantially if the gold standard contains parallel steps.

**6. Longer-term: train to order end-to-end.** Add a **listwise objective** (ListMLE = Plackett-Luce NLL, [Xia et al. 2008](https://icml.cc/Conferences/2008/papers/167.pdf)) on top of the reranker's per-step scores, or a **pointer/seq2seq decoder** ([Gong et al. 2016](https://arxiv.org/abs/1611.04953); BERSON [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.511/)) that emits the permutation conditioned on the full step set and the partial order so far. This directly optimizes full-sequence accuracy and removes the O(n²)-independent-decision compounding entirely — the sentence-ordering evidence (RankTxNet, BERSON > B-TSort) shows global objectives beat pairwise-then-topo-sort on PMR.

**Minimal change with maximal expected impact:** steps 1-3 (calibrate → weighted tournament → MFAS/Kemeny instead of topo-sort) + step 5 (partial-order evaluation). These directly address why topo-sort scores 0/30: it cannot survive cycles, and exact-chain scoring punishes valid alternative orderings of parallel steps.

---

## References (with URLs)

**Pairwise models & spectral ranking**
- Bradley & Terry 1952, *Biometrika* 39(3/4):324-345 — https://doi.org/10.1093/biomet/39.3-4.324 · https://www.jstor.org/stable/2334029
- Maystre & Grossglauser, "Fast and Accurate Inference of Plackett-Luce Models," NeurIPS 2015 — https://proceedings.neurips.cc/paper/2015/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html
- `choix` library — http://choix.lum.li/en/latest/ · API https://choix.lum.li/en/latest/api.html · https://github.com/lucasmaystre/choix
- Negahban, Oh, Shah, "Rank Centrality," *Operations Research* 65(1):266-287, 2017 — https://arxiv.org/abs/1209.1688 · https://pubsonline.informs.org/doi/10.1287/opre.2016.1534
- Chen, Fan, Ma, Wang, "Spectral Method and Regularized MLE Are Both Optimal for Top-K Ranking," *Ann. Statist.* 2019 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6785035/
- Jiang, Lim, Yao, Ye, "Statistical Ranking and Combinatorial Hodge Theory," *Math. Prog.* 2011 — https://arxiv.org/abs/0811.1067 · https://web.stanford.edu/~yyye/hodgeRank2011.pdf
- Shah & Wainwright, "Simple, Robust and Optimal Ranking from Pairwise Comparisons," JMLR 18(199), 2017 — https://arxiv.org/abs/1512.08949 · https://www.jmlr.org/papers/v18/16-206.html
- Copeland / Condorcet — https://en.wikipedia.org/wiki/Copeland's_method · https://en.wikipedia.org/wiki/Condorcet_method

**MFAS / Kemeny**
- Karp 1972, "Reducibility Among Combinatorial Problems" — https://www.cs.umd.edu/~gasarch/BLOGPAPERS/Karp.pdf
- Alon, "Ranking Tournaments," *SIAM J. Discrete Math.* 20(1), 2006 — https://www.cs.tau.ac.il/~nogaa/PDFS/paley.pdf
- Charbit, Thomassé, Yeo, "The minimum FAS problem is NP-hard for tournaments," 2007 — https://perso.ens-lyon.fr/stephan.thomasse/liste/feedback.pdf
- Conitzer, "Computing Slater Rankings," AAAI-06 — https://www.cs.cmu.edu/~conitzer/slaterAAAI06.pdf
- Kenyon-Mathieu & Schudy, "How to Rank with Few Errors," STOC 2007 — https://cs.brown.edu/~claire/Publis/kenyonschudy.pdf · https://dl.acm.org/doi/pdf/10.1145/1250790.1250806
- Bartholdi, Tovey, Trick, "Voting Schemes for Which It Can Be Difficult to Tell Who Won," *Soc. Choice Welf.* 1989 — https://link.springer.com/article/10.1007/BF00303169
- Dwork, Kumar, Naor, Sivakumar, "Rank Aggregation Methods for the Web," WWW 2001 — https://www.stat.uchicago.edu/~lekheng/meetings/mathofranking/ref/kumar.pdf
- Eades, Lin, Smyth, "A fast and effective heuristic for the FAS problem," *IPL* 47(6), 1993 — https://www.sciencedirect.com/science/article/abs/pii/002001909390079O
- Feedback arc set / Kemeny-Young (overviews) — https://en.wikipedia.org/wiki/Feedback_arc_set · https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method
- Weighted FAS for pairwise ranking (2024) — https://arxiv.org/abs/2412.16181
- NetworkX (no FAS; building blocks) — https://networkx.org/documentation/stable/reference/algorithms/dag.html · igraph FAS — https://igraph.org/r/doc/feedback_arc_set.html

**Sentence ordering & listwise losses**
- Gong et al., "End-to-End Neural Sentence Ordering Using Pointer Network," 2016 — https://arxiv.org/abs/1611.04953
- Logeswaran, Lee, Radev, "Sentence Ordering and Coherence Modeling using RNNs," AAAI 2018 — https://arxiv.org/abs/1611.02654
- Chen, Qiu, Huang, "Neural Sentence Ordering," 2016 — https://arxiv.org/abs/1607.06952
- Prabhumoye, Salakhutdinov, Black, "Topological Sort for Sentence Ordering," ACL 2020 — https://aclanthology.org/2020.acl-main.248/ · https://arxiv.org/pdf/2005.00432
- Cui, Li, Zhang, "BERT-enhanced Relational Sentence Ordering Network (BERSON)," EMNLP 2020 — https://aclanthology.org/2020.emnlp-main.511/
- Kumar et al., "Deep Attentive Ranking Networks (RankTxNet)," AAAI 2020 — https://arxiv.org/abs/2001.00056
- Re-BART, "Is Everything in Order?" EMNLP 2021 (re-tabulates baselines) — https://arxiv.org/abs/2104.07064
- Cao et al., "Learning to Rank: From Pairwise to Listwise (ListNet)," ICML 2007 — https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
- Xia et al., "Listwise Approach to Learning to Rank (ListMLE)," ICML 2008 — https://icml.cc/Conferences/2008/papers/167.pdf

**Calibration & partial-order metrics**
- Platt, "Probabilistic Outputs for SVMs," 1999 — https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods
- Zadrozny & Elkan, "Transforming classifier scores into multiclass probability estimates," KDD 2002 — https://dl.acm.org/doi/10.1145/775047.775151
- Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks," ICML 2017 — https://arxiv.org/abs/1706.04599 · https://proceedings.mlr.press/v70/guo17a.html
- Naeini, Cooper, Hauskrecht, "Obtaining Well Calibrated Probabilities Using Bayesian Binning," AAAI 2015 — https://ojs.aaai.org/index.php/AAAI/article/view/9602
- Kendall, "A New Measure of Rank Correlation," *Biometrika* 1938 — https://www.jstor.org/stable/2332226 · formulas https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
- Goodman & Kruskal, "Measures of Association," *JASA* 1954 — https://www.jstor.org/stable/2281536 · https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_gamma
- Fagin, Kumar, Mahdian, Sivakumar, Vee, "Comparing Partial Rankings," *SIAM J. Discrete Math.* 20(3), 2006 — https://epubs.siam.org/doi/abs/10.1137/05063088X
- Fagin, Kumar, Sivakumar, "Comparing Top k Lists," *SIAM J. Discrete Math.* 17(1), 2003 — https://epubs.siam.org/doi/10.1137/S0895480102412856
- Dilworth, "A Decomposition Theorem for Partially Ordered Sets," 1950 — https://www.jstor.org/stable/1969503 · https://en.wikipedia.org/wiki/Dilworth%27s_theorem

---

### Verification flags (items not quotable verbatim from a primary source)
1. Rank Centrality `ℓ∞`/`ℓ2` error-bound exponents (square roots) in Chen et al. 2019 — confirm against the PDF; the minimax-optimality and top-K threshold are reliable.
2. KMS PTAS exact running-time expression — the `(1+ε)` claim and Kemeny↔weighted-FAST reduction are corroborated; the `n^{O(1/ε…)}` bound was not extracted.
3. `O(log n · log log n)` general-FAS approximation attribution to Even-Naor-Schieber-Sudan — ratio confirmed on Wikipedia; original not opened.
4. BERSON / B-TSort τ/PMR values — from the re-tabulating Re-BART paper, not the originals; exact 2005.00432 table not cleanly extracted.
5. Fagin et al. 2006 `{0,1,p}` penalty-tau definition — reproduced from the 2003 companion + secondary literature; 2006 PDF body not machine-extractable.
6. The end-to-end "calibrate pairwise precedence before weighted aggregation, then threshold/abstain into antichains" pipeline and "precision/recall over precedence pairs" — syntheses of independently-sourced primitives, not single canonical results.
7. FIDE Elo K-factor values — administrative policy, not theory.
