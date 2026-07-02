# Causal Embedding Survey — Reference Implementation

Runnable reference code accompanying the survey on causal embedding models for
AEP/AJO **workflow ordering** and **follow-up reranking**. It implements the
**highest-ROI ideas** from the research notes (`research/07_rank_aggregation_ordering.md`,
`research/08_best_ideas.md`, `research/01_project_context.md`) — specifically the
**Tier-1, no-new-model** wins that target the documented *0/30 ordering failure*:

| Research idea | Module |
|---|---|
| Idea 1 — robust rank aggregation instead of naive topo-sort (the biggest single win) | `ordering.py` |
| Idea 2 — calibrate pairwise scores (temperature scaling) before aggregating | `calibration.py` |
| Idea 4 — evaluate as a **partial order** (tie-aware τ, precedence P/R/F1), not an exact chain | `metrics.py` |
| Stage-2 follow-up reranking (model-free baseline) | `followup_rerank.py` |
| End-to-end demonstration: robust aggregation **beats** naive topo-sort | `run_ordering_demo.py` |

Everything is **CPU-light and dependency-light** — *no torch / transformers are loaded or
trained* (per the hardware/OOM rule). Pairwise scores are read from existing scored JSON
files or simulated.

## Modules

### `ordering.py` — the core "0/30 fix"
Turns a noisy `NxN` matrix `P[i,j] = P(step_i causally precedes step_j)` into a global order.
- `topo_sort_baseline(items, P)` — naive tournament + **topological sort** (Kahn). Reproduces
  the brittle failure mode: it stalls on **any directed cycle** (noisy pairwise scores almost
  always produce A→B→C→A), then emits stuck nodes in arbitrary order.
- `mfas_order(items, P)` — weighted **Minimum-Feedback-Arc-Set** (= Kemeny / minimum-violations).
  Edges weighted by **log-odds** of `P`; solved with **igraph `feedback_arc_set`**
  (Eades-Lin-Smyth heuristic / exact integer-program for n≤12), then polished by a small local
  search. Pure-numpy Eades fallback if igraph is missing. Degrades gracefully on cycles.
- `bradley_terry_order(items, P)` — fit **Bradley-Terry / Plackett-Luce** strengths with
  **`choix`** (I-LSR → MLE), sort by strength. Pure-numpy MM (Zermelo) fallback.
- `kemeny_order(items, P)` — **exact** minimum-violations by permutation search for n≤8, else
  delegates to `mfas_order`.

### `calibration.py` — temperature scaling (Idea 2)
- `fit_temperature(logits, labels)` — fit one scalar `T>0` minimising validation NLL
  (`scipy.optimize.minimize_scalar`). Preserves arg-max, so pairwise accuracy is unchanged.
- `apply_temperature(logits, T)` → calibrated probabilities; `expected_calibration_error`;
  `reliability_diagram(...)` saves a reliability-diagram **PDF** (matplotlib).

### `metrics.py` — partial-order evaluation (Idea 4)
`exact_match` / `perfect_match_rate` (PMR), **tie-aware Kendall τ-b** and **Spearman** (scipy),
and **precedence-pair precision/recall/F1** (credits any correct precedence regardless of which
linear extension was chosen; supports partial-order gold given as ordered buckets/antichains).
`evaluate_orders(preds, golds)` returns the aggregate dict used by the demo.

### `followup_rerank.py` — model-free follow-up reranker (DEMO)
Ranks candidate follow-up queries for a question with a **pluggable** score function:
- `TfidfScorer` — TF-IDF cosine similarity (sklearn) — the lexical/BM25-style baseline.
- `CausalNextStepScorer` — TF-IDF + a "causal next-step" heuristic that up-weights actionable /
  temporal phrasing ("how do I", "after", "troubleshoot", "configure") and down-weights pure
  definitional repeats — preferring the *enabling next action*.
Operates on `/mnt/localssd/z_followupq/prod_jan_feb_26-aep-ajo_follow-up-queries.json`
(334 questions × 10 candidates). No GPU.

### `run_ordering_demo.py` — end-to-end demo
Compares `topo_sort_baseline` vs `mfas_order` vs `bradley_terry_order` vs `kemeny_order` and
prints `metrics.py` metrics for each, over two data sources (below).

## How to run
```bash
cd implementation
python ordering.py          # ordering aggregators demo (incl. a hard cyclic tournament)
python calibration.py       # temperature-scaling demo + saves a reliability-diagram PDF
python metrics.py           # ordering-metrics demo
python followup_rerank.py   # follow-up reranking demo on the real query file
python run_ordering_demo.py # END-TO-END: baseline vs MFAS vs BT vs Kemeny
```

## Dependencies
`numpy`, `scipy`, `scikit-learn`, `python-igraph`, `choix`, `matplotlib` (pandas allowed but
unused). Installed in this environment via `pip install python-igraph choix matplotlib`
(igraph 1.0.0, choix 0.4.1, matplotlib 3.11.0; numpy/scipy/sklearn already present). All three
installed cleanly — the pure-numpy fallbacks in `ordering.py` were not needed here but remain
for portability.

## Real score files wired in
- **`/mnt/localssd/ajo_orchestrated_workflows_flat_scored.json`** (30 workflows, 3 steps; the
  **bi-encoder** scores) — the documented near-0/30 case. Each entry has `correct_order` and a
  `pairs` list with `score = P(label1 precedes label2)`, rebuilt into the `NxN` matrix.
- **`/mnt/localssd/ajo_orchestrated_workflows_flat_scored_ce2x.json`** (8 workflows; the
  **stronger cross-encoder** scores) — the contrasting "good pairwise model" case.
- **`/mnt/localssd/ajo_orchestrated_workflows_flat.json`** (raw text) — used to size the
  controlled simulation.

## Results (captured from `python run_ordering_demo.py`)

### (A) REAL bi-encoder scores — `ajo_orchestrated_workflows_flat_scored.json` (n=30, 3 steps)
```
method                               PMR    tau_b    spear    prec     rec      F1
stored_pipeline_predicted_order    0.067   -0.111   -0.100   0.444   0.444   0.444
topo_sort_baseline                 0.267    0.133    0.133   0.567   0.567   0.567
mfas_order                         0.067   -0.111   -0.100   0.444   0.444   0.444
bradley_terry_order                0.033   -0.178   -0.167   0.411   0.411   0.411
kemeny_order                       0.067   -0.089   -0.067   0.456   0.456   0.456
```
**Finding:** this bi-encoder's pairwise signal is only **~0.41 accurate** vs gold precedence
(73/180 — *worse than chance*). It is a **broken pairwise model**, so no aggregator can recover
it (results are erratic; confidence-weighted MFAS even trusts the confidently-wrong edges).
This matches research `01` §4: bi-encoder = 0/30, the failure is partly the model, not only the
sort. (Temperature scaling correctly drives `T`→its upper bound, detecting a near-useless signal.)

### (A2) REAL cross-encoder scores — `..._scored_ce2x.json` (n=8)
```
method                               PMR    tau_b    spear    prec     rec      F1
topo_sort_baseline                 1.000    1.000    1.000   1.000   1.000   1.000
mfas_order                         1.000    1.000    1.000   1.000   1.000   1.000
bradley_terry_order                0.875    0.917    0.938   0.958   0.958   0.958
kemeny_order                       1.000    1.000    1.000   1.000   1.000   1.000
```
**Finding:** this cross-encoder is **~0.96 pairwise-accurate** (46/48); with a good pairwise
signal *every* aggregator (including topo-sort) gets 8/8. This confirms research `01` §4:
the **cross-encoder's cross-attention is the right backbone**.

### (B) CONTROLLED noisy-pairwise-model simulation — the clean demonstration of Idea 1
A *calibrated* pairwise model of ~0.80 per-pair accuracy (mirroring the documented DeBERTa CE
~21/30) over latent orders of size 3/5/8; correct edges are more confident than wrong ones;
steps are presented **shuffled** (so input order leaks no gold signal). The **same** noisy,
cyclic scores are fed to every method:
```
method                               PMR    tau_b    spear    prec     rec      F1
topo_sort_baseline                 0.104    0.243    0.271   0.622   0.622   0.622
mfas_order                         0.317    0.816    0.881   0.908   0.908   0.908
bradley_terry_order                0.265    0.806    0.886   0.903   0.903   0.903
kemeny_order                       0.317    0.816    0.881   0.908   0.908   0.908
```
**Finding (headline):** with a *decent-but-noisy* pairwise signal that produces cycles, naive
topological sort **collapses** (PMR 0.10, τ-b 0.24, F1 0.62) while weighted **MFAS / Kemeny**
recover most of the order (**PMR 0.32 — 3× higher**, τ-b **0.82**, F1 **0.91**), and
**Bradley-Terry** is close behind — achieved as **pure post-processing, no new model**. This is
the core 0/30 → many-correct fix described in research `07`/`08` Idea 1.

## Honest summary of what the demo shows
1. **Aggregation matters, but only on a usable pairwise signal.** Robust aggregation (MFAS /
   Kemeny / Bradley-Terry) clearly beats naive topo-sort **when the pairwise model is
   decent-but-noisy and cyclic** (source B). It **cannot** rescue a broken pairwise model
   (source A, the real bi-encoder at 0.41 accuracy).
2. **The cross-encoder is the right backbone** (source A2: 0.96 pairwise → 8/8).
3. **Calibration + partial-order metrics** are the cheap enablers: temperature scaling makes the
   log-odds weights meaningful, and tie-aware τ-b / precedence-F1 reveal partial-credit progress
   that exact-PMR hides.

The two highest-ROI next steps from the research that need a (separate, GPU) retrain — **train on
the full within-doc transitive closure (drop MAX_GAP)** and **add a NEUTRAL/abstain class for
parallel steps** — are intentionally out of scope here (they require model training, which the
OOM rule forbids in this environment), but the aggregation + calibration + evaluation machinery
above is exactly what consumes their outputs.
