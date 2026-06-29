# AUC / P@1 with Hard Negatives + Reverse Pair

A much harder, direction-aware evaluation than the standard random-negative AUC/P@1.

## Setup

For every test pair `(a_i, b_i)`, the gold `a_i → b_i` (label 1) is scored against:
- **7 semantic hard negatives** `a_i → b_j` — the model's own nearest other positives
  (kNN in the positive-tower space; chunked on GPU, never N×N on CPU).
- **the reverse pair** `b_i → a_i` (label 0) — the same texts, direction flipped:
  `b_i` through the anchor tower, `a_i` through the positive tower.

The reverse pair is the crucial addition: it is topically identical to the gold, so
only a model that captures **direction** (not just topic) can rank gold above it.

Scoring is the model's own:  `cos(space) + β·mean_k σ(τ·(t_cand − t_query))`.
For space-only models β=0; for the time model we sweep β.

Embeddings + mined hard-neg indices are cached to disk per checkpoint
(`hardneg_reverse_cache/`) so reruns are cheap and CPU RAM stays bounded.

## Results (test set)

| Dataset | AUC vanilla | AUC lorentz | AUC lorentz+time | P@1 vanilla | P@1 lorentz |
|---------|------------:|------------:|-----------------:|------------:|------------:|
| aep_causal   | 0.692 | **0.706** | 0.692 | **0.320** | 0.311 |
| followupqg   | 0.863 | **0.868** | 0.855 | 0.747 | **0.760** |
| multiwoz_v24 | **0.600** | 0.599 | — | **0.221** | 0.215 |
| qrecc        | 0.601 | **0.616** | — | 0.200 | **0.210** |

`vanilla` = e5 + forward InfoNCE, K=4, space-only.
`lorentz` = e5 + bidirectional InfoNCE, K=7, space-only (main model).
`lorentz+time` = e5 + ordering loss + deeper time head (β=0.3 in scoring).

### Directional accuracy (gold forward > reverse pair)

| Dataset | vanilla | lorentz | lorentz+time |
|---------|--------:|--------:|-------------:|
| aep_causal   | 0.751 | 0.763 | 0.760 |
| followupqg   | 0.958 | 0.970 | 0.942 |
| multiwoz_v24 | 0.954 | 0.957 | — |
| qrecc        | 0.870 | 0.914 | — |

### β-sweep for the time model (does the Lorentzian term help?)

| β | aep AUC | aep dir-acc | fq AUC | fq dir-acc |
|---|--------:|------------:|-------:|-----------:|
| 0.0 | 0.6925 | 0.760 | 0.8546 | 0.952 |
| 0.3 | 0.6922 | 0.760 | 0.8553 | 0.942 |
| 1.0 | 0.6898 | 0.756 | 0.8534 | 0.924 |
| 2.0 | 0.6847 | 0.744 | 0.8426 | 0.862 |

## Conclusions

1. **This eval is genuinely hard.** AUC falls from ~0.97 (random negatives) to
   0.60–0.87, and P@1 from ~0.93 to 0.20–0.76. The earlier random-negative
   AUC/P@1 were easy; hard-neg + reverse is a real stress test.

2. **vanilla ≈ lorentz.** The bidirectional/K=7 "method" gives a hair's-edge on
   3 of 4 datasets (aep +0.014 AUC, qrecc +0.015) and ties elsewhere. No
   meaningful separation — consistent with the earlier finding that the backbone,
   not the method, drives quality.

3. **The Lorentzian time head does NOT help reject the reverse pair.** Adding the
   time term monotonically *lowers* AUC and directional accuracy (β-sweep). The
   deeper MLP time head, even trained with an explicit ordering loss, contributes
   nothing on the honest reverse-pair test.

4. **Direction IS partly captured — but by the untied towers, not the clock.**
   Directional accuracy is 0.75–0.97 *for the space-only models too*. Because the
   anchor and positive towers are separate networks, `cos(anchor(a), pos(b)) ≠
   cos(anchor(b), pos(a))`, which already encodes some asymmetry. The Lorentzian
   time term adds no further directional signal on top of that.

   > Note: an earlier "directional accuracy = 1.000" came from a *degenerate*
   > metric that flipped only the time-term sign while keeping the forward
   > embeddings. Encoding the reverse pair properly (swapping which tower each
   > text goes through) gives the 0.75–0.97 numbers above.

**Bottom line:** on a direction-aware hard test, the winning model is still "a
strong e5 bi-encoder." The two untied towers already supply most of the usable
directional signal; the explicit Lorentzian time component does not improve
hard-negative or reverse-pair discrimination.
