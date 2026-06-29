# Reverse-Direction Finetuning Results

**Loss:** Pairwise DPO-style preference loss
```
L = -log σ( (s(A,B) - s(B,A)) / τ )

s(A,B) = enc_anchor(A) · enc_positive(B)   ← forward  (label 1)
s(B,A) = enc_anchor(B) · enc_positive(A)   ← reverse  (label 0)
```
No hard-negative mining needed. Each `(A, B)` pair supplies both signals by swapping encoder roles.

**Backbone:** `bert-base-uncased` · CLS pooling · untied two-tower · learnable τ (init 0.05)

---

## Results — forward s(A→B) vs reverse s(B→A), 2 candidates per pair

| dataset | N test | AUC | P@1 | mean s(A→B) | mean s(B→A) |
|---|---:|---:|---:|---:|---:|
| aep_causal | 562 | 0.8861 | 0.8861 | 0.4519 | 0.1529 |
| followupqg | 501 | **1.0000** | **1.0000** | 0.6889 | −0.2293 |
| multiwoz_v24 | 7,368 | 0.9976 | 0.9976 | 0.6292 | −0.3450 |
| qrecc | 5,204 | 0.9862 | 0.9862 | 0.5074 | −0.1063 |
| workflow | 260,487 | 0.8852 | 0.8852 | 0.5205 | 0.1371 |

AUC = fraction of test pairs where s(A→B) > s(B→A). Chance = 0.5.

---

## Observations

- **followupqg** and **multiwoz_v24** achieve near-perfect directional discrimination (AUC 1.00 / 0.998). The reverse scores go deeply negative, meaning the model strongly repels the wrong direction.
- **qrecc** is also very strong (AUC 0.986).
- **aep_causal** and **workflow** are good but lower (AUC ~0.885). Both have long, overlapping texts where the two towers see similar content regardless of swap — harder to separate directionally.
- Across all datasets the reverse score s(B→A) is consistently much lower than s(A→B), confirming the model has learned the directional preference.

---

## Improvement over old eval (fixed)

The previous eval computed `enc_anchor(A) · enc_positive(A)` as the reverse score (self-similarity), which does **not** match the training objective. The current eval uses the correct `enc_anchor(B) · enc_positive(A)` = s(B,A), consistent with the loss.

---

## Training config

| param | value |
|---|---|
| loss | pairwise DPO: `-log σ((s(A,B)−s(B,A))/τ)` |
| backbone | `google-bert/bert-base-uncased` |
| pooling | CLS |
| batch size | 32 |
| lr | 2e-5 (encoder), 1e-3 (τ) |
| epochs | 3 (small datasets), 2 (large) |
| train caps | aep_causal: 5,487 · followupqg: 2,790 · multiwoz: 15k · qrecc: 15k · workflow: 20k |

## Reproduce

```bash
bash finetune_eval/train_all_reverse.sh
```
