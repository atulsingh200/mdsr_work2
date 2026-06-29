# GRITLM for Causal (Directional) Embeddings — Experiments

Can **GRITLM-7B** ([Generative Representational Instruction Tuning](https://arxiv.org/abs/2402.09906),
Muennighoff et al. 2024) represent **causal/temporal relatedness** rather than plain semantic
similarity? Concretely, for an anchor *a* (cause / step at time *t*) and a positive *b*
(effect / next step at *t+1*):

- **High** score for the true continuation *a → b*
- **Low** score for a mere paraphrase of *a*, and for the **reversed** order *b → a*

Dataset: `data_6/aep_causal` (Adobe Experience Platform doc pairs; `anchor` = cause, `positive` = effect).
Splits: train 5487 / val 562 / test 562.

---

## TL;DR

| | Direction acc (a→b vs b→a) | Retrieval acc@1 | Retrieval MRR |
|---|:---:|:---:|:---:|
| **Zero-shot** GRITLM (instructions only) | **0.46–0.58** (≈ chance) | 0.40 | 0.557 |
| **Bi-encoder** LoRA finetune (full data)    | **0.855**      | 0.160 | 0.282 |
| **Cross-encoder** LoRA finetune (full data) | **0.941**      | 0.080 | 0.179 |

- **Zero-shot, GRITLM has no causal directionality** — even with explicit directional cause/effect
  instructions it scores *a→b* and *b→a* the same (margin ≈ 0.002), and ranks paraphrases of the cause
  **above** the true effect. It is a pure semantic-similarity model.
- **Directionality is learnable via LoRA.** Using the **reversed pair `b→a` as a hard negative**,
  both architectures learn it; the **cross-encoder wins decisively (0.941)**.
- **Retrieval degrades** under the direction objective — the reverse-pair negative pushes apart
  topically related text. The cross-encoder is a **reranker** (O(N²)), not a first-stage retriever.

**Recommended design:** bi-encoder retriever (fast, O(N)) → cross-encoder reranker (0.94 directional acc).

---

## Task & metrics

**Direction accuracy** — pairwise 2-way choice, pool = **1 hard negative** (same two texts, reversed).
Random baseline = 0.50. Evaluated on all **562** test pairs.
```
correct  iff  score(a→b) > score(b→a)
```

**Retrieval acc@1 / MRR** — build an N×N score matrix `S[i][j] = score(anchor_i, effect_j)`; for each
query row *i* the gold answer is column *i*; negatives = the other **N−1** effects.
```
acc@1 = mean(rank_i == 1)      MRR = mean(1 / rank_i)
```
⚠️ Retrieval pools differ between runs (bi-encoder N=100, cross-encoder N=50), so the retrieval
numbers are **not** directly comparable; direction accuracy (N=562) **is**.

---

## Experiments & results

### 1. Zero-shot (inference only) — `test_gritlm_causal.py`, `test_gritlm_direction.py`

- **Toy** "Turn off the switch": ranks paraphrases ("flip the switch off" 0.78) **above** the causal
  effect ("the bulb goes dark" 0.65), under both semantic and causal instructions.
- **Directionality** (N=50): acc **0.46**, mean forward == mean reverse (margin **+0.0001**).
  → asymmetric instructions produce essentially **symmetric** embeddings; no causal signal.

### 2. Bi-encoder LoRA finetune — `finetune_gritlm_direction.py`

Two embeddings (cause-role instr on *a*, effect-role instr on *b*) + cosine.
Loss = `InfoNCE(in-batch) + λ_dir · margin(forward > reverse)`, λ_dir = 1.0, MARGIN = 0.05.

| | PRE | POST |
|---|:---:|:---:|
| Direction acc | 0.575 | **0.805** |
| Direction margin (fwd − rev) | +0.002 | +0.109 |
| Retrieval acc@1 (pool 100) | 0.400 | 0.160 |
| Retrieval MRR | 0.557 | 0.282 |

### 3. Cross-encoder LoRA finetune — `finetune_gritlm_pairwise.py`

Anchor & positive **concatenated** into one input (`PASSAGE 1: a … PASSAGE 2: b`), mean-pooled,
then a `Linear(4096→1)` head → **one raw scalar** (logit; sigmoid only inside the BCE loss).
Negative = same pair **reversed** (`PASSAGE 1: b … PASSAGE 2: a`).


| | PRE | POST |
|---|:---:|:---:|
| Direction acc | 0.521 | **0.941** |
| Direction margin (pos − neg) | +0.051 | **+2.678** |
| Retrieval acc@1 (pool 50) | 0.020 | 0.080 |
| Retrieval MRR | 0.083 | 0.179 |

Both: GRITLM-7B backbone frozen, **LoRA r=16, α=32, dropout 0.05** on all attn + MLP projections
(~42M trainable, 0.59%); AdamW lr 1e-4; **1 epoch** over all 5487 train pairs; bf16; bidirectional
attention + mean pooling (instruction tokens excluded from the pool).

---

## Why the cross-encoder wins on direction

The cross-encoder feeds *a* and *b* into the model **together**, so bidirectional attention lets every
token of *a* attend to *b* and judge whether the ordering is causal. The bi-encoder must compress each
passage **independently** into a vector and can only express direction through the (weak) asymmetric
instruction channel. Margins reflect this: **+2.68** (cross) vs **+0.11** (bi).

The flip side: the cross-encoder produces **no standalone embedding**, so retrieval requires scoring
every candidate (O(N²)) — usable only as a reranker over a shortlist.

---

## Reproduce

Isolated env (the project's own `.venv` has `transformers 5.9`, which is **incompatible** with GRITLM's
2024 remote code — `config.rope_theta` `AttributeError`):

```bash
# /mnt/localssd/gritlm_test_env : python3.10, torch 2.4.1+cu124, transformers==4.44.2, gritlm 1.0.2
PY=/mnt/localssd/gritlm_test_env/bin/python
export HF_HOME=/mnt/localssd/.cache/huggingface
```

```bash
# zero-shot probes
CUDA_VISIBLE_DEVICES=0 $PY test_gritlm_causal.py
CUDA_VISIBLE_DEVICES=0 $PY test_gritlm_direction.py

# bi-encoder finetune (full data), GPU 1
CUDA_VISIBLE_DEVICES=1 N_TRAIN=100000 B=8 EPOCHS=1 N_EVAL_DIR=562 N_EVAL_RET=100 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY finetune_gritlm_direction.py > ft_biencoder_full.log 2>&1

# cross-encoder finetune (full data), GPU 0
CUDA_VISIBLE_DEVICES=0 N_TRAIN=100000 B=4 ACCUM=2 MAXLEN=512 EPOCHS=1 N_EVAL_DIR=562 N_EVAL_RET=50 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY finetune_gritlm_pairwise.py > ft_pairwise_full.log 2>&1
```

Key env knobs: `N_TRAIN` (cap, 100000 ⇒ all 5487), `B`/`ACCUM` (batch / grad-accum), `MAXLEN`,
`EPOCHS`, `N_EVAL_DIR`, `N_EVAL_RET`, `LR`, `MARGIN`. One epoch ≈ 35–40 min on a single A100.

---

## Files

| File | What |
|---|---|
| `test_gritlm_causal.py` | Zero-shot: toy causal-vs-semantic + small retrieval probe |
| `test_gritlm_direction.py` | Zero-shot directionality (forward vs reverse) |
| `finetune_gritlm_direction.py` | **Bi-encoder** LoRA finetune (cosine + InfoNCE + direction margin) |
| `finetune_gritlm_pairwise.py` | **Cross-encoder** LoRA finetune (concat pair + scalar head, reversed = neg) |
| `ft_biencoder_full.log`, `ft_pairwise_full.log` | Full-dataset PRE/POST logs (results above) |

---

## Next steps

- **Recover retrieval:** lower `λ_dir` (0.1–0.3) / larger batch so the direction objective stops
  destroying semantic retrieval (target: dir ≥ 0.80 **and** MRR > 0.50).
- **2-stage pipeline:** bi-encoder top-k retrieve → cross-encoder rerank; measure end-to-end.
- **Fair retrieval eval:** rerun both with the same (ideally full 562) candidate pool.
- More epochs / hard-negative mining beyond the reversed pair.
