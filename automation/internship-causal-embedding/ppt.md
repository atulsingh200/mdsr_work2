# Directional Causal Classification
### From hand-built tiers → an automated, any-product pipeline

A walkthrough of how the dataset is built, what we automated, how the model
trains, and the final results.

---

## Slide 1 — The Goal

Train a classifier that understands **learning order** in product documentation.

> Given two snippets of documentation text, which one is **more foundational**
> (should be learned first)?

- `label = 1` → `text_1` comes **before** `text_2` (correct order)
- `label = 0` → reversed order

The key idea: assign every doc a **tier** (1 = basics → N = advanced), then teach
the model to tell the direction between tiers.

---

## Slide 2 — The Original Builder: What We Start With

**`build_classification_data.py`** takes two inputs:

```
TOC.md  (Adobe GitHub)                 corpus.json  (scraped pages)
─────────────────────                  ──────────────────────────────
+ Introduction                         [
  + Overview        {#2a}                { sourceUrl: ".../overview",
    + JO Overview                          chunks: [{data: "AJO is…"}] },
+ Channels                               { sourceUrl: ".../push-channel",
  + Push Channel    {#9h}                  chunks: [{data: "Push lets…"}] },
    + Push Overview                      ...
                                       ]
```

- **TOC.md** = the *learning structure* (what order topics come in)
- **corpus.json** = the *actual page text*, stored as chunks per URL

---

## Slide 3 — The Builder Pipeline (Steps 1–4)

```
STEP 1  PARSE TOC → LEAVES
   leaf_A = {title:"JO Overview", chapter:"intro", slug:"journey-optimizer-overview"}
   leaf_B = {title:"Push Overview", chapter:"channels", sub_anchor:"push-channel"}

STEP 2  ASSIGN SUB-CHAPTER CODE & TIER     ★ THE MANUAL STEP ★
   leaf_A → sub "2a" → TIER 1  (Product basics)
   leaf_B → sub "9h" → TIER 6  (Channels)
   T1 ── T2 ── T3 ── ... ── T14 ── T15
   Basics Admin Data        AI    Capstone

STEP 3  MATCH LEAF → CORPUS DOC (try URLs most-specific first)
   leaf_A → corpus_doc_A (tier 1)   leaf_B → corpus_doc_B (tier 6)

STEP 4  RECONSTRUCT TEXT (join all chunks[].data with "\n\n")
   body_A = "Adobe Journey Optimizer is a...\n\nKey capabilities...\n\n..."
```

---

## Slide 4 — The Builder Pipeline (Steps 5–7)

```
STEP 5  SEGMENT each doc (sentence-based, 2–5 sentences per segment)
   doc_A → [seg_A0, seg_A1, seg_A2]   doc_B → [seg_B0, seg_B1]

STEP 6  SPLIT docs → TRAIN / VAL / TEST  (per tier, no leakage:
   all of a doc's segments share one split)

STEP 7  EMIT DIRECTIONAL PAIRS across tiers
   seg_A0 (T1) × seg_B0 (T6):
     FORWARD : text_1=A, text_2=B → label 1
     REVERSE : text_1=B, text_2=A → label 0
```

**Output:** `directional_{train,val,test}.jsonl` + `manifest.json`
→ **110,775 / 6,004 / 6,004** rows (real AJO run).

---

## Slide 5 — The Manual Bottleneck

**Only STEP 2 needs a human.** Steps 1, 3–7 are deterministic code.

STEP 2 reads three **hand-written** tables baked into the builder
(`build_classification_data.py`, ~250 lines):

| Table | Hand-authored content |
|---|---|
| `SUBCHAPTER_TO_TIER` | `"2a":1, "9h":6, ...` (which sub-chapter → which tier) |
| `CHAPTER_RULES` | `"journey-optimizer-overview":"2a", ...` (slug → sub-chapter) |
| `BY_ABSOLUTE_URL` | external-URL → sub-chapter overrides |

> A person read all 188 docs, designed the 15-tier order, and typed every mapping.
> **AJO-only — must be rewritten by hand for every new product.**

---

## Slide 6 — What We Automate

Replace the human-authored tier tables with a **Claude call**.

```
BEFORE                              AFTER
──────                              ─────
STEP 2 reads tables a HUMAN wrote   STEP 2 reads tables CLAUDE wrote
(weeks of work, AJO only)           (seconds, ANY Adobe product)
```

- **Input to Claude:** the doc list (url, title, chapter, sub_anchor, slug)
- **Claude's job (system prompt):** infer the prerequisite-ordered tier
  hierarchy + assign every doc → `assignments.json`
- Everything else in the builder is **unchanged**.

---

## Slide 7 — The New Automated Pipeline

```
TOC.md + corpus.json + --product "Adobe Express"
        │
        ▼
generate_dataset.py
  • parse TOC → resolve docs against corpus → build doc list
        │  doc list + TIER_GENERATION_SYSTEM_PROMPT.md
        ▼
  ✨ CLAUDE (via Bedrock)  → infers tiers, assigns every doc
        │  assignments.json
        ▼
build_dataset_from_assignments.py   (same deterministic mechanics)
        │
        ▼
  directional_{train,val,test}.jsonl
```

---

## Slide 8 — Sample Data (with labels)

From `directional_test.jsonl` — each row is a labeled pair:

**Example A — `label = 1`** (foundational → advanced, correct order)
```
text_1: "...here’s my unsubscription page and the confirmation page..."   (Tier 4, sub 8c)
text_2: "...consent policies, data governance in Journey Optimizer..."     (Tier 12, sub 18b)
label : 1     tier_1=4  <  tier_2=12   ✓ correct direction
```

**Example B — `label = 0`** (advanced → foundational, reversed)
```
text_1: "...select the code-based experience channel and click create..."  (Tier 6, sub 9b)
text_2: "...unsubscription page and the confirmation page..."              (Tier 4, sub 8c)
label : 0     tier_1=6  >  tier_2=4    ✗ reversed direction
```

---

## Slide 9 — The Model: How It Reads a Pair

Two **untied** encoders + a feature combiner + a head.

```
text_1 → encoder_A → A           text_2 → encoder_B → B
            (CLS pooling + L2 normalize, 384-dim each)
                 └──────────┬──────────┘
                            ▼
          feature = [ A ; B ; A−B ; A∗B ]   →  1536-dim
                            ▼
                    classification head  →  single logit
```

Why these four pieces (`heads.py`):

| Part | Captures |
|---|---|
| `A`, `B` | absolute position of each text in embedding space |
| `A − B` | **direction** — flips sign when you swap the texts (→ label flips) |
| `A ∗ B` | **similarity** — are they even on related topics? |

---

## Slide 10 — Training Setup

`train-classifier` (config from README "best run"):

- **Backbone:** `BAAI/bge-small-en-v1.5` (two untied towers, fine-tuned)
- **Head:** MLP `1536 → 512 → 128 → 1` (LayerNorm + GELU + Dropout)
- **Loss:** `BCEWithLogitsLoss` on the directional logit
- **Data:** 110,775 train / 6,004 val / 6,004 test
- Best checkpoint tracked by **val accuracy**; final eval on the test split with
  per-tier-pair accuracy breakdown.

```bash
uv run train-classifier \
  --data-dir data/aep_causal_classification \
  --head-type mlp --head-hidden-dims 512,128 \
  --epochs 3 --batch-size 32 --lr-encoder 2e-5 --lr-head 1e-3
```

---

## Slide 11 — Final Test Results

**MLP best-config** — `BAAI/bge-small-en-v1.5` + MLP `512 → 128`

| Metric | Score |
|---|---|
| **Test Accuracy** | **0.8571** |
| **Test AUC** | **0.9031** |
| **Test F1** | **0.8558** |
| **Test Loss** | **1.1182** |

→ The model correctly orders ~86% of unseen doc-snippet pairs, with strong
ranking quality (AUC 0.90).

---

## Slide 12 — Summary

```
┌────────────┬────────────────────────┬─────────────────────────────┐
│            │        BEFORE          │            AFTER            │
├────────────┼────────────────────────┼─────────────────────────────┤
│  TIERS     │  hand-written tables   │  Claude + system prompt     │
│            │  (AJO only, weeks)     │  (any product, seconds)     │
│  BUILDER   │  build_classification  │  build_dataset_from_        │
│            │  _data.py              │  assignments.py             │
│            │   (same mechanics — unchanged in both)               │
│  OUTPUT    │     directional_{train,val,test}.jsonl               │
│  MODEL     │  dual-encoder + [A;B;A−B;A∗B] + MLP head             │
│  RESULT    │  Acc 0.857 · AUC 0.903 · F1 0.856                    │
└────────────┴────────────────────────┴─────────────────────────────┘
```

**We automated the one manual step** (tier authoring) and kept everything else,
so the same dataset can now be built for **any Adobe product** — and the trained
classifier reaches **85.7% test accuracy**.

---

## Slide 13 — GritLM: the state-of-the-art encoder

**GRIT — Generative Representational Instruction Tuning** (arXiv 2402.09906)

**Summary:** one LLM does **both embedding and generation**, selected by an
instruction.
- **Embedding mode:** bidirectional attention + mean pooling (contrastive loss)
- **Generation mode:** causal attention + LM head (next-token loss)
- **GRITLM 7B** (from Mistral 7B) sets open-model **SOTA on MTEB (66.8)** while still
  beating all generative models its size — unifying both at no performance loss, and
  speeding up RAG by **>60%**.

**Why we use it:** it is **state-of-the-art in *both* embedding and generation**, so a
single model can serve the tier-direction encoder *and* any generative needs. Its
**instruction-conditioned** embeddings let us tell it exactly what to represent.

**Input format (embedding mode):**
```
<s><|user|>
{instruction}        ← e.g. "Given a scientific paper title, retrieve the paper's abstract"
<|embed|>
{text to represent}  ← e.g. "Bitcoin: A Peer-to-Peer Electronic Cash System"
```
→ Mean-pool the final hidden states (instruction tokens **excluded** from pooling) →
embedding vector.
## Slide 14 — GritLM
**Encoder-side performance — MTEB (56 datasets):**

| Task → | Classif. | Cluster. | PairClassif. | Rerank | Retrieval | STS | Summ. | **Avg** |
|---|---|---|---|---|---|---|---|---|
| **Metric** | Acc. | V-Meas. | AP | MAP | nDCG | Spear. | Spear. | — |
| **# datasets** | 12 | 11 | 3 | 4 | 15 | 10 | 1 | **56** |
| **GRITLM 7B** | 79.5 | 50.6 | 87.2 | 60.5 | 57.4 | 83.4 | 30.4 | **66.8** |
| E5 Mistral 7B | 78.5 | 50.3 | 88.3 | 60.2 | 56.9 | 84.6 | 31.4 | 66.6 |
| BGE Large (0.34B) | 76.0 | 46.1 | 87.1 | 60.0 | 54.3 | 83.1 | 31.6 | 64.2 |

**Training data:** embedding = E5 + S2ORC scientific (**"E5S"**); base model = Mistral 7B.

**What each metric means:**
- **Acc.** — % of items classified correctly (Classification).
- **V-Measure** — clustering quality; balances homogeneity + completeness (Clustering).
- **AP** (Average Precision) — area under the precision–recall curve (Pair Classification).
- **MAP** (Mean Average Precision) — mean per-query average precision; rewards correct ranking order (Reranking).
- **nDCG** — normalized discounted cumulative gain; ranking quality weighting hits near the top (Retrieval).
- **Spearman** — rank correlation between predicted and human similarity scores (STS, Summarization).
