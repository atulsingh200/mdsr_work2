where it stays at **0.9143**. Why?

**Short answer.** BM25 picks negatives by *word overlap with the anchor*. On `followupqg`, the gold positives (short conversational follow-ups) **don't share many words with their anchors** either — so BM25's lexical proxy for "hard" doesn't actually find semantically close negatives. The semantic encoder still has a wide margin.

---

## Numbers — followupqg, 501 anchors, MiniLM-L6-v2 (off-the-shelf)

| Quantity | Cosine sim |
|---|---:|
| `sim(anchor, gold positive)` | **0.4529** |
| `sim(anchor, single BM25 neg)` (avg) | 0.1267 |
| `sim(anchor, hardest BM25 neg)` (worst per anchor, then averaged) | **0.2728** |
| **Gap: gold − hardest neg** | **+0.1801** |
| % anchors where some BM25 neg outscores gold | 20.2 % |

The gold is, on average, **0.18 cosine units closer to the anchor than even the hardest BM25 negative**. BM25 found lexical near-misses, not semantic ones, so the encoder still wins on ~80 % of anchors.

---

## Example — easy win  (gap = +0.46)

**Anchor.** *"ELI5: Why does bass in speakers seem to limit at higher volumes? The speaker may have circuitry designed to limit bass at higher volume to prevent damage to itself. Small speakers aren't good at producing bass anyway."*

| | Cosine sim with anchor |
|---|---:|
| **Gold:** *"Thank you!! …is there a reason why speakers that hold more bass are quieter than a speaker that pushes the mids and highs more?…"* | **0.7503** |
| BM25 neg #1: *"Thank you. What causes the air in the atmosphere to have higher pressure…"* | 0.1630 |
| BM25 neg #3: *">Your ears pick up more bass from your voice resonating in your head…"* | 0.2909 |

BM25 surfaced topical-word matches ("bass", "voice") that aren't the actual follow-up. The gold's specific reaction outscores all of them — by a lot.


## Takeaway

BM25 is a **lexical** hard-negative miner. On datasets where positives and anchors *don't* share much vocabulary (like `followupqg`, where follow-ups use new words), BM25's "hard" negatives aren't actually semantically hard — and a semantic encoder still ranks the gold above them. On every other dataset in this benchmark (where positives *do* share anchor vocabulary), BM25 surfaces genuine near-misses and the AUC collapses.

Source code: [`bm25_semantic_diagnostic.py`](bm25_semantic_diagnostic.py) · full numbers: [`bm25_semantic_diagnostic.md`](bm25_semantic_diagnostic.md).
