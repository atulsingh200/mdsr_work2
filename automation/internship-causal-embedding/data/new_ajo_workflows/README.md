# `new_ajo_workflows` — Execution-Order (Workflow) Pair Dataset

This directory contains `directional_train.jsonl`, `directional_val.jsonl`, and
`directional_test.jsonl`: pairs of sentences from Adobe Experience League ("AJO"/AEP)
**tutorial transcripts**, labeled with which sentence describes the step that is
executed **first**.

**Builder script:** [`src/followup_data/builders/build_workflow_order.py`](../../src/followup_data/builders/build_workflow_order.py)
This was verified against the actual data files below — the schema, split sizes, per-document
row caps, and exact 50/50 label balance all match what the script produces. (Note: the script's
`OUTDIR` constant points to `/home/claude/data_workflow_order`, a sandbox path from the run that
generated this data; the output files were copied here afterwards. The generation logic itself is
otherwise unmodified.)

---

## 1. What problem this dataset is for

The other `directional_*.jsonl` datasets in this repo (tier/subchapter based) teach a model
"topic A conceptually precedes topic B" at a coarse, cross-document level. This dataset instead
teaches **fine-grained procedural ordering**: given two individual action steps taken from the
*same* how-to video transcript, which one a user actually performs first. It is a same-document,
sentence-level complement to the coarse tier ordering signal.

## 2. Step-by-step: how the dataset is built

1. **Load the raw AEP/AJO doc corpus.**
   `load_docs()` reads `aep_docs_collection_v_24-03-2026.json`. Each entry is a group of chunks
   sharing one `sourceUrl`. For every chunk, the text is taken from `data`/`text`/`content`
   (first one present), and any chunk that is more than 30% raw URL characters
   (`_urlfrac(t) <= 0.3` filter) is dropped as boilerplate/link-spam. The surviving chunk texts
   for a URL are joined with newlines into one `body` string. Result: `docs[url] = {title, text}`.

2. **Keep only tutorial pages.** `is_tutorial(url)` filters down to URLs containing `/tutorials/`
   — these are the "Learn" video-walkthrough pages, which is where a real sequential
   step-by-step procedure exists (as opposed to reference/conceptual docs, which don't have a
   single linear order).

3. **Split each transcript into sentences.** `sentences(text)`:
   - strips the trailing "recommendation-more-help..." boilerplate block (`HELP_RE`),
   - collapses whitespace,
   - splits on `.`/`!`/`?` sentence boundaries.

4. **Filter sentences down to genuine action steps.** `action_steps(text)` keeps a sentence only if
   **all** of the following hold:
   - word count is between 5 and 60 (`min_words`/`max_words`) — drops fragments and run-on
     narration blocks,
   - it does **not** match `DROP_RE` — narration/meta phrases like *"in this video"*, *"let's
     first look"*, *"the framework"*, *"thanks for watching"*, *"on one hand / on the other
     hand"*, *"overview of"* are excluded because they describe the video, not an action,
   - it **does** match `ACTION_RE` — must contain at least one of ~35 imperative action verbs
     (`select, click, choose, navigate, go to, open, create, add, enter, type, drag, drop, name,
     save, publish, configure, set, define, map, ingest, upload, import, export, build, enable,
     toggle, check, fill, specify, activate, send, launch, review, preview, test, apply, assign,
     switch, search, scroll, hit, press, pick, provide, confirm`).

   The surviving sentences, **in their original transcript order**, are the candidate "steps"
   for that tutorial.

5. **Discard thin transcripts.** A tutorial is only used if it yields **≥ 3 action steps**
   (`len(steps) >= 3`) — otherwise there's no meaningful sequence to learn from.

6. **Resolve tier/subchapter metadata (best-effort, cosmetic).** `resolve_sub_tier(url)` looks
   the doc's URL slug up in the tier-classification builder's `CHAPTER_RULES` table (reused from
   the main classification-dataset build script) to attach a `tier`/`sub` label. This is carried
   through purely for schema compatibility with the other `directional_*.jsonl` files — it plays
   **no role in the ordering/pairing logic**. When a slug isn't found in the table both fields
   are `null` (this is common — about 84% of rows in `directional_train.jsonl` have `tier_1 =
   null`, since most tutorial slugs aren't in the classification chapter table).

7. **Document-level train/val/test split (leakage-safe).** All qualifying tutorial URLs are
   shuffled with a fixed seed (`random.Random(42)`) and split **by document**: 10% test, 10%
   val, 80% train (`int(0.1*n)` each for test/val, remainder train). Splitting by document
   (not by pair) means no two sentences from the same transcript ever appear in different
   splits — this prevents the model from memorizing a specific transcript's step order in
   training and then "seeing" a held-out pair from the same doc at eval time.

8. **Build candidate step pairs within a sliding window.** For each document's ordered step list
   `steps[0..n-1]`, every pair `(steps[i], steps[j])` with `i < j <= i + MAX_GAP` (`MAX_GAP = 4`)
   is generated. This windowing is deliberate: pairing *every* step against *every later* step
   (`O(n^2)`) would let steps that are many paragraphs apart — and often topically unrelated even
   within the same tutorial — dominate the dataset. Restricting `j - i` to at most 4 keeps pairs
   locally coherent (they describe parts of the same sub-procedure) instead of coarse
   whole-document narrative order.

9. **Shuffle and cap pairs per document.** The candidate pairs for a doc are shuffled and
   truncated to at most `PER_DOC_CAP = 60` pairs, so a handful of very long transcripts can't
   dominate the dataset numerically relative to short ones.

10. **Emit both directions (antisymmetric supervision).** For every retained pair `(a, b)`
    where `a` occurs before `b` in the transcript, **two** rows are written:
    - `{text_1: a, text_2: b, label: 1}` — "`a` is executed before `b`" (forward, correct order)
    - `{text_1: b, text_2: a, label: 0}` — the same two sentences, swapped, labeled "false"

    This is the core labeling logic and the reason the dataset is exactly 50/50 balanced by
    construction: it is **not** teaching "label 1 = these two texts are related"; it's teaching a
    genuine antisymmetric comparator — *for this same pair of texts*, order matters, and swapping
    which one is `text_1` must flip the label. This prevents the model from learning a
    topic/register/verb shortcut (e.g. "sentences with 'select' are always label 1") instead of
    actually comparing the two positions.

    So each retained `(a, b)` candidate produces exactly 2 rows → per-doc cap is really up to
    `60 * 2 = 120` rows (confirmed empirically: the max row count for any single document in
    `directional_train.jsonl` is exactly 120).

11. **Shuffle and write out.** Rows for each split are shuffled once more and written to
    `directional_{train,val,test}.jsonl`, one JSON object per line.

12. **Sanity logging.** The script prints doc/row counts and the train label balance to stderr
    for a quick build-time sanity check.


## 4. Verified dataset statistics

| split | rows   | unique docs | label balance      |
|-------|--------|-------------|---------------------|
| train | 21,846 | 253         | 10,923 / 10,923 (1/0) |
| val   | 2,470  | 31          | 1,235 / 1,235       |
| test  | 2,616  | 31          | 1,308 / 1,308       |


## 5. Sample data — a reconstructed step sequence

Because pairs only span a `MAX_GAP = 4` window, the true step order for a whole tutorial can be
reconstructed from the forward (`label=1`) pairs. Example, from
[`journey-dry-run`](https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-dry-run)
(`tier=9`, `sub=5c`):

```
0. "You'll learn how to activate a journey in dry run mode, configure options like
    disabling wait times and external data sources, and interpret profile flow metrics
    to validate audience segmentation and conditional logic."
1. "In this demo, I'll provide a walkthrough of the Journey Dry Run feature in Adobe
    Journey Optimizer."
2. "Dry Run is a capability allowing marketeers to smoke test a journey without
    actually sending any messages to end users."
3. "It will be useful to check if the journey is working as expected before making
    it live."
4. "I will choose to not disable wait activities."
5. "I can for instance go to the last 24 hours report by clicking on its
    dedicating button."
6. "There, I can visualize journey metrics, validate that email and SMS metrics are
    empty as no communication are sent during Dry Run execution, and click on the
    export button to download this report for future reference."
```

### Raw example rows produced from this document

Forward pair (correct order, `label = 1`):
```json
{"text_1": "Dry Run is a capability allowing marketeers to smoke test a journey without actually sending any messages to end users.",
 "text_2": "I can for instance go to the last 24 hours report by clicking on its dedicating button.",
 "label": 1, "tier_1": 9, "tier_2": 9, "sub_1": "5c", "sub_2": "5c",
 "url_1": "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-dry-run",
 "url_2": "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-dry-run"}
```

The **same two sentences, swapped**, emitted as the antisymmetric counterpart (`label = 0`):
```json
{"text_1": "I can for instance go to the last 24 hours report by clicking on its dedicating button.",
 "text_2": "Dry Run is a capability allowing marketeers to smoke test a journey without actually sending any messages to end users.",
 "label": 0, "tier_1": 9, "tier_2": 9, "sub_1": "5c", "sub_2": "5c",
 "url_1": "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-dry-run",
 "url_2": "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-dry-run"}
```

Another example from a different tutorial (`tier`/`sub` unresolved → `null`, forward pair):
```json
{"text_1": "Now we'll create a feed, and we'll select a segment that specifically uses that dataset to demonstrate the data governance capabilities.",
 "text_2": "And then we'll go ahead and select the Healthy Brews Loyalty Gold Member segment.",
 "label": 1, "tier_1": null, "tier_2": null, "sub_1": null, "sub_2": null,
 "url_1": "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/audiences/segment-match/segment-match-data-governance",
 "url_2": "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/audiences/segment-match/segment-match-data-governance"}
```

