# `new_aep_workflow_scrap` — Phase-Level Execution-Order Pair Dataset

This directory contains `directional_train.jsonl`, `directional_val.jsonl`, and
`directional_test.jsonl`: pairs of **phase-level** procedural text extracted from
scraped Adobe Experience Platform (AEP) documentation pages, labeled with which phase
happens **first** in the underlying workflow.

**Builder script:** [`extract_workflows.py`](extract_workflows.py)
Verified against the actual data files — confirmed by:
- file timestamps: `extract_workflows.py` was last modified 2026-06-25 13:32, and
  `workflow_phases.json` + all three `directional_*.jsonl` files were written in the
  same run, at the same second (2026-06-27 03:20:45) — exactly the outputs this script
  declares it writes.
- schema match: every row has exactly the fields the script's docstring promises
  (`text_1, text_2, label, step_1, step_2, url, title`).
- logic match: phase word-count band (24–56 words), gap distribution (`{1, 2}` only,
  from `MAX_GAP=2`), document-level split with zero doc/pair overlap across splits, and
  the "11 workflows produce zero surviving rows" behavior (see §5) are all exactly
  reproduced by re-deriving them from `workflow_phases.json` and the pair files.

> **Note — don't confuse with other files in this directory.** This folder also
> contains a *separate, newer* experimental pipeline (`extract_procedural_workflows.py`
> → `all_procedural_workflows.json` → `build_directional_procedural.py` →
> `directional_procedural/directional_*.jsonl`, all dated 2026-07-01) and a
> one-off `workflow_pairs.jsonl` (dated 2026-06-25, no direct builder found). Those are
> unrelated to the three `directional_*.jsonl` files this README documents.

---

## 1. What problem this dataset is for

Unlike [`new_ajo_workflows`](../new_ajo_workflows/README.md), which pairs individual
**sentences** from video tutorial transcripts, this dataset pairs whole **phases** —
~24–56 word paragraphs, each describing one stage of a procedure (e.g. "Configure the
merge policy: ..."), extracted from AEP's written how-to documentation
(`doc_dataset.xlsx`, a scrape of doc pages, not videos). The target granularity was
chosen to match two downstream eval sets referenced in the script's docstring:
`ajo_orchestrated_workflows_flat.json` and `test_samples_milan.json`, both of which
compare paragraph/phase-sized workflow steps rather than single sentences.

## 2. Step-by-step: how the dataset is built

1. **Load the raw scrape.** `doc_dataset.xlsx` is read with pandas; each row has
   `url`, `title`, `content` (markdown-ish scraped page body).

2. **Gate out non-procedural / irrelevant URLs.** `SKIP_URL` drops release notes,
   `/api/` reference pages, `business.adobe.com`, community forum pages, XDM/schema
   reference pages, and changelogs — none of these describe a linear workflow.

3. **Drop stub video pages.** If a page's content is short (< 1500 chars) and only
   embeds a video (`video.tv.adobe.com`) with no real text structure, it's skipped —
   there's no extractable text procedure.

4. **Document-level pre-cleaning (`preclean_doc`).** Line by line:
   - strips fenced code blocks entirely,
   - strips the anchor-slug suffix Adobe's doc generator appends to every heading
     (e.g. `## Configure the connection configure-the-connection` → `## Configure the
     connection`), and demotes H4–H6 headings to plain text so their slugs don't leak
     into prose,
   - drops breadcrumb/metadata lines (`Documentation`, `Last update`, `* Topics:`,
     `Applies to:`),
   - truncates the whole document at the `recommendation-more-help` footer marker —
     everything after that is boilerplate "related articles" junk,
   - drops code lines, markdown table rows, and lines that are just a video embed or
     asset filename.

5. **Split into H2/H3 sections.** `sections()` splits the cleaned doc on `##`/`###`
   headings into `(heading, body)` pairs, additionally repairing the case where an
   anchor slug wraps onto the next line and would otherwise leak into the body text.
   Sections are further filtered by `BOILER_HEAD` (drops non-procedural headings like
   *Prerequisites, Overview, Next steps, See also, FAQ, Troubleshoot, Glossary, ...*)
   and any heading containing a noise token or a `?` (Q&A headings).
   If a doc has no usable H2/H3 structure but does have numbered steps, `_h1_fallback_sections`
   treats the whole H1 body as one big pseudo-section so its steps aren't lost.

6. **Extract ordered phases per document — 3-strategy router (`extract_doc`).**
   Different doc layouts need different handling, tried in this order:
   - **S1 — "fat numbered items" (quick-start guides).** If a section's body has
     numbered list items (`1. ...`, `2. ...`) whose first sentence alone is already
     ≥ 10 words, each numbered item becomes one phase (`phase_from_item`): take the
     item's first sentence, and if it's under the 24-word floor, keep appending
     following sentences from the same item until the floor is met (or sentences run
     out), then clamp/reject by the 24–56 word band and require an action verb match
     (`_ok_phrase`). If this strategy alone yields ≥ 3 phases for the doc, its output
     is used directly (capped at 6 phases) and S2/S3 are skipped for that doc.
   - **S2/S3 — one phase per section (concept/reference-style docs).** Otherwise,
     each surviving section is collapsed into a single phase (`phase_from_clauses`):
     start from `"<Heading>: "`, then append clause-by-clause (either the first
     sentence of each numbered item in the section, or up to 5 sentences of plain
     body text) until the 56-word ceiling would be exceeded. This produces one
     "phase" per logical section rather than per numbered item.
   - Both strategies funnel through the same acceptance gate, `_ok_phrase`: reject if
     it matches `NOISE_TOKEN` (leftover code/API/table/markdown residue) or its word
     count falls outside 24–56, and require at least one action verb from `ACTION_RE`
     (`create, configure, select, set up, define, add, build, send, activate,
     publish, upload, ingest, map, enable, navigate, log in, access, update,
     execute, populate, verify, review, monitor, connect, install, design, author,
     choose, deliver, target, schedule, apply, drag, enter, open, click, integrate,
     generate, deploy, launch, test`).
   - All inline cleanup (`clean()`) strips markdown links/bold, "Learn more.../For
     more information..." boilerplate phrases, HTML entities, stray backticks/code
     ticks, `IMPORTANT/NOTE/WARNING/CAUTION/TIP` callout labels, UUID-like tokens, and
     4+ segment hyphenated slug/code-id residue (e.g. `trigger-your-marketo-engage-email`)
     while allowlisting genuine hyphenated English (`out-of-the-box`, `state-of-the-art`).

7. **Dedupe near-identical phases within a doc.** Phases whose first 45 characters
   (lowercased) match an already-kept phase are dropped, to avoid the same phase being
   captured twice from overlapping section/heading extraction.

8. **Gate on workflow (phase-count) size.** A document is kept as a "workflow" only if
   its final phase count is between `MIN_PHASES=3` and `MAX_PHASES=12` — too few
   phases isn't a real sequence, too many usually means the extractor over-segmented
   a reference page rather than found a genuine procedure.

9. **Write `workflow_phases.json`** — one record per kept workflow: `{id, url, title,
   t1, t2, ..., tN}`, `N` phases in original document order. **1301 workflows** total
   as of the current build.

10. **Document-level train/val/test split (leakage-safe).** Workflow indices are
    shuffled with a fixed seed (`random.Random(42)`) and split 80/10/10 by document
    (`int(0.1*n)` each for test/val, remainder train) — identical splitting strategy
    to `new_ajo_workflows`, for the same leakage-avoidance reason.

11. **Build candidate phase pairs within a bounded-gap window.** For each workflow's
    ordered phase list, every pair `(phases[i], phases[j])` with `i < j <= i + MAX_GAP`
    (`MAX_GAP = 2`, i.e. consecutive phases and "skip-one" phases) is generated —
    tighter than `new_ajo_workflows`'s window of 4, since phases are already
    coarse-grained stage summaries rather than single sentences, so even a 2-phase
    gap already spans a meaningful chunk of the procedure.

12. **Emit both directions (antisymmetric supervision), unconditionally.** For every
    candidate pair `(a, b)` (`a` occurs earlier in the doc), **both** rows are always
    written:
    - `{text_1: a, text_2: b, label: 1, step_1: "t{i+1}", step_2: "t{j+1}"}` — forward
    - `{text_1: b, text_2: a, label: 0, step_1: "t{j+1}", step_2: "t{i+1}"}` — reverse

    Same rationale as `new_ajo_workflows`: this is what makes the dataset an
    antisymmetric comparator rather than a topic/register classifier, and is why the
    label balance is exactly 50/50 by construction. (No per-document pair cap is
    applied here — unlike `new_ajo_workflows`'s `PER_DOC_CAP=60` — because with only
    3–12 phases per doc and a gap of 2, the natural pair count per doc is already
    small: at most `2*12 - 3 = 21` unordered pairs → 42 rows.)

13. **Global cross-split dedup on `(text_1, text_2, label)`.** Because many AEP doc
    pages share near-identical boilerplate phases (e.g. destination "Supported
    identities" tables, generic "Review" steps), the exact same pair can be produced
    by more than one document. A single global `seen` set is built by iterating
    **train first, then val, then test**, keeping only the first occurrence of each
    `(text_1, text_2, label)` key and dropping later duplicates. Iterating train first
    guarantees any such collision is resolved in favor of train, so **no eval (val/test)
    pair is ever also present in train** — a second, pair-level leakage guard on top
    of the doc-level split. This is also why **11 of the 1301 workflows in
    `workflow_phases.json` contribute zero rows** to the final files (verified): every
    pair those 11 docs produced was already an exact duplicate of a pair from another
    (earlier-processed) document.

14. **Shuffle and write out**, then log doc/row counts and train label balance to
    stderr for a build-time sanity check.

## 3. Row schema

| field    | meaning                                                                 |
|----------|--------------------------------------------------------------------------|
| `text_1` | first phase in the pair (~24–56 words)                                  |
| `text_2` | second phase in the pair                                                |
| `label`  | `1` if `text_1`'s phase occurs **before** `text_2`'s in the workflow, else `0` |
| `step_1` | phase id of `text_1` within its source workflow, e.g. `"t2"`           |
| `step_2` | phase id of `text_2`, e.g. `"t3"`                                       |
| `url`    | source documentation page (single URL — both phases always come from the same doc) |
| `title`  | title of the source documentation page                                  |

Note the field names differ slightly from `new_ajo_workflows` (`url`/`title` instead
of `url_1`/`url_2`, and `step_1`/`step_2` instead of `tier_1`/`sub_1`/etc.) — this
dataset was built independently and does not carry tier/subchapter classification
metadata.

## 4. Verified dataset statistics

| split | rows   | unique docs | label balance     |
|-------|--------|-------------|--------------------|
| train | 13,904 | 1,033       | 6,952 / 6,952 (1/0) |
| val   | 1,676  | 130         | 838 / 838          |
| test  | 1,548  | 127         | 774 / 774          |

- **1301 total workflows extracted**, with a phase-count distribution of: 3 phases ×418,
  4×255, 5×177, 6×179, 7×70, 8×61, 9×54, 10×56, 11×20, 12×11 docs.
- **No document overlap across splits** (0 shared URLs) and **no `(text_1, text_2,
  label)` pair overlap across splits** — confirms both the doc-level split (step 10)
  and the train-first global dedup (step 13).
- **Exact 50/50 label balance in every split** — confirms every retained pair is
  emitted in both directions (step 12).
- **Gap distribution in train: gap=1 → 7,852 rows, gap=2 → 6,052 rows** — confirms
  `MAX_GAP=2` (consecutive + skip-one only, no larger gaps).
- `text_1` word count is always in `[24, 56]` (mean ≈ 42.1), confirming the `LO, HI`
  phase-length band.
- **1,290 of the 1,301 workflows** in `workflow_phases.json` are represented in the
  final pair files; the other 11 were fully absorbed by the global dedup (step 13),
  verified directly (e.g. `journey-global-report-cja-web`, `url-parameter-encryption`,
  `buying-group-marketing` all vanish this way).

## 5. Sample data — a real 5-phase workflow

From [`profile/merge-policies/ui-guide`](https://experienceleague.adobe.com/en/docs/experience-platform/profile/merge-policies/ui-guide)
("Merge Policies UI Guide"), extracted via strategy S2/S3 (one phase per section):

```
t1. "Create a merge policy: To create a new merge policy, select Create merge
     policy on the merge policies tab to enter the new merge policy workflow,
     the New merge policy workflow requires you to provide important information
     for your new merge policy through a series of guided steps."
t2. "Configure: The first step in the workflow allows you to configure your
     merge policy by providing basic information, this information includes:
     Name: The name of your merge policy should be descriptive yet concise,
     schema class: The XDM schema class associated with the merge policy."
t3. "Select Profile datasets: On the Select Profile datasets screen, you must
     select the Merge method that you wish to use for your merge policy, also
     displayed on the screen is the total number of Profile datasets in your
     organization that relate to the schema class that was selected on the
     previous screen."
t4. "Select ExperienceEvent datasets: The next step in the workflow requires
     you to select ExperienceEvent datasets, this screen is influenced by the
     merge method that you selected on the Select Profile datasets screen."
t5. "Review: The final step in the workflow is to review your merge policy,
     the Review screen displays information about your merge policy, including
     the ID stitching method selected, merge method selected, and the datasets
     included."
```

With 5 phases and `MAX_GAP=2`, the gap-bounded pairs are `(1,2) (1,3) (2,3) (2,4)
(3,4) (3,5) (4,5)` — 7 unordered pairs × 2 directions = **14 rows**, all of which
landed in `train` for this doc (verified — every row for this URL is present in
`directional_train.jsonl` and none in val/test, consistent with the doc-level split).

### Raw example rows from this workflow

Forward pair, gap = 1 (`label = 1`):
```json
{"text_1": "Create a merge policy: To create a new merge policy, select Create merge policy on the merge policies tab to enter the new merge policy workflow, the New merge policy workflow, requires you to provide important information for your new merge policy through a series of guided steps.",
 "text_2": "Configure configure: The first step in the workflow allows you to configure your merge policy by providing basic information, this information includes: Name : The name of your merge policy should be descriptive yet concise, schema class : The XDM schema class associated with the merge policy.",
 "label": 1, "step_1": "t1", "step_2": "t2",
 "url": "https://experienceleague.adobe.com/en/docs/experience-platform/profile/merge-policies/ui-guide",
 "title": "Merge Policies UI Guide"}
```

The same two phases, swapped — the antisymmetric counterpart (`label = 0`):
```json
{"text_1": "Configure configure: The first step in the workflow allows you to configure your merge policy by providing basic information, this information includes: Name : The name of your merge policy should be descriptive yet concise, schema class : The XDM schema class associated with the merge policy.",
 "text_2": "Create a merge policy: To create a new merge policy, select Create merge policy on the merge policies tab to enter the new merge policy workflow, the New merge policy workflow, requires you to provide important information for your new merge policy through a series of guided steps.",
 "label": 0, "step_1": "t2", "step_2": "t1",
 "url": "https://experienceleague.adobe.com/en/docs/experience-platform/profile/merge-policies/ui-guide",
 "title": "Merge Policies UI Guide"}
```

Forward pair with gap = 2, skipping an intermediate phase (`label = 1`):
```json
{"text_1": "Configure configure: The first step in the workflow allows you to configure your merge policy by providing basic information, this information includes: Name : The name of your merge policy should be descriptive yet concise, schema class : The XDM schema class associated with the merge policy.",
 "text_2": "Select ExperienceEvent datasets: The next step in the workflow requires you to select ExperienceEvent datasets, this screen is influenced by the merge method that you selected on the Select Profile datasets screen.",
 "label": 1, "step_1": "t2", "step_2": "t4",
 "url": "https://experienceleague.adobe.com/en/docs/experience-platform/profile/merge-policies/ui-guide",
 "title": "Merge Policies UI Guide"}
```

## 6. Key design decisions to remember

- **Phase-level, not sentence-level.** Each `text_1`/`text_2` is a full ~24–56 word
  paragraph describing a whole workflow stage, sized to match downstream eval sets
  (see §1), not a single action sentence like `new_ajo_workflows`.
- **Multi-strategy extraction (S1 "fat numbered items" vs. S2/S3 "one phase per
  section")** — because the source is scraped written documentation with wildly
  inconsistent layouts (quick-start numbered guides vs. concept/reference pages with
  prose sections), a single fixed strategy would miss most real procedures.
- **Same-document only, bounded gap (`MAX_GAP=2`)** — narrower window than
  `new_ajo_workflows`'s `MAX_GAP=4` since phases are already coarse; a gap of 2 phases
  already covers most of a short workflow.
- **Antisymmetric supervision, always both directions** — identical rationale to
  `new_ajo_workflows`: forces a real pairwise comparator instead of a
  content/topic shortcut.
- **Two independent leakage guards**: document-level split (§10) *and* a train-first
  global `(text_1, text_2, label)` dedup (§13) that additionally prevents duplicate
  boilerplate phases (common across similar destination/connector doc pages) from
  crossing from train into val/test.
- **No per-document pair cap** (unlike `new_ajo_workflows`) — unnecessary here since
  the phase-count ceiling (`MAX_PHASES=12`) and narrow gap window already bound the
  pairs-per-doc naturally.
