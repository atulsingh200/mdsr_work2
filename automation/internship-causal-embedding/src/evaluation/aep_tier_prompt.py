"""LLM zero-shot prediction prompt for the `aep_gapdecay_notier11` directional task.

Target dataset: `data/aep_gapdecay_notier11/directional_*.jsonl`. Each row is
`{text_1, text_2, label, tier_1, tier_2, sub_1, sub_2, url_1, url_2}`. Like the
AJO tier dataset (`ajo_tier_prompt.py`), this is NOT literal step order within
one document -- `text_1`/`text_2` are short (1-2 sentence) snippets pulled from
DIFFERENT Adobe Experience Platform (AEP) documentation pages, and `label` is
decided purely by which snippet's sub-chapter sits in the lower-ranked tier of
the 10-tier hierarchy below (`label=1` <=> tier_1 <= tier_2 in rank; ties are
pre-filtered out by the builder). See `directional_manifest.json`'s
`tier_rank`/`subchapter_to_tier` and `aep_assignments.json`'s `tiers`/
`subchapters` (with real `title`/`rationale` fields) for the ground truth this
was built from.

The 10 tiers kept by the gap-decay builder (tier 11 "Advanced Applications" is
dropped -- see `directional_manifest.json`'s `drop_tiers`/`tiers_kept`),
foundational -> advanced, exactly as titled in `aep_assignments.json`:

  T1  Platform Foundation           -- core infrastructure & access controls that enable everything else
  T2  Data Architecture             -- schemas and data modeling that define structure for all data
  T3  Data Collection               -- getting data into the platform using the defined schemas
  T4  Data Storage & Catalog        -- managing and organizing collected data within the platform
  T5  Identity Foundation           -- core identity resolution that enables unified customer profiles
  T6  Profile & Segmentation        -- building unified profiles and audiences from identified data
  T7  Data Processing & Analytics   -- advanced analysis and ML capabilities that operate on profiles and data
  T8  Activation & Destinations     -- sending processed data and audiences to external systems
  T9  Governance & Privacy          -- controls and compliance that govern all data operations
  T10 Monitoring & Operations       -- observing and managing the running platform

This is real domain/curriculum knowledge (an AEP architect's mental model of how
the platform's capabilities build on each other), not an artificial decision
checklist -- the label genuinely IS this order, so telling the model the order
is giving it the actual rule the data was generated from. What's deliberately
NOT given: the tier NUMBER of either snippet (that would just be the answer),
and no per-pair mechanical checklist -- the model still has to (a) figure out
what each snippet is actually about, and (b) place that topic in the hierarchy
itself, which is real reasoning, not lookup.

Only `text_1`/`text_2` are exposed (no url/tier/sub metadata), matching the
cross-encoder's own input at inference, so an LLM-baseline accuracy computed
with this prompt is comparable to a trained model's test metrics.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are an expert in Adobe Experience Platform (AEP) who has taught this \
product to many new users, and knows how its capabilities build on each other. \
Roughly, in the order someone learns it:

  platform foundation & access control -> data architecture (schemas) -> data \
collection (ingestion) -> data storage & catalog -> identity foundation -> \
profile & segmentation -> data processing & analytics -> activation & \
destinations -> governance & privacy -> monitoring & operations

This is the real architecture AEP is taught in -- e.g. you define a schema \
before you can collect data into it, and you resolve identities before you can \
build a unified profile from them, and you build an audience/segment before you \
can activate it to a destination. Use it as your compass, not as a table to \
look values up in.

You are given two short passages, Passage A and Passage B, each pulled from a \
different AEP documentation page (how-to guides, reference/glossary \
definitions, or tutorial transcripts). They are often just one or two \
sentences, sometimes vague, dense with jargon, or referring to things ("this", \
"it") from earlier in their own source that you can't see.

Read both passages and work out, as best you can, what AEP capability or topic \
each one is really about. Then decide which of those two topics a learner or \
implementer would naturally rely on or set up earlier.

If a passage is too vague or generic to tell what topic it's really about, or \
the two topics genuinely don't depend on each other, say so honestly in your \
reasoning -- then still make your best call for the verdict.

The order the passages are shown to you (A first, then B) carries NO information \
about which is earlier.

Respond in EXACTLY this format, with nothing before or after it:
<a few sentences of honest reasoning: what each passage seems to be about, and \
which comes earlier in the platform's architecture, and why>
VERDICT: BEFORE      (Passage A's topic comes earlier)
VERDICT: NOT_BEFORE  (Passage B's topic comes earlier, or they're independent)\
"""

USER_PROMPT = """\
Passage A: {text_1}
Passage B: {text_2}

Which passage's topic would a learner or implementer rely on earlier in Adobe \
Experience Platform, and why? Reason about it, then end with exactly one \
VERDICT line.\
"""

_VERDICT_RE = re.compile(r"VERDICT:\s*(BEFORE|NOT_BEFORE)", re.IGNORECASE)


def build_user_prompt(text_1: str, text_2: str) -> str:
    return USER_PROMPT.format(text_1=text_1, text_2=text_2)


def parse_verdict(response_text: str) -> int | None:
    """Map the model's VERDICT line to the dataset's label convention (1/0).

    Returns None if no VERDICT line was found (caller should treat as a parse
    failure, not silently default to a label).
    """
    m = _VERDICT_RE.search(response_text)
    if not m:
        return None
    return 1 if m.group(1).upper() == "BEFORE" else 0


if __name__ == "__main__":
    # Dry-run smoke test: fill the templates with a couple of real rows and
    # print them, no API call.
    import json
    import sys
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[2] / (
        "data/aep_gapdecay_notier11/directional_test.jsonl"
    )
    rows = [json.loads(l) for l in data_path.open()][:4]
    print(f"=== SYSTEM_PROMPT ===\n{SYSTEM_PROMPT}\n", file=sys.stderr)
    for r in rows:
        print("=== USER_PROMPT ===", file=sys.stderr)
        print(build_user_prompt(r["text_1"], r["text_2"]), file=sys.stderr)
        print(
            f"(ground-truth label: {r['label']}, tier_1={r['tier_1']}, tier_2={r['tier_2']})\n",
            file=sys.stderr,
        )
