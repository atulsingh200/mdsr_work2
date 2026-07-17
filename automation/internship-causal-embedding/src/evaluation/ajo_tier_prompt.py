"""LLM zero-shot prediction prompt for the `ajo_gapdecay_15tier` directional task.

Target dataset: `data/ajo_gapdecay_15tier/directional_*.jsonl`. Each row is
`{text_1, text_2, label, tier_1, tier_2, sub_1, sub_2, url_1, url_2}`. Unlike the
procedural_workflow task, this is NOT literal step order within one document --
`text_1`/`text_2` are short (often 1-2 sentence) snippets pulled from DIFFERENT
AJO video-tutorial transcripts, and `label` is decided purely by which snippet's
sub-chapter sits in the lower-ranked tier of the 13-tier curriculum below
(`label=1` <=> tier_1 <= tier_2 in rank; ties are pre-filtered out by the
builder). See `directional_manifest.json`'s `tier_rank`/`subchapter_to_tier` and
`build_classification_data.py`'s `SUBCHAPTER_TO_TIER` for the ground truth this
was built from.

The 13 tiers kept by the gap-decay builder (tiers 1 "Product orientation" and 15
"Use cases/labs/capstones" are dropped -- see `directional_manifest.json`'s
`drop_tiers`/`tiers_kept`), foundational -> advanced:

  T2  Admin floor
  T3  Data foundations
  T4  Profiles, audiences & subscriptions
  T5  Content authoring foundations
  T6  Channels & campaigns (first sends)
  T7  Journeys (core)
  T8  Personalization & decisioning
  T9  Advanced journey patterns & experimentation
  T10 Multi-journey orchestration
  T11 Admin configuration
  T12 Governance & privacy
  T13 Observability & reporting
  T14 AI agents & assistants

This is real domain/curriculum knowledge (an AJO instructor's mental syllabus),
not an artificial decision checklist -- the label genuinely IS this order, so
telling the model the order is giving it the actual rule the data was generated
from, the same way the procedural prompt gives a human SME real product
knowledge. What's deliberately NOT given: the tier NUMBER of either snippet (that
would just be the answer), and no per-pair mechanical checklist -- the model
still has to (a) figure out what each vague, pronoun-heavy transcript snippet is
actually about, and (b) place that topic on the syllabus itself, which is real
reasoning, not lookup.

Only `text_1`/`text_2` are exposed (no url/tier/sub metadata), matching the
cross-encoder's own input at inference, so an LLM-baseline accuracy computed
with this prompt is comparable to a trained model's test metrics.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are an expert in Adobe Journey Optimizer (AJO) who has taught this product \
to many new users, and knows the order in which someone naturally learns it. \
Roughly, in the order a learner would go:

  admin floor & permissions -> data foundations (schemas, datasets, identities) \
-> profiles, audiences & subscriptions -> content authoring foundations -> \
channels & campaigns (first sends) -> journeys (core) -> personalization & \
decisioning -> advanced journey patterns & experimentation -> multi-journey \
orchestration -> deeper admin configuration -> governance & privacy -> \
observability & reporting -> AI agents & assistants

This is the real syllabus AJO tutorials are taught in -- e.g. you build an \
audience before you can target a campaign with it, and you author content before \
you send it in a channel. Use it as your compass, not as a table to look values \
up in.

You are given two short passages, Passage A and Passage B, each pulled from a \
different AJO video-tutorial transcript. They are often just one or two \
sentences, sometimes vague or referring to things ("this", "it") from earlier in \
their own video that you can't see.

Read both passages and work out, as best you can, what AJO capability or topic \
each one is really about. Then decide which of those two topics a learner would \
naturally reach earlier on the syllabus above.

If a passage is too vague or generic to tell what topic it's really about, or the \
two topics genuinely don't depend on each other, say so honestly in your \
reasoning -- then still make your best call for the verdict.

The order the passages are shown to you (A first, then B) carries NO information \
about which is earlier.

Respond in EXACTLY this format, with nothing before or after it:
<a few sentences of honest reasoning: what each passage seems to be about, and \
which comes earlier on the syllabus, and why>
VERDICT: BEFORE      (Passage A's topic comes earlier)
VERDICT: NOT_BEFORE  (Passage B's topic comes earlier, or they're independent)\
"""

USER_PROMPT = """\
Passage A: {text_1}
Passage B: {text_2}

Which passage's topic would a learner reach earlier in Adobe Journey Optimizer, \
and why? Reason about it, then end with exactly one VERDICT line.\
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
        "data/ajo_gapdecay_15tier/directional_test.jsonl"
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
