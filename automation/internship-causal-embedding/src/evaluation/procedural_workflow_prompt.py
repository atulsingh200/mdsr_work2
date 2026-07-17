"""LLM zero-shot prediction prompt for the `procedural_workflow` directional task.

Target dataset: `data/new_aep_workflow_scrap/directional_procedural/directional_*.jsonl`
(same data the `procedural_workflow` cross-encoder run was trained/evaluated on —
see `runs/crossencoder2x_deberta/procedural_workflow/config.json`). Each row is
`{text_1, text_2, label, step_1, step_2, url, title}` where `label=1` means text_1
is the step that happens first in the source procedure, and `label=0` means the
pair is shown in reversed order.

This module intentionally exposes only `text_1`/`text_2` to the prompt (no
`title`/`url`/step ids) to match the cross-encoder's own input at inference
(`input_form: "[CLS] text_1 [SEP] text_2 [SEP]"`), so an LLM-baseline accuracy
computed with this prompt is a fair comparison against `test_metrics.json`.

Why the prompt looks the way it does (learned from inspecting the actual data,
not assumed):
- Pairs range from single short UI actions ("Select Save") to full paragraph-level
  phases (up to a few hundred words).
- A meaningful fraction of "steps" are not sequential actions at all but sibling
  items in an enumerated list or FAQ (e.g. two independent "If you want to do X,
  do Y" bullets for a deprecation policy).
- Presentation order of Step A / Step B is randomized in the source data (a
  50/50 coin flip decided which member of the pair became text_1), so it carries
  zero signal.

The prompt deliberately does NOT hand the model an explicit decision checklist
(no "check if A produces X that B needs", no enumerated fallback rules). A rigid
checklist gets applied mechanically to every pair and produces the same
templated justification each time -- which is just a different flavor of the
"mugging the pattern instead of learning the true reason" failure this prompt is
meant to avoid. Instead it asks the model to look at the two texts and reason
about them on their own terms, the way a person familiar with the product would.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are an expert in Adobe Experience Platform (AEP) and Adobe Journey Optimizer \
(AJO) product workflows.

You are given two steps, Step A and Step B, taken from the same how-to procedure \
in Adobe's official documentation. Read both carefully and think about what is \
actually happening in each one -- what the user is doing, and what state that \
leaves the product in.

Then decide: does one of these steps have to happen before the other for it to \
make sense, and if so, which one comes first? Explain your reasoning in your own \
words, based only on what the two texts describe -- not on which one is labeled A \
or which one is written first, since that carries no information here.

Respond in EXACTLY this format, with nothing before or after it:
<a few sentences of honest, concrete reasoning about these two specific steps>
VERDICT: BEFORE      (Step A must be completed before Step B)
VERDICT: NOT_BEFORE  (Step B comes first, or the two are independent)\
"""

USER_PROMPT = """\
Step A: {text_1}
Step B: {text_2}

Which step must happen first, and why? Reason about it, then end with exactly \
one VERDICT line.\
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
    # print them, no API call. Useful to sanity-check formatting before wiring
    # this into a batch-calling script once API credentials are available.
    import json
    import sys
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[2] / (
        "data/new_aep_workflow_scrap/directional_procedural/directional_test.jsonl"
    )
    rows = [json.loads(l) for l in data_path.open()][:3]
    print(f"=== SYSTEM_PROMPT ===\n{SYSTEM_PROMPT}\n", file=sys.stderr)
    for r in rows:
        print("=== USER_PROMPT ===", file=sys.stderr)
        print(build_user_prompt(r["text_1"], r["text_2"]), file=sys.stderr)
        print(f"(ground-truth label: {r['label']})\n", file=sys.stderr)
