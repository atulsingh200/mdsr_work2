#!/usr/bin/env python3
"""
Procedural Workflow Extractor for AEP doc_dataset.xlsx

Strategy:
  - For each document, scan for numbered sequences starting at 1 and going 1,2,3,...,N.
  - Each contiguous numbered sequence (1→N) is one workflow.
  - A document may produce multiple workflows (one per numbered sequence block).
  - Multi-line items are joined: lines are consumed until the next numbered item or
    a blank line that is followed by a non-continuation line.
  - Light cleaning: strip markdown bold/links, code ticks, control words.
  - Filter: sequences with < 3 steps are discarded (not a real workflow).
  - Nested sub-steps (indented 1. 2. 3. inside a parent) are merged into the
    parent step text rather than treated as separate workflows.

Output: all_procedural_workflows.json
  List of {id, url, title, num_steps, steps: [step1_text, step2_text, ...]}
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_STEPS = 3          # discard sequences shorter than this
MAX_STEPS = 50         # sanity cap

# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
BOLD_RE   = re.compile(r"\*{1,3}(.+?)\*{1,3}")
LINK_RE   = re.compile(r"\[([^\]]+)\]\([^)]*\)")
BADGE_RE  = re.compile(r"\[([^\]]*)\]\{[^}]*\}")
TICK_RE   = re.compile(r"`[^`]*`")        # inline code ticks -> keep text inside
WS_RE     = re.compile(r"\s+")
IMG_RE    = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # images
# metadata / footer lines
FOOTER_RE = re.compile(
    r"^(recommendation-more-help|last update|created for|topics:|"
    r"\* topics|\* applies to|applies to:|note tip|note$|warning$|"
    r"important$|caution$|tip$|---)",
    re.I
)
CALLOUT_RE = re.compile(r"^\s*(NOTE|WARNING|IMPORTANT|CAUTION|TIP)\s*$", re.I)
MEDIA_RE   = re.compile(r"media_[a-f0-9]+\.", re.I)   # image file tokens
URL_RE     = re.compile(r"https?://\S+")


def clean_text(t: str) -> str:
    """Clean a single step text."""
    t = IMG_RE.sub("", t)
    t = LINK_RE.sub(r"\1", t)
    t = BADGE_RE.sub(r"\1", t)
    t = BOLD_RE.sub(r"\1", t)
    t = TICK_RE.sub(lambda m: m.group(0)[1:-1], t)   # strip ticks, keep content
    t = URL_RE.sub("", t)
    t = MEDIA_RE.sub("", t)
    t = t.replace("&gt;", ">").replace("&amp;", "&").replace("&lt;", "<")
    t = t.replace("*", "").replace("`", "").replace("_", " ")
    t = t.replace("  ", " ")
    t = WS_RE.sub(" ", t).strip(" .;:,")
    return t


def is_junk_line(line: str) -> bool:
    """True if the line should be ignored during step text collection."""
    stripped = line.strip()
    if not stripped:
        return False   # blank handled separately
    if FOOTER_RE.match(stripped):
        return True
    if CALLOUT_RE.match(stripped):
        return True
    if MEDIA_RE.search(stripped):
        return True
    if stripped.startswith("!["):
        return True
    return False


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------
# Matches a numbered list item: optional leading spaces (0-8), then N. text
# N can be 1-2 digits. Captures (indent_spaces, number, text).
STEP_LINE_RE = re.compile(r"^( {0,8})(\d{1,2})\.\s+(.+)$")

# Sub-step: indented more than the parent (>= 9 spaces or a tab-level deeper)
# We handle by checking relative indentation vs the current sequence's indent.

def _base_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def extract_workflows_from_doc(content: str, url: str, title: str):
    """
    Return a list of workflow dicts extracted from one document.
    Each workflow: {url, title, steps: [str, ...]}
    """
    lines = content.splitlines()

    # -------------------------------------------------------------------
    # Step 1: pre-clean lines (drop footer, media tokens, blank headings)
    # -------------------------------------------------------------------
    cleaned_lines = []
    stop = False
    for line in lines:
        if stop:
            break
        if "recommendation-more-help" in line.lower():
            stop = True
            continue
        if "video.tv.adobe.com" in line.lower():
            continue
        # strip trailing whitespace / trailing markdown artifacts
        cleaned_lines.append(line.rstrip())

    lines = cleaned_lines

    # -------------------------------------------------------------------
    # Step 2: collect numbered sequences
    # -------------------------------------------------------------------
    # We scan line by line. When we see "  1. <text>" we start a new sequence.
    # When we see "  2. <text>" right after (possibly with blank/non-step lines
    # in between that are continuation text), we append to it.
    # A sequence ends when:
    #   - A new "1. " is encountered (new sequence begins)
    #   - A heading (## / ###) is encountered that is not followed by steps
    #   - End of document

    workflows = []
    cur_steps: list[str] = []       # accumulated steps for current sequence
    cur_indent: int = -1             # indentation level of the sequence
    cur_step_text: str = ""          # text being built for current step

    def finish_step():
        nonlocal cur_step_text
        t = clean_text(cur_step_text)
        if t:
            cur_steps.append(t)
        cur_step_text = ""

    def finish_sequence():
        nonlocal cur_steps, cur_indent, cur_step_text
        finish_step()
        if len(cur_steps) >= MIN_STEPS:
            workflows.append(list(cur_steps))
        cur_steps = []
        cur_indent = -1
        cur_step_text = ""

    in_sequence = False
    expected_next = 1     # next step number we expect

    i = 0
    while i < len(lines):
        line = lines[i]
        m = STEP_LINE_RE.match(line)

        if m:
            indent   = len(m.group(1))
            num      = int(m.group(2))
            text     = m.group(3).strip()

            # ----- Detect sub-steps: indented deeper than current sequence -----
            if in_sequence and num < expected_next and indent > cur_indent:
                # This is a sub-step inside the current step — merge into step text
                if cur_step_text:
                    cur_step_text += " " + text
                i += 1
                continue

            # ----- num == 1: start a new sequence -----
            if num == 1:
                if in_sequence:
                    finish_sequence()
                # Start new sequence
                in_sequence = True
                cur_indent = indent
                expected_next = 2
                cur_step_text = text
                i += 1
                continue

            # ----- num == expected_next: continuing current sequence -----
            if in_sequence and num == expected_next and abs(indent - cur_indent) <= 3:
                finish_step()
                cur_step_text = text
                expected_next = num + 1
                i += 1
                continue

            # ----- num > expected_next: skip gap (e.g., jumped from 3→5) -----
            # Treat as continuation sub-numbering or reset
            if in_sequence and num > expected_next:
                # Could be a sub-list that restarts — just add to current step
                if cur_step_text:
                    cur_step_text += " " + text
                i += 1
                continue

            # ----- otherwise: doesn't fit -> finish current and ignore this line -----
            if in_sequence:
                finish_sequence()
                in_sequence = False
            i += 1
            continue

        else:
            # Non-step line
            stripped = line.strip()

            # Heading -> ends current sequence
            if stripped.startswith("#"):
                if in_sequence:
                    finish_sequence()
                    in_sequence = False
                i += 1
                continue

            # Blank line -> might be spacing between steps in same sequence
            if not stripped:
                # Look ahead: is next non-blank line a continuation of this step
                # or the next numbered item?
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                # Don't break on blank lines — they're just markdown spacing
                i += 1
                continue

            # Junk line
            if is_junk_line(line):
                i += 1
                continue

            # Regular text — if in a sequence, it's continuation of current step
            if in_sequence and cur_step_text:
                cur_step_text += " " + stripped
            i += 1

    # flush
    if in_sequence:
        finish_sequence()

    # Build result dicts
    result = []
    for steps in workflows:
        if MIN_STEPS <= len(steps) <= MAX_STEPS:
            result.append({
                "url":       url,
                "title":     title,
                "num_steps": len(steps),
                "steps":     steps,
            })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SKIP_URL_RE = re.compile(
    r"release-notes|/api/|business\.adobe\.com|experienceleaguecommunities|"
    r"changelog",
    re.I,
)


def main():
    here = Path(__file__).parent
    xlsx = here / "doc_dataset.xlsx"
    out  = here / "all_procedural_workflows.json"

    print(f"Loading {xlsx} ...", file=sys.stderr)
    df = pd.read_excel(xlsx)
    print(f"Loaded {len(df)} rows.", file=sys.stderr)

    all_workflows = []
    doc_count = 0
    wf_id = 1

    for _, row in df.iterrows():
        url     = str(row.get("url",     "") or "")
        title   = str(row.get("title",   "") or "")
        content = str(row.get("content", "") or "")

        if SKIP_URL_RE.search(url):
            continue
        # Skip very short / video-only stubs
        if len(content) < 500:
            continue

        doc_count += 1
        wfs = extract_workflows_from_doc(content, url, title)
        for wf in wfs:
            wf["id"] = wf_id
            wf_id += 1
        all_workflows.extend(wfs)

    print(f"Processed {doc_count} docs.", file=sys.stderr)
    print(f"Extracted {len(all_workflows)} workflows total.", file=sys.stderr)

    # Step-count distribution
    from collections import Counter
    dist = Counter(wf["num_steps"] for wf in all_workflows)
    print("Step-count distribution:", dict(sorted(dist.items())), file=sys.stderr)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_workflows, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(all_workflows)} workflows to {out}", file=sys.stderr)

    # Print a sample
    print("\n--- Sample workflow ---", file=sys.stderr)
    if all_workflows:
        sample = next(
            (w for w in all_workflows
             if "implement-in-websites/implement-solutions/analytics" in w["url"]),
            all_workflows[0]
        )
        print(f"URL: {sample['url']}", file=sys.stderr)
        print(f"Steps ({sample['num_steps']}):", file=sys.stderr)
        for k, s in enumerate(sample["steps"], 1):
            print(f"  {k}. {s[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()
