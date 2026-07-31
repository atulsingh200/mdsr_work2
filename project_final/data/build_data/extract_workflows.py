#!/usr/bin/env python3
"""
Phase-level workflow extractor for AEP scraped docs (doc_dataset.xlsx).

Goal: produce text_1/text_2 ordered-step pairs where each step is a PHASE-level
paragraph (~24-56 words) describing a whole stage of a procedure, matching the
granularity of the eval targets:
  /mnt/localssd/ajo_orchestrated_workflows_flat.json  ({id,t1,t2,t3})
  /mnt/localssd/test_samples_milan.json               ({label1,label2,text1,text2})

Fully deterministic (no LLM). Multi-strategy: different doc layouts get different
handling. See /home/colligo/.claude/plans/keen-sprouting-thimble.md for the plan.

Output (in this directory):
  workflow_phases.json                      list of {id,url,title,t1,t2,...}
  directional_{train,val,test}.jsonl        {text_1,text_2,label,step_1,step_2,url,title}
"""
import json
import re
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Phase word-count band (target 24-56; small slack for clamping)
LO, HI = 24, 56
MIN_PHASES, MAX_PHASES = 3, 12   # workflows kept in this phase-count range
MAX_GAP = 2                      # pair phases up to 2 apart (consecutive + skip-one)

# ---------------------------------------------------------------------------
# Cleaning regexes
# ---------------------------------------------------------------------------
ANCHOR_RE = re.compile(r"\s+[a-z0-9]+(?:-[a-z0-9]+)+$")          # trailing slug anchor
BOLD_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
BADGE_RE = re.compile(r"\[([^\]]*)\]\{[^}]*\}")
LEARN_RE = re.compile(
    r"\b(Learn (more|how)[^.]*|For more information[^.]*|see [^.]*documentation)\.?", re.I)
WS_RE = re.compile(r"\s+")
# residue that must never appear inside a phase (final reject net)
NOISE_TOKEN = re.compile(
    r"video\.tv|recommendation-more-help|width=|https?://|code language|<script|"
    r"\{\{|alloy\(|^\s*[\{\[]|`|\*\*|\| |---|"
    r"\b(GET|POST|PUT|DELETE|PATCH)\s+/|API format|accordion|\d+-row-\d+|"
    r"\d+-align-\w+", re.I)

# uppercase callout words that leak from note/important blocks
CALLOUT_WORD = re.compile(r"\b(IMPORTANT|NOTE|WARNING|CAUTION|TIP)\b")

# 4+word hyphen tokens are almost always anchor slugs / code ids (e.g.
# "trigger-your-marketo-engage-email", "data-aep-click-token"); strip unless
# they are genuine hyphenated English (allowlist).
LONG_HYPHEN_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){3,}\b")
REAL_HYPHEN = {"out-of-the-box", "state-of-the-art"}


def _drop_long_hyphen(t: str) -> str:
    return LONG_HYPHEN_RE.sub(lambda m: m.group(0) if m.group(0).lower() in REAL_HYPHEN else "", t)

# code / table line detectors (line-level stripping during cleaning)
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
CODE_LINE_RE = re.compile(r"(code language|<script|</|\{\{|alloy\(|=\s*function|var\s+\w+\s*=)", re.I)
TABLE_LINE_RE = re.compile(r"^\s*\|?.*\|.*\|")  # >=2 pipes -> table row


def clean(t: str) -> str:
    """Inline-clean a fragment of text for use inside a phase."""
    t = LINK_RE.sub(r"\1", t)
    t = BADGE_RE.sub(r"\1", t)
    t = BOLD_RE.sub(r"\1", t)
    t = LEARN_RE.sub("", t)
    t = t.replace("&gt;", ">").replace("&amp;", "&").replace("&lt;", "<")
    t = t.replace("*", "").replace("`", "")          # stray markdown / code ticks
    t = CALLOUT_WORD.sub("", t)                        # IMPORTANT/NOTE/... leak
    t = UUID_RE.sub("", t)                             # example UUIDs / xxxx tokens
    t = _drop_long_hyphen(t)                           # anchor-slug / code-id residue
    t = t.replace("---", " ")
    t = WS_RE.sub(" ", t).strip(" .;,:")
    return t


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HEAD_ANCHOR_RE = re.compile(r"\s+[a-z0-9&]+(?:-[a-z0-9&]+)+-?$")  # trailing slug echo
UUID_RE = re.compile(r"\b[0-9a-fx]{8}-[0-9a-fx]{4}-[0-9a-fx]{4}-[0-9a-fx]{4}-[0-9a-fx]{12}\b", re.I)


def _strip_heading_anchor(line: str) -> str:
    """Strip the trailing anchor slug on any heading; demote H4-H6 to plain text
    so their slugs don't leak into prose (sections() only splits H2/H3)."""
    m = HEADING_RE.match(line)
    if not m:
        return line
    level, text = m.group(1), HEAD_ANCHOR_RE.sub("", m.group(2)).strip()
    return text if len(level) >= 4 else f"{level} {text}"


def preclean_doc(content: str) -> str:
    """Document-level cleaning: drop breadcrumbs, metadata, code, tables, footers."""
    content = CODE_FENCE_RE.sub(" ", content)
    out_lines = []
    for line in content.splitlines():
        l = _strip_heading_anchor(line.rstrip())
        if not l.strip():
            out_lines.append("")
            continue
        low = l.lower()
        if low.startswith("documentation") and "#" not in l:
            continue
        if low.startswith(("last update", "* topics", "topics:", "created for",
                           "* applies to", "applies to:", "* topics")):
            continue
        if "recommendation-more-help" in low:
            break  # footer -> rest of doc is junk
        if CODE_LINE_RE.search(l):
            continue
        if TABLE_LINE_RE.match(l):
            continue
        if "video.tv.adobe.com" in low or low.endswith((".json", ".mp4")):
            continue
        out_lines.append(l)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
SKIP_URL = re.compile(
    r"release-notes|/api/|business\.adobe\.com|experienceleaguecommunities|"
    r"/xdm/|/schemas?/|changelog", re.I)

BOILER_HEAD = re.compile(
    r"^(getting started|prerequisites?|overview|next steps?|see also|"
    r"additional resources|related|faq|frequently asked|video|transcript|"
    r"terminology|glossary|about|introduction|what is|limitations?|troubleshoot|"
    r"supported identit|export type|supported audiences|environment|error codes?)\b", re.I)

ACTION_RE = re.compile(
    r"\b(create|configure|select|set up|setup|set|define|add|build|send|activate|"
    r"publish|upload|ingest|map|enable|navigate|log ?in|access|update|execute|"
    r"populate|verify|review|monitor|connect|install|design|author|choose|deliver|"
    r"target|schedule|apply|drag|enter|open|click|integrate|generate|deploy|"
    r"launch|test)\b", re.I)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def sents(t: str):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def sections(content: str):
    """Split cleaned content into (heading, body) pairs on H2/H3."""
    parts = re.split(r"(?m)^#{2,3}\s+(.+)$", content)
    secs = []
    for i in range(1, len(parts), 2):
        head = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # anchor slug sometimes wraps onto the next line, e.g.
        #   "## Change ... for profile enable-existing-\n dataset-for-profile"
        # -> heading ends with a dangling hyphen; drop the fragment and the
        #    slug tail that leaked to the start of the body.
        if re.search(r"[a-z0-9]+(?:-[a-z0-9]+)*-$", head):
            head = re.sub(r"\s+[a-z0-9]+(?:-[a-z0-9]+)*-$", "", head)
            body = re.sub(r"^\s*[a-z0-9]+(?:-[a-z0-9]+)*\s+", " ", body.lstrip())
        head = ANCHOR_RE.sub("", head).strip()
        secs.append((head, body))
    return secs


def numbered_items(body: str):
    """Return text chunks following each '1. ' '2. ' marker."""
    items = re.split(r"(?m)^\s{0,6}\d+\.\s+", body)
    return [x for x in items[1:] if x.strip()]


def _ok_phrase(text: str) -> bool:
    if NOISE_TOKEN.search(text):
        return False
    w = text.split()
    if len(w) < LO or len(w) > HI:
        return False
    return bool(ACTION_RE.search(text))


def phase_from_clauses(head: str, clauses):
    """S2/S3: collapse a section into one phase: 'Heading: clause, clause, ...'."""
    head_c = clean(head)
    if not head_c:
        return None
    base = head_c[0].upper() + head_c[1:]
    acc = base
    for c in clauses:
        c = clean(c)
        if len(c.split()) < 2 or NOISE_TOKEN.search(c):
            continue
        cand = acc + ": " + c if acc == base else acc + ", " + c[0].lower() + c[1:]
        if len(cand.split()) > HI:
            break
        acc = cand
    acc = acc.rstrip(" .,;:") + "."
    return acc if _ok_phrase(acc) else None


def phase_from_item(item: str):
    """S1: turn one 'fat' numbered item into a phase."""
    ss = sents(item)
    if not ss:
        return None
    txt = clean(ss[0])
    k = 1
    while len(txt.split()) < LO and k < len(ss):
        nxt = clean(ss[k])
        if nxt and not NOISE_TOKEN.search(nxt):
            txt = (txt + " " + nxt).strip()
        k += 1
    txt = txt.rstrip(" .,;:") + "."
    return txt if _ok_phrase(txt) else None


def _h1_fallback_sections(content: str):
    """For docs with numbered steps but no usable H2/H3, treat each H3-less
    numbered block under the H1 as its own pseudo-section so steps aren't lost."""
    if not re.search(r"(?m)^\s{0,6}\d+\.\s", content):
        return []
    m = re.search(r"(?m)^#\s+(.+)$", content)
    head = ANCHOR_RE.sub("", m.group(1)).strip() if m else "Workflow"
    body = content[m.end():] if m else content
    return [(head, body)]


def extract_doc(content: str):
    """Strategy router -> ordered list of phase strings for one doc."""
    secs = sections(content)
    secs = [(h, b) for h, b in secs
            if not BOILER_HEAD.match(clean(h)) and not NOISE_TOKEN.search(h)
            and "?" not in h]  # drop Q&A headings
    if not secs:
        secs = _h1_fallback_sections(content)
    if not secs:
        return []

    # --- S1: fat numbered items as phases (quick-start guides) ---
    fat = []
    for h, b in secs:
        for it in numbered_items(b):
            first = sents(it)[0] if sents(it) else it
            if len(clean(first).split()) >= 10:
                p = phase_from_item(it)
                if p:
                    fat.append(p)
    if len(fat) >= 3:
        return fat[:6]

    # --- S2/S3: each section -> one phase ---
    phases = []
    for h, b in secs:
        items = numbered_items(b)
        if items:
            clauses = [sents(it)[0] if sents(it) else it for it in items]
        else:
            clauses = sents(b)[:5]
        p = phase_from_clauses(h, clauses)
        if p:
            phases.append(p)
    return phases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    here = Path(__file__).parent
    df = pd.read_excel(here / "doc_dataset.xlsx")
    print(f"Loaded {len(df)} rows", file=sys.stderr)

    rng = random.Random(42)
    workflows = []  # list of dict(url, title, phases)
    phase_dist = Counter()

    for _, row in df.iterrows():
        url = str(row.get("url", "") or "")
        title = str(row.get("title", "") or "")
        content = str(row.get("content", "") or "")

        if SKIP_URL.search(url):
            continue
        # video-only stub: short + has a video, no numbered/section structure
        if len(content) < 1500 and "video.tv.adobe.com" in content:
            continue

        cleaned = preclean_doc(content)
        phases = extract_doc(cleaned)

        # dedupe near-identical phases
        seen, uniq = set(), []
        for p in phases:
            k = p[:45].lower()
            if k not in seen:
                seen.add(k)
                uniq.append(p)

        # procedural gate: action verb enforced per-phase. Keep workflows whose
        # phase count is in the configured range.
        if not (MIN_PHASES <= len(uniq) <= MAX_PHASES):
            continue

        phase_dist[len(uniq)] += 1
        workflows.append({"url": url, "title": title, "phases": uniq})

    print(f"Workflows kept ({MIN_PHASES}-{MAX_PHASES} phases): {len(workflows)}", file=sys.stderr)
    print(f"Phase-count distribution: {dict(sorted(phase_dist.items()))}", file=sys.stderr)

    # ---- write workflow_phases.json ----
    wf_records = []
    for i, wf in enumerate(workflows, 1):
        rec = {"id": i, "url": wf["url"], "title": wf["title"]}
        for j, p in enumerate(wf["phases"], 1):
            rec[f"t{j}"] = p
        wf_records.append(rec)
    with open(here / "workflow_phases.json", "w") as f:
        json.dump(wf_records, f, ensure_ascii=False, indent=2)

    # ---- document-level split (no leakage) 80/10/10 ----
    idx = list(range(len(workflows)))
    rng.shuffle(idx)
    n = len(idx)
    ntest, nval = int(0.1 * n), int(0.1 * n)
    split = {}
    for rank, wi in enumerate(idx):
        split[wi] = "test" if rank < ntest else "val" if rank < ntest + nval else "train"

    # ---- build pairs: gap 1..MAX_GAP, forward + reverse ----
    # gap=1 -> consecutive; gap=2 -> skip-one. Bounded adjacency (not all-pairs)
    # keeps a valid ordering signal while limiting memorization surface.
    rows = {"train": [], "val": [], "test": []}
    for wi, wf in enumerate(workflows):
        sp = split[wi]
        phases = wf["phases"]
        for i in range(len(phases)):
            for j in range(i + 1, min(i + 1 + MAX_GAP, len(phases))):
                a, b = phases[i], phases[j]
                base = {"url": wf["url"], "title": wf["title"]}
                rows[sp].append({"text_1": a, "text_2": b, "label": 1,
                                 "step_1": f"t{i+1}", "step_2": f"t{j+1}", **base})
                rows[sp].append({"text_1": b, "text_2": a, "label": 0,
                                 "step_1": f"t{j+1}", "step_2": f"t{i+1}", **base})

    # global dedup of identical (text_1, text_2, label): boilerplate phrasing is
    # produced by multiple docs. Iterate train first so any cross-split collision
    # is resolved in favour of train -> no eval pair is ever also in train.
    seen = set()
    for sp in ("train", "val", "test"):
        kept = []
        for r in rows[sp]:
            key = (r["text_1"], r["text_2"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            kept.append(r)
        rows[sp] = kept

    for sp in ("train", "val", "test"):
        rng.shuffle(rows[sp])
        with open(here / f"directional_{sp}.jsonl", "w") as f:
            for r in rows[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(len(rows[sp]) for sp in rows)
    lbl = defaultdict(int)
    for r in rows["train"]:
        lbl[r["label"]] += 1
    print(f"\n[done] total pairs: {total}", file=sys.stderr)
    print(f"[done] split sizes: " + ", ".join(f"{sp}={len(rows[sp])}" for sp in rows), file=sys.stderr)
    print(f"[done] train label balance: {dict(lbl)}", file=sys.stderr)
    print(f"[done] wrote workflow_phases.json ({len(wf_records)} workflows)", file=sys.stderr)


if __name__ == "__main__":
    main()
