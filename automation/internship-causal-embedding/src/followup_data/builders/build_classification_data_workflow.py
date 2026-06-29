#!/usr/bin/env python3
"""
Build a WORKFLOW-ORDERED directional classification dataset from AJO tutorials.

This is a modified clone of ``build_classification_data.py``. The original is
left untouched. The goal here is to teach a model genuine *sequence / causality*
("which step comes before which") instead of a topic-tier classification proxy.

Why this exists
---------------
The model trained on the old dataset scored 2/30 on AJO orchestrated workflows
but 10/10 on a tier-eval drawn from the same corpus -- it had learned to match
topic clusters, not order. Root causes addressed here:

  * Old data had ZERO same-document / same-tier pairs, so the model never saw
    within-workflow step ordering.
  * 86.5% of old pairs were far-apart tier pairs -> easy topic shortcut.
  * ~29% of training text carried ``recommendation-more-help <uuid>`` plus
    HTML/CSS/JS, forum chrome, breadcrumbs, virus-scanner lines, raw UUIDs.
  * Tiers were curriculum-themed, not workflow-ordered.

What changed vs. the original (see CHANGES)
-------------------------------------------
1. SUBCHAPTER_TO_TIER re-numbered to the operational workflow order
   (Setup -> Data -> Audience -> Content -> Channel -> Campaign -> Journey ->
    Personalization -> Orchestration -> Conflict mgmt -> Governance ->
    Reporting -> AI -> Use-cases). A few mis-routed slugs are re-pointed
    (see SLUG_REASSIGNMENTS).
2. clean_text(): regex noise pipeline applied to every chunk before segmenting.
3. Sentence segments default to 1-3 sentences (was 2-5); 1-sentence segments OK.
4. NEW intra-document (execution-order) pairs: earlier sentence = earlier step.
5. NEW intra-tier (cross-sub-chapter) pairs, gap-weighted (more low-gap).
6. Cross-tier per-pair cap decays linearly with tier gap (moderate decay).
7. Split is done at the FINAL-TEXT level (each unique cleaned segment lives in
   exactly one of train/val/test; a pair is emitted only if both sides share a
   split). Labels are ALWAYS emitted both directions: forward=1, reverse=0.
8. Manifest reports per-source counts, label balance, and a leak-check.

Audit note (sub-chapter -> tier)
--------------------------------
The workflow order was verified against the real corpus document titles per
sub-chapter. Campaigns (4a-4c) were split out of the channels tier into their
own tier so "build a channel message" precedes "send a campaign" precedes
"orchestrate a journey". The ``2a`` intro bucket was a dumping ground for a few
advanced docs routed there by slug; those are re-pointed in SLUG_REASSIGNMENTS.

v2 (current defaults): smaller (~50k), de-duplicated dataset. Each unique
unordered (text_1, text_2) pair is emitted EXACTLY ONCE with a seeded random
direction (label 1 forward / label 0 reverse) -- no fwd/rev twins, which were
letting the model memorize "seen these two strings, flip on order" (84.8% test
acc but 1/30 AJO = overfit). Caps cut hard, cross-tier hardest, so intra-doc +
intra-tier (real step order) dominate the mix. Tier table, cleaning, splitting,
and schema are unchanged from v1.

Invoke (current v2 defaults):
  python3 build_classification_data_workflow.py \
      --corpus  ../../data/comparison_test/aep_docs_collection_v_24-03-2026.json \
      --out-dir ../../data/aep_causal_workflow_v2 \
      --unit sentence --min-sentences 1 --max-sentences 3 \
      --pair-scope all \
      --base-cap 180 --gap-decay 0.064 --far-gap-floor 20 \
      --intra-doc-max-gap 4 --intra-doc-cap 50 \
      --intra-tier-base-cap 220 --intra-tier-floor 20 \
      --split-level text \
      --seed 42 --val-frac 0.10 --test-frac 0.10

Outputs (in OUT_DIR):
  directional_train.jsonl
  directional_val.jsonl
  directional_test.jsonl
  directional_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 15-tier table (sub_chapter_code -> tier index) -- WORKFLOW ORDER
# ---------------------------------------------------------------------------
# Tier index == position in the operational workflow a marketer follows:
#   set up the product -> bring in data -> build audiences -> author content ->
#   wire up channels -> send campaigns -> orchestrate journeys -> personalize ->
#   advanced/experiment -> manage conflicts -> govern -> report -> AI -> labs.
SUBCHAPTER_TO_TIER: dict[str, int] = {
    # T1 Product orientation
    "2a": 1, "2b": 1,
    # T2 Admin / setup (access + sandboxes)
    "17a": 2, "17c": 2,
    # T3 Data foundations (schema, ingest, sources, export)
    "14a": 3, "14b": 3, "14c": 3,
    # T4 Profiles, audiences & subscriptions
    "8a": 4, "8b": 4, "8c": 4,
    # T5 Content authoring foundations
    "10a": 5, "10b": 5, "10c": 5, "10d": 5, "10e": 5, "10f": 5, "10g": 5,
    # T6 Channels
    "9a": 6, "9b": 6, "9c": 6, "9d": 6, "9e": 6,
    "9f": 6, "9g": 6, "9h": 6, "9i": 6, "9j": 6, "9k": 6,
    # T7 Campaigns (first sends) -- split out of channels
    "4a": 7, "4b": 7, "4c": 7,
    # T8 Journeys (core)
    "5a": 8, "5b": 8,
    # T9 Personalization & decisioning
    "11a": 9, "11b": 9, "13a": 9, "13b": 9, "13c": 9,
    # T10 Advanced journey patterns & experimentation
    "5c": 10, "12a": 10, "12b": 10,
    # T11 Multi-journey orchestration / conflict management
    "7a": 11, "7b": 11,
    # T12 Admin configuration
    "16a": 12, "16b": 12, "16c": 12, "17b": 12,
    # T13 Governance & privacy
    "18a": 13, "18b": 13, "18c": 13,
    # T14 Observability & reporting
    "15a": 14, "15b": 14, "15c": 14, "15d": 14, "15e": 14,
    # T15 AI agents/assistants + use cases, labs, capstones
    "19a": 15, "19b": 15, "2c": 15,
    "6": 15, "20a": 15, "20b": 15, "20c": 15, "20d": 15,
    "21a": 15, "21b": 15, "22": 15,
}

# Slug-level re-routing fixes (applied AFTER the original CHAPTER_RULES resolve a
# code). These correct docs that the introduction chapter's by_slug rule pulls
# into 2a even though they are advanced topics. Maps slug -> corrected code.
SLUG_REASSIGNMENTS: dict[str, str] = {
    "use-decisioning-to-personalize-web-offers": "13a",   # decisioning, not intro
    "create-audiences-using-web-sdk": "8b",               # audiences, not intro
    "identity-stitching-in-aep": "14a",                   # data foundations
}


# ---------------------------------------------------------------------------
# Sub-chapter assignment rules  (unchanged from the original builder)
# ---------------------------------------------------------------------------
CHAPTER_RULES: dict[str, dict] = {
    # --- Flat chapters (leaves grouped thematically by slug) ---
    "introduction-to-journey-optimizer": {
        "by_slug": {
            "journey-optimizer-overview": "2a",
            "introduction": "2a",
            "key-capabilities-and-user-interface": "2a",
            "architecture": "2a",
            "mobile-capabilities": "2b",
            "mobile-capabilities-for-developers": "2b",
            "ai-assistant": "2c",
        }
    },
    "journeys": {
        "by_slug": {
            "journey-designer-overview": "5a",
            "journey-agent-overview": "5a",
            "new-journey-designer": "5a",
            "introduction-to-building-a-journey": "5a",
            "lookup-dataset": "5b",
            "test-a-journey": "5b",
            "publish-a-journey": "5b",
            "content-decision-activity": "5b",
            "use-case-transactional-journey": "5c",
            "use-case-business-event": "5c",
            "use-case-read-audience": "5c",
            "use-case-audience-qualification": "5c",
            "mastering-multi-attribute-filtering": "5c",
            "journey-dry-run": "5c",
            "unlock-journey-reentry-with-supplemental-id": "5c",
            "update-content-in-live-journey": "5c",
            "copy-a-journey": "5c",
            "trigger-daily-journey-runs-after-batch-segmentation-completion": "5c",
        }
    },
    "loyalty": {
        "by_slug": {"create-a-loyalty-challenge": "6"}
    },
    "conflict-management": {
        "by_slug": {
            "identify-potential-conflicts": "7a",
            "assign-priority-score": "7a",
            "journey-frequency-capping-and-prioritization": "7b",
            "configure-and-apply-quiet-hours": "7b",
        }
    },
    "profiles-audiences-subscriptions": {
        "by_slug": {
            "profiles-and-audiences-overview": "8a",
            "unified-profile-and-segmentation-overview": "8a",
            "create-audiences-using-the-rule-builder": "8b",
            "import-and-activate-an-audience-by-uploading-a-csv-file": "8b",
            "subscriptions-and-landing-pages": "8c",
        }
    },
    "personalize-content": {
        "by_slug": {
            "personalization-editor-overview": "11a",
            "personalization-editor-playground": "11a",
            "profile-and-audience-membership-based-personalization": "11b",
            "add-offer-decisioning-to-messages": "11b",
            "use-contextual-event-information-for-personalization": "11b",
            "use-helper-functions-for-personalization": "11b",
            "use-and-manage-saved-expressions-in-personalization-library": "11b",
            "create-dynamic-content": "11b",
        }
    },
    "experimentation": {
        "by_slug": {
            "introduction-to-experimentation": "12a",
            "experimentation-agent-overview": "12a",
            "content-experiments-for-emails": "12b",
        }
    },
    "data-management": {
        "by_slug": {
            "set-up-data-overview": "14a",
            "create-schema": "14a",
            "map-identities": "14a",
            "create-datasets-and-ingest-data": "14b",
            "configure-source-connectors": "14b",
            "configure-dataset-export-destination": "14c",
            "export-datasets": "14c",
        }
    },
    "report-and-monitor": {
        "by_slug": {
            "report-and-monitor": "15a",
            "introduction-to-reporting": "15a",
            "monitor-and-analyze-your-journey-with-live-reports": "15b",
            "journey-reports": "15b",
            "channel-level-reports": "15c",
            "custom-action-monitoring-report": "15c",
            "all-time-reports": "15d",
            "export-reports-in-csv-format": "15d",
            "alerts": "15e",
            "enhanced-reporting-with-customer-journey-analytics": "15e",
        }
    },
    "access-control": {
        "by_slug": {
            "access-management": "17a",
            "attribute-based-access-control": "17b",
            "create-and-manage-sandboxes": "17c",
        }
    },
    "data-governance-and-privacy": {
        "by_slug": {
            "data-governance-framework": "18a",
            "classify-data-using-lables": "18a",
            "create-data-usage-policies": "18b",
            "enforce-data-usage-policies-in-journey-optimizer-channels": "18b",
            "mask-data-in-messages": "18c",
        }
    },
    "ai-assistant": {
        "by_slug": {}
    },

    # --- Nested chapters (use sub-chapter anchor) ---
    "create-campaigns": {
        "by_sub_anchor": {
            "action-campaigns": "4a",
            "api-triggered-campaigns": "4b",
            "orchestrated-campaigns": "4c",
        }
    },
    "channels": {
        "by_sub_anchor": {
            "code-based-experience-channel": "9b",
            "direct-mail-channel": "9c",
            "email-channel": "9d",
            "content-cards": "9e",
            "in-app-channel": "9f",
            "live-activities": "9g",
            "push-channel": "9h",
            "sms-channel": "9i",
            "web-channel": "9j",
            "whatsapp": "9k",
        },
        "by_direct_slug": {
            "mobile-app-optimization-overview": "9a",
        },
    },
    "content-management": {
        "by_sub_anchor": {
            "assets": "10b",
            "fragments": "10c",
            "content-templates": "10d",
            "multilingual-messaging": "10e",
            "ai-assistant": "10f",
        },
        "by_direct_slug": {
            "message-authoring-overview": "10a",
            "create-an-email-using-genstudio": "10g",
        },
    },
    "decision-capabilities": {
        "by_sub_anchor": {
            "decisioning": "13a",
            "decision-management": "13b",
        },
    },
    "configuration": {
        "by_sub_anchor": {
            "channel-configuration": "16a",
            "journey-configuration": "16b",
            "business-rules": "16c",
        }
    },
    "use-cases": {
        "by_sub_anchor": {
            "use-case-playbooks": "20d",
        },
        "by_direct_slug": {
            "customer-onboarding": "20a",
            "abandoned-cart": "20a",
            "enhance-customer-engagement": "20c",
        },
    },
    "exercises-and-challenges": {
        "by_sub_anchor": {
            "summit-labs": "21b",
        },
        "by_direct_slug": {
            "challenges": "21a",
        },
    },
    "live-sessions-and-deep-dives": {
        "by_slug": {
            "experience-league-live-show-recordings": "22",
        }
    },
}

# Absolute-URL overrides (TOC entries that are full http(s) URLs)
BY_ABSOLUTE_URL: dict[str, str] = {
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/channels/code-based-experience-channel/create-a-code-based-experience-campaign": "4a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/create-audiences-using-web-sdk/introduction": "8b",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/audiences/audience-builder/evaluate-audiences-on-demand": "8b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/trigger-journey-on-form-submission/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-real-time-weather-data/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-ranking-formulas-based-on-user-zip-code-and-income/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/use-decisioning-in-email-channel/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/use-decisioning-to-personalize-web-offers/introduction": "13c",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/introduction-to-journey-optimizer/ai-assistant": "19a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/content-management/ai-assistant/ai-assistant-for-content-generation-overview": "19a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-agent-overview": "19b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/experimentation/experimentation-agent-overview": "19b",
    "https://experienceleague.adobe.com/en/docs/experience-platform/rtcdp/use-cases/personalization-insights-engagement/use-cases-luma": "20a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-real-time-weather-data/introduction": "20b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-ranking-formulas-based-on-user-zip-code-and-income/introduction": "20b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/scaling-orchestration-to-omnichannel-engagement/introduction.md": "20c",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/use-case-playbooks/overview": "20d",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/use-case-playbooks/configure-a-playbook-sandbox": "20d",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/use-case-playbooks/create-and-publish-a-playbook-instance": "20d",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/configure-a-training-sandbox/introduction-and-prerequisites": "21a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/challenges/introduction-and-prerequisites": "21a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/build-personalized-mobile-moments/lab-overview": "21b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/scaling-orchestration-to-omnichannel-engagement/introduction": "21b",
}


# ---------------------------------------------------------------------------
# TOC parsing  (unchanged from the original builder)
# ---------------------------------------------------------------------------
TOC_URL = "https://raw.githubusercontent.com/AdobeDocs/journey-optimizer-learn.en/main/help/_ajo-main/TOC.md"
BASE_URL = "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials"

_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_ANCHOR = re.compile(r"\{#([a-z0-9-]+)\}")
_RE_TRAILING_ATTRS = re.compile(r"\s*\{[^}]+\}\s*$")


def _strip_attrs(s: str) -> str:
    return _RE_TRAILING_ATTRS.sub("", s).strip()


def _extract_anchor(s: str) -> str | None:
    m = _RE_ANCHOR.search(s)
    return m.group(1) if m else None


def _leaf_slug(target: str) -> str | None:
    target = _strip_attrs(target)
    if target.startswith("http"):
        return None
    m = re.match(r"/help/.+/([^/]+)\.md$", target) or re.match(r"/help/([^/]+)\.md$", target)
    return m.group(1) if m else None


def fetch_toc(url: str = TOC_URL) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def parse_toc(toc_text: str) -> list[dict]:
    """Return flat list of leaves, each with full ancestor-anchor path."""
    entries: list[tuple[int, str, str, str | None]] = []
    for ln in toc_text.splitlines():
        s = ln.strip()
        if not s.startswith("+"):
            continue
        indent = (len(ln) - len(ln.lstrip())) // 2
        link = _RE_LINK.search(s)
        if link:
            entries.append((indent, "leaf", link.group(1), link.group(2)))
        else:
            name = _RE_ANCHOR.sub("", s[1:]).strip()
            entries.append((indent, "header", name, _extract_anchor(s)))

    leaves: list[dict] = []
    stack: list[tuple[int, str, str | None]] = []
    for ind, kind, name, extra in entries:
        while stack and stack[-1][0] >= ind:
            stack.pop()
        if kind == "header":
            stack.append((ind, name, extra))
        else:
            title, target = name, extra
            if not stack:
                continue
            chapter_name = stack[0][1]
            chapter_anchor = stack[0][2]
            sub_anchor = stack[1][2] if len(stack) > 1 else None
            sub_sub_anchor = stack[2][2] if len(stack) > 2 else None
            leaves.append({
                "title": title,
                "target": target,
                "chapter_name": chapter_name,
                "chapter_anchor": chapter_anchor,
                "sub_anchor": sub_anchor,
                "sub_sub_anchor": sub_sub_anchor,
                "slug": _leaf_slug(target),
                "absolute": _strip_attrs(target) if _strip_attrs(target).startswith("http") else None,
            })
    return leaves


# ---------------------------------------------------------------------------
# URL resolution  (unchanged from the original builder)
# ---------------------------------------------------------------------------
def url_candidates(leaf: dict) -> list[str]:
    if leaf["absolute"]:
        return [leaf["absolute"], leaf["absolute"].rstrip("/")]
    ch = leaf["chapter_anchor"]
    slug = leaf["slug"]
    if not ch or not slug:
        return []
    depths: list[list[str]] = []
    full = [ch]
    if leaf["sub_anchor"]:
        full.append(leaf["sub_anchor"])
    if leaf["sub_sub_anchor"]:
        full.append(leaf["sub_sub_anchor"])
    for L in range(len(full), 0, -1):
        depths.append(full[:L] + [slug])
    return [f"{BASE_URL}/{'/'.join(p)}" for p in depths]


# ---------------------------------------------------------------------------
# Sub-chapter assignment  (CHANGE 1: apply SLUG_REASSIGNMENTS after resolve)
# ---------------------------------------------------------------------------
def assign_subchapter(leaf: dict) -> str | None:
    code = _assign_subchapter_raw(leaf)
    if code is not None and leaf.get("slug") in SLUG_REASSIGNMENTS:
        return SLUG_REASSIGNMENTS[leaf["slug"]]
    return code


def _assign_subchapter_raw(leaf: dict) -> str | None:
    if leaf["absolute"] and leaf["absolute"] in BY_ABSOLUTE_URL:
        return BY_ABSOLUTE_URL[leaf["absolute"]]
    rules = CHAPTER_RULES.get(leaf["chapter_anchor"] or "")
    if not rules:
        return None
    sub = leaf["sub_anchor"]
    slug = leaf["slug"]
    if sub and "by_sub_anchor" in rules and sub in rules["by_sub_anchor"]:
        return rules["by_sub_anchor"][sub]
    if slug:
        if "by_direct_slug" in rules and slug in rules["by_direct_slug"]:
            return rules["by_direct_slug"][slug]
        if "by_slug" in rules and slug in rules["by_slug"]:
            return rules["by_slug"][slug]
    return None


# ---------------------------------------------------------------------------
# CHANGE 2: noise cleanup
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HELP_RE = re.compile(r"recommendation-more-help.*", re.S)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.S | re.I)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.S | re.I)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Lines that are clearly site chrome / CSS / JS / forum noise -> drop whole line.
_CHROME_LINE_RE = re.compile(
    r"^\s*(?:"
    r"Skip to main content|Create new post|Login|Home|Product Communities|"
    r"Adobe Employee|Adobe Journey Optimizer|EXL Footer\b.*|"
    r"Sorry, our virus scanner.*|OK|en|Forum\|.*|"
    r"(?:Author|Creator|Last update)\s*:.*|"
    r"[-.\w]*\s*(?:start|end)\b.*|"            # "Topics customizations end", etc.
    r"@(?:media|keyframes|import|font-face)\b.*|"
    r"--[\w-]+\s*:.*|"                          # CSS custom properties
    r"[.#]?[\w-]+\s*\{.*|"                      # CSS selector line opening a block
    r"\}|\{|"                                   # lone braces
    r"[\w-]+\s*:\s*[^;]+;\s*$|"                 # single css declaration line
    r"z-index\b.*|overflow\b.*|position\b.*|display\b.*"
    r")\s*$",
    re.I,
)
# Sentence-level transcript boilerplate to drop after sentence splitting.
_DROP_SENT_RE = re.compile(
    r"\b(?:thanks? for watching|thank you for watching|in this video|"
    r"let me show you|hi everyone|hello everyone|"
    r"visit the product documentation|for more information)\b",
    re.I,
)
# Leading navigation breadcrumb header that prefixes most pages, e.g.:
#   "Documentation \n Journey Optimizer \n Journey Optimizer Tutorials \n
#    <Title><Title> <Title>Understand what ..."
# The header is a run of short title-case lines (no sentence punctuation)
# starting at "Documentation". We consume "Documentation" plus all following
# lines that look like breadcrumb/title lines (short, no terminal . ! ?) up to
# the first real prose line. This is more robust than enumerating exact labels.
_BREADCRUMB_RE = re.compile(
    r"^\s*Documentation\b[ \t]*\n+"
    r"(?:[ \t]*[^\n]{0,80}\n+)*?"           # short breadcrumb/title lines (non-greedy)
    r"(?=[^\n]*[.!?])",                      # stop before the first line with sentence punctuation
    re.I,
)
# Doubled / tripled phrase: a span immediately repeated verbatim
# ("Data flow diagramData flow diagram", "TitleTitle Title"). Collapse repeats.
# Bounded length to avoid pathological backtracking. The optional whitespace
# between copies catches both "XX" and "X X" forms.
_DOUBLED_RE = re.compile(r"(.{6,90}?)\s*\1")


def _url_fraction(text: str) -> float:
    url_chars = sum(len(m.group()) for m in _URL_RE.finditer(text))
    return url_chars / max(len(text), 1)


def clean_text(text: str) -> str:
    """Strip boilerplate / markup / IDs from a raw corpus chunk.

    Applied once per chunk in load_corpus, before any segmentation. The order
    matters: kill big blocks (script/style/help footer) first, then per-line
    chrome, then collapse whitespace.
    """
    if not text:
        return ""
    t = _HELP_RE.sub(" ", text)
    t = _SCRIPT_RE.sub(" ", t)
    t = _STYLE_RE.sub(" ", t)
    t = _TAG_RE.sub(" ", t)
    t = _UUID_RE.sub(" ", t)
    # Strip the leading navigation breadcrumb (Documentation / Tutorials / ...).
    t = _BREADCRUMB_RE.sub("", t)
    # Drop chrome / CSS / JS lines.
    kept: list[str] = []
    for line in t.splitlines():
        if _CHROME_LINE_RE.match(line):
            continue
        kept.append(line)
    t = "\n".join(kept)
    # Collapse runs of whitespace into single spaces.
    t = re.sub(r"\s+", " ", t).strip()
    # Collapse immediately-repeated phrases ("TitleTitle Title" artifacts).
    # Iterate to a fixed point (handles chained triples), bounded for safety.
    for _ in range(4):
        new = _DOUBLED_RE.sub(r"\1", t)
        if new == t:
            break
        t = new
    return t


# ---------------------------------------------------------------------------
# Corpus loading  (CHANGE 2: clean each chunk before concatenating)
# ---------------------------------------------------------------------------
def load_corpus(path: Path, max_url_fraction: float = 0.3) -> dict[str, dict]:
    """Map sourceUrl -> {'text': cleaned concatenated chunks, 'title', 'n_chunks'}."""
    with path.open() as f:
        data = json.load(f)
    out: dict[str, dict] = {}
    for group in data:
        if not group:
            continue
        url = group[0]["metadata"].get("sourceUrl")
        if not url:
            continue
        title = group[0]["metadata"].get("title", "")
        texts: list[str] = []
        for entry in group:
            for ch in entry.get("chunks", []) or []:
                raw = ch.get("data") or ch.get("text") or ch.get("content") or ""
                if not raw:
                    continue
                if _url_fraction(raw) > max_url_fraction:
                    continue
                cleaned = clean_text(raw)
                if cleaned:
                    texts.append(cleaned)
        body = "\n\n".join(texts).strip()
        if not body:
            body = clean_text(title)
        if url in out:
            if len(body) > len(out[url]["text"]):
                out[url] = {"text": body, "title": title, "n_chunks": len(texts)}
        else:
            out[url] = {"text": body, "title": title, "n_chunks": len(texts)}
    return out


# ---------------------------------------------------------------------------
# Sentence segmentation  (CHANGE 3: 1-3 sentences, allow 1-sentence trailing)
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_MIN_SEG_CHARS = 25  # drop tiny fragments left after cleaning


def split_sentences(text: str, min_sentences: int = 1, max_sentences: int = 3) -> list[str]:
    """Pack consecutive sentences into segments of min..max sentences.

    When min_sentences == 1, a 1-sentence trailing segment is kept (not merged).
    Segments shorter than _MIN_SEG_CHARS after stripping are dropped.
    """
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in _SENT_RE.split(clean) if s.strip()]
    if not sentences:
        return []

    segments: list[list[str]] = []
    buf: list[str] = []
    for s in sentences:
        buf.append(s)
        if len(buf) >= max_sentences:
            segments.append(buf)
            buf = []
    if buf:
        segments.append(buf)

    # Merge a short trailing fragment only when we actually require >1 sentence.
    if min_sentences > 1 and len(segments) > 1 and len(segments[-1]) < min_sentences:
        segments[-2].extend(segments.pop())

    out = [" ".join(seg).strip() for seg in segments]
    return [s for s in out if len(s) >= _MIN_SEG_CHARS]


# ---------------------------------------------------------------------------
# CHANGE 4 helper: action-sentence filter (optional)
# ---------------------------------------------------------------------------
_ACTION_VERBS = (
    "select", "click", "choose", "navigate", "go to", "open", "create", "add",
    "enter", "type", "drag", "drop", "name", "save", "publish", "configure",
    "set", "define", "map", "ingest", "upload", "import", "export", "build",
    "enable", "toggle", "check", "fill", "specify", "activate", "send", "launch",
    "review", "preview", "test", "apply", "assign", "switch", "search", "scroll",
    "hit", "press", "pick", "provide", "confirm", "remove", "delete", "duplicate",
    "deduplicate", "pause", "wait", "target", "filter", "enrich", "deliver", "push",
)
_ACTION_RE = re.compile(
    r"\b(" + "|".join(v.replace(" ", r"\s") for v in _ACTION_VERBS) + r")\b", re.I
)


def _is_action_segment(seg: str) -> bool:
    if _DROP_SENT_RE.search(seg):
        return False
    return bool(_ACTION_RE.search(seg))


# ---------------------------------------------------------------------------
# CHANGE 1 audit print
# ---------------------------------------------------------------------------
def _print_tier_audit(resolved: list[dict], corpus: dict[str, dict]) -> None:
    """Print sub-chapter -> tier with example document titles, for review."""
    by_sub: dict[str, list[str]] = defaultdict(list)
    for r in resolved:
        by_sub[r["subchapter"]].append(corpus.get(r["url"], {}).get("title", r["url"])[:64])

    def codekey(c: str):
        m = re.match(r"(\d+)([a-z]?)", c)
        return (int(m.group(1)), m.group(2)) if m else (999, c)

    print("[audit] sub-chapter -> tier (workflow order) with sample titles:", file=sys.stderr)
    for code in sorted(by_sub, key=codekey):
        tier = SUBCHAPTER_TO_TIER.get(code, "?")
        titles = "; ".join(by_sub[code][:3])
        print(f"        {code:>4s} -> T{tier:<2}  ({len(by_sub[code])} docs)  {titles}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sub-chapter ordering within a tier
# ---------------------------------------------------------------------------
def sub_order(code: str) -> int:
    """Rank of a sub-chapter within its tier, by the letter suffix (a=1, b=2...).

    Numeric-only codes (e.g. '6', '22') have no suffix -> rank 0.
    """
    m = re.match(r"\d+([a-z])$", code)
    if not m:
        return 0
    return ord(m.group(1)) - ord("a") + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="aep_docs_collection_v_24-03-2026.json")
    ap.add_argument("--assignments", default=None,
                    help="Pre-resolved assignments JSON ({url,subchapter,tier} list "
                         "or {'assignments':[...]}); when set, TOC fetch is skipped.")
    ap.add_argument("--out-dir", default="data/aep_causal_workflow_v2")
    ap.add_argument("--unit", choices=["sentence"], default="sentence",
                    help="Only 'sentence' is supported in the workflow builder.")
    ap.add_argument("--min-sentences", type=int, default=1)
    ap.add_argument("--max-sentences", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--pair-scope", choices=["adjacent", "all"], default="all",
                    help="adjacent = only tier i<->i+1; all = every cross-tier pair "
                         "(gap decay then thins the far pairs).")
    ap.add_argument("--split-level", choices=["text"], default="text",
                    help="Only final-text-level splitting is supported "
                         "(each unique cleaned segment -> one split, no leakage).")
    # cross-tier gap decay
    ap.add_argument("--base-cap", type=int, default=180,
                    help="Max chunk-pair TUPLES per adjacent (gap=1) tier-pair per "
                         "split. v2: single direction per pair (no fwd+rev doubling).")
    ap.add_argument("--gap-decay", type=float, default=0.064,
                    help="Linear decay per unit of tier gap: "
                         "cap(g)=max(floor, base*(1-decay*(g-1))).")
    ap.add_argument("--far-gap-floor", type=int, default=20,
                    help="Minimum tuples kept for far tier-pairs.")
    # intra-tier (cross-sub-chapter)
    ap.add_argument("--intra-tier-base-cap", type=int, default=220,
                    help="Max tuples per (sub_i<sub_j) pair within a tier at sub-gap=1; "
                         "scaled by 1/sub_gap.")
    ap.add_argument("--intra-tier-floor", type=int, default=20)
    # intra-doc (execution order)
    ap.add_argument("--intra-doc-max-gap", type=int, default=4,
                    help="Pair sentence i with j only if 0 < j-i <= this.")
    ap.add_argument("--intra-doc-cap", type=int, default=50,
                    help="Max sentence-pair tuples kept per document.")
    ap.add_argument("--action-only", action="store_true", default=False,
                    help="Restrict intra-doc pairs to action-verb sentences.")
    ap.add_argument("--cap-splits", choices=["train", "all"], default="all",
                    help="Apply caps to train only, or to all splits. Default 'all' "
                         "so val/test are also bounded (keeps balance proportional).")
    ap.add_argument("--cap-seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- corpus --
    corpus_path = Path(args.corpus)
    print(f"[corpus] loading + cleaning {corpus_path}", file=sys.stderr)
    corpus = load_corpus(corpus_path)
    print(f"[corpus] {len(corpus)} unique URLs", file=sys.stderr)

    # -- resolve docs --
    resolved, url_misses, subchapter_misses = _resolve_docs(args, corpus)

    tier_docs: dict[int, list[dict]] = defaultdict(list)
    for r in resolved:
        tier_docs[r["tier"]].append(r)
    _print_tier_audit(resolved, corpus)
    print("[resolve] docs per tier:", file=sys.stderr)
    for t in sorted(tier_docs):
        subs = defaultdict(int)
        for d in tier_docs[t]:
            subs[d["subchapter"]] += 1
        subs_str = ", ".join(f"{k}:{v}" for k, v in sorted(subs.items()))
        print(f"           T{t:>2}  n={len(tier_docs[t]):>3}  [{subs_str}]", file=sys.stderr)

    # -- segment every doc into sentence segments --
    print(f"[segment] unit=sentence  min={args.min_sentences} max={args.max_sentences}", file=sys.stderr)
    # doc_segs[url] = ordered list of segment strings (sentence order preserved)
    doc_segs: dict[str, list[str]] = {}
    for r in resolved:
        body = corpus[r["url"]]["text"]
        doc_segs[r["url"]] = split_sentences(body, args.min_sentences, args.max_sentences)
    total_segs = sum(len(s) for s in doc_segs.values())
    print(f"[segment] {total_segs} segments across {len(doc_segs)} docs", file=sys.stderr)

    # -- CHANGE 7: final-text-level split (each unique segment -> one split) --
    # Pool unique segments per tier, assign 80/10/10, deterministic per tier.
    seg_split: dict[str, str] = {}          # normalized text -> split
    split_counts: dict[str, int] = defaultdict(int)
    segs_by_tier: dict[int, set[str]] = defaultdict(set)
    for r in resolved:
        for s in doc_segs[r["url"]]:
            segs_by_tier[r["tier"]].add(s)
    for t in sorted(segs_by_tier):
        segs = sorted(segs_by_tier[t])      # deterministic order before shuffle
        rng = random.Random(args.seed + (hash(f"T{t}") % 100000))
        rng.shuffle(segs)
        n = len(segs)
        n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * args.val_frac)) if n - n_test >= 2 else 0
        for i, s in enumerate(segs):
            sp = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            # First-seen tier wins if a segment somehow appears in two tiers.
            if s not in seg_split:
                seg_split[s] = sp
                split_counts[sp] += 1
    print(f"[split] text-level segments per split: {dict(split_counts)}", file=sys.stderr)

    def split_of(seg: str) -> str:
        return seg_split.get(seg, "train")

    # -- writers + emit helpers (CHANGE A: one record per unique unordered pair,
    #    direction = seeded coin flip; no fwd+rev twins) --
    writers = {sp: (out_dir / f"directional_{sp}.jsonl").open("w") for sp in ("train", "val", "test")}
    counts: dict[str, int] = defaultdict(int)
    label_counts: dict[int, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    # collect texts per split for the leak assertion
    split_texts: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    # CHANGE A: dedup unordered text pairs across ALL sources -> emit each once
    seen_pairs: set[frozenset[str]] = set()

    def emit_pair(earlier: str, later: str,
                  t_e: int, t_l: int, sub_e: str, sub_l: str,
                  url_e: str, url_l: str, source: str) -> None:
        """Emit ONE record per unique unordered (earlier, later) pair.

        Direction is a seeded per-pair coin flip: forward (earlier->later,
        label 1) or reverse (later->earlier, label 0). Never both -> removes
        the fwd/rev memorization twin. Deduped across sources by unordered key.
        Only emits when both segments share a split (leak guard).
        """
        sp = split_of(earlier)
        if sp != split_of(later):
            return
        if earlier == later:
            return
        key = frozenset((earlier, later))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        # seeded per-pair direction (deterministic, reproducible)
        rng_dir = random.Random((args.cap_seed, "dir", earlier, later).__hash__())
        if rng_dir.random() < 0.5:
            # forward, label 1 (workflow-earlier text first)
            rec = {
                "text_1": earlier, "text_2": later, "label": 1,
                "tier_1": t_e, "tier_2": t_l, "sub_1": sub_e, "sub_2": sub_l,
                "url_1": url_e, "url_2": url_l,
            }
            label_counts[1] += 1
        else:
            # reverse, label 0
            rec = {
                "text_1": later, "text_2": earlier, "label": 0,
                "tier_1": t_l, "tier_2": t_e, "sub_1": sub_l, "sub_2": sub_e,
                "url_1": url_l, "url_2": url_e,
            }
            label_counts[0] += 1
        writers[sp].write(json.dumps(rec) + "\n")
        counts[sp] += 1
        source_counts[source] += 1
        split_texts[sp].add(earlier)
        split_texts[sp].add(later)

    cap_splits = {"train"} if args.cap_splits == "train" else {"train", "val", "test"}

    # ===================================================================
    # SOURCE A -- intra-document execution-order pairs (CHANGE 4)
    # ===================================================================
    print("[pairs] source A: intra-document (execution order)", file=sys.stderr)
    for r in resolved:
        url = r["url"]
        segs = doc_segs[url]
        if args.action_only:
            segs = [s for s in segs if _is_action_segment(s)]
        if len(segs) < 2:
            continue
        # candidate (i,j) within window, j>i
        cand: list[tuple[int, int]] = []
        for i in range(len(segs)):
            for j in range(i + 1, min(i + 1 + args.intra_doc_max_gap, len(segs))):
                cand.append((i, j))
        rng = random.Random((args.cap_seed, "intra_doc", url).__hash__())
        rng.shuffle(cand)
        cand = cand[: args.intra_doc_cap]
        for i, j in cand:
            emit_pair(segs[i], segs[j], r["tier"], r["tier"],
                      r["subchapter"], r["subchapter"], url, url, "intra_doc")

    # ===================================================================
    # SOURCE B -- intra-tier cross-sub-chapter pairs, gap-weighted (CHANGE 5)
    # ===================================================================
    print("[pairs] source B: intra-tier (cross sub-chapter, gap-weighted)", file=sys.stderr)
    for t in sorted(tier_docs):
        # group this tier's segments by sub-chapter, tagged with their url
        sub_segs: dict[str, list[tuple[str, str]]] = defaultdict(list)  # sub -> [(seg,url)]
        for r in tier_docs[t]:
            for s in doc_segs[r["url"]]:
                sub_segs[r["subchapter"]].append((s, r["url"]))
        subs = sorted(sub_segs, key=sub_order)
        for a in range(len(subs)):
            for b in range(a + 1, len(subs)):
                sub_e, sub_l = subs[a], subs[b]
                gap = abs(sub_order(sub_l) - sub_order(sub_e)) or 1
                cap = max(args.intra_tier_floor, args.intra_tier_base_cap // gap)
                lows = sub_segs[sub_e]
                highs = sub_segs[sub_l]
                if not lows or not highs:
                    continue
                # sample cap tuples from the cartesian product without materializing it
                rng = random.Random((args.cap_seed, "intra_tier", t, sub_e, sub_l).__hash__())
                n_take = cap if t not in () else cap  # cap applies to all tiers
                pairs = _sample_cross(lows, highs, n_take, rng)
                for (se, ue), (sl, ul) in pairs:
                    emit_pair(se, sl, t, t, sub_e, sub_l, ue, ul, "intra_tier")

    # ===================================================================
    # SOURCE C -- cross-tier pairs with linear gap decay (CHANGE 6)
    # ===================================================================
    print("[pairs] source C: cross-tier (linear gap decay)", file=sys.stderr)
    tiers_sorted = sorted(tier_docs)
    if args.pair_scope == "adjacent":
        tier_pairs = [(tiers_sorted[i], tiers_sorted[i + 1]) for i in range(len(tiers_sorted) - 1)]
    else:
        tier_pairs = [(tiers_sorted[i], tiers_sorted[j])
                      for i in range(len(tiers_sorted)) for j in range(i + 1, len(tiers_sorted))]

    # tier -> list of (seg, sub, url)
    tier_seg_index: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for r in resolved:
        for s in doc_segs[r["url"]]:
            tier_seg_index[r["tier"]].append((s, r["subchapter"], r["url"]))

    cross_pre: dict[str, int] = {}
    cross_post: dict[str, int] = {}
    for t_low, t_hi in tier_pairs:
        g = t_hi - t_low
        cap = max(args.far_gap_floor, round(args.base_cap * (1.0 - args.gap_decay * (g - 1))))
        lows = tier_seg_index.get(t_low, [])
        highs = tier_seg_index.get(t_hi, [])
        if not lows or not highs:
            continue
        cross_pre[f"T{t_low}-T{t_hi}"] = len(lows) * len(highs)
        rng = random.Random((args.cap_seed, "cross", t_low, t_hi).__hash__())
        pairs = _sample_cross(
            [(s, (sub, url)) for s, sub, url in lows],
            [(s, (sub, url)) for s, sub, url in highs],
            cap, rng,
        )
        before = source_counts["cross_tier"]
        for (se, (sub_e, ue)), (sl, (sub_l, ul)) in pairs:
            emit_pair(se, sl, t_low, t_hi, sub_e, sub_l, ue, ul, "cross_tier")
        # actual emitted = delta in real source count (accounts for split + dedup skips)
        cross_post[f"T{t_low}-T{t_hi}"] = source_counts["cross_tier"] - before

    for w in writers.values():
        w.close()

    # -- CHANGE 7: leak assertion --
    leak = {
        "train_n_text": len(split_texts["train"]),
        "val_n_text": len(split_texts["val"]),
        "test_n_text": len(split_texts["test"]),
        "train_inter_val": len(split_texts["train"] & split_texts["val"]),
        "train_inter_test": len(split_texts["train"] & split_texts["test"]),
        "val_inter_test": len(split_texts["val"] & split_texts["test"]),
    }
    print(f"[leak-check] {leak}", file=sys.stderr)
    assert leak["train_inter_val"] == 0 and leak["train_inter_test"] == 0 and leak["val_inter_test"] == 0, \
        f"LEAK DETECTED across splits: {leak}"

    # -- manifest (CHANGE 8) --
    manifest = {
        "builder": "build_classification_data_workflow.py",
        "workflow_tier_table": SUBCHAPTER_TO_TIER,
        "slug_reassignments": SLUG_REASSIGNMENTS,
        "cleaning": "v1",
        "unit": "sentence",
        "min_sentences": args.min_sentences,
        "max_sentences": args.max_sentences,
        "pair_scope": args.pair_scope,
        "split_level": "text",
        "labels": "single-direction-per-pair (random fwd=1/rev=0, deduped unordered)",
        "dedup": "unordered-pair (frozenset), emitted once across all sources",
        "base_cap": args.base_cap,
        "gap_decay": args.gap_decay,
        "far_gap_floor": args.far_gap_floor,
        "intra_tier_base_cap": args.intra_tier_base_cap,
        "intra_tier_floor": args.intra_tier_floor,
        "intra_doc_max_gap": args.intra_doc_max_gap,
        "intra_doc_cap": args.intra_doc_cap,
        "action_only": args.action_only,
        "cap_splits": args.cap_splits,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_docs_resolved": len(resolved),
        "n_url_misses": len(url_misses),
        "n_subchapter_misses": len(subchapter_misses),
        "docs_per_tier": {t: len(tier_docs[t]) for t in tiers_sorted},
        "segment_split_counts": dict(split_counts),
        "sample_counts": dict(counts),
        "total_samples": sum(counts.values()),
        "n_unique_pairs": len(seen_pairs),
        "label_balance": dict(label_counts),
        "label_balance_note": "approximate 50/50 by seeded per-pair coin flip",
        "source_counts": dict(source_counts),
        "cross_tier_pre_cap": cross_pre,
        "cross_tier_post_cap": cross_post,
        "leak_check": leak,
        "url_misses": url_misses,
        "subchapter_misses": subchapter_misses,
    }
    (out_dir / "directional_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[done] samples per split: {dict(counts)}  total={sum(counts.values())}", file=sys.stderr)
    print(f"[done] per-source: {dict(source_counts)}", file=sys.stderr)
    print(f"[done] label balance: {dict(label_counts)}", file=sys.stderr)
    print(f"[done] manifest: {out_dir/'directional_manifest.json'}", file=sys.stderr)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sample_cross(lows: list, highs: list, n_take: int, rng: random.Random) -> list:
    """Sample up to n_take (low, high) tuples from the cartesian product of two
    lists without materializing the full product. Each element of lows/highs is
    an arbitrary tuple whose first element is the segment text."""
    total = len(lows) * len(highs)
    if total == 0:
        return []
    if total <= n_take:
        return [(lo, hi) for lo in lows for hi in highs]
    seen: set[tuple[int, int]] = set()
    out: list = []
    n_low, n_high = len(lows), len(highs)
    # rejection-sample distinct (i,j) index pairs
    attempts = 0
    max_attempts = n_take * 20 + 100
    while len(out) < n_take and attempts < max_attempts:
        i = rng.randrange(n_low)
        j = rng.randrange(n_high)
        attempts += 1
        if (i, j) in seen:
            continue
        seen.add((i, j))
        out.append((lows[i], highs[j]))
    return out


def _resolve_docs(args, corpus: dict[str, dict]):
    """Resolve docs either from a pre-resolved assignments file or the AJO TOC.

    Returns (resolved, url_misses, subchapter_misses)."""
    resolved: list[dict] = []
    url_misses: list[dict] = []
    subchapter_misses: list[dict] = []

    if args.assignments:
        print(f"[assignments] loading {args.assignments}", file=sys.stderr)
        with open(args.assignments) as _f:
            _data = json.load(_f)
        _items = _data["assignments"] if isinstance(_data, dict) and "assignments" in _data else _data
        _seen: set[str] = set()
        for it in _items:
            raw_url = it["url"]
            while "/./" in raw_url:
                raw_url = raw_url.replace("/./", "/")
            if raw_url in _seen:
                continue
            _seen.add(raw_url)
            if raw_url not in corpus:
                url_misses.append({"url": raw_url, "reason": "not in corpus"})
                continue
            sub = it.get("subchapter")
            tier = it.get("tier")
            # honour any slug-level reassignment for parity with the TOC path
            slug = raw_url.rstrip("/").split("/")[-1]
            if sub is not None and slug in SLUG_REASSIGNMENTS:
                sub = SLUG_REASSIGNMENTS[slug]
                tier = SUBCHAPTER_TO_TIER.get(sub, tier)
            if not sub or tier is None:
                subchapter_misses.append({"url": raw_url, "reason": "missing subchapter/tier"})
                continue
            resolved.append({
                "url": raw_url,
                "title": corpus[raw_url]["title"] or it.get("title", ""),
                "subchapter": sub,
                "tier": int(tier),
            })
        print(f"[assignments] {len(resolved)} resolved, {len(url_misses)} url misses, "
              f"{len(subchapter_misses)} subchapter misses", file=sys.stderr)
    else:
        print(f"[toc] fetching {TOC_URL}", file=sys.stderr)
        toc = fetch_toc()
        leaves = parse_toc(toc)
        print(f"[toc] parsed {len(leaves)} leaves", file=sys.stderr)
        for leaf in leaves:
            sub = assign_subchapter(leaf)
            url_found = None
            for u in url_candidates(leaf):
                if u in corpus:
                    url_found = u
                    break
            if not url_found:
                url_misses.append({"chapter": leaf["chapter_name"], "title": leaf["title"], "target": leaf["target"]})
                continue
            if not sub:
                subchapter_misses.append({"chapter": leaf["chapter_name"], "title": leaf["title"], "url": url_found})
                continue
            if sub not in SUBCHAPTER_TO_TIER:
                subchapter_misses.append({"chapter": leaf["chapter_name"], "title": leaf["title"], "url": url_found, "reason": f"unknown code {sub}"})
                continue
            resolved.append({
                "url": url_found,
                "title": leaf["title"],
                "chapter": leaf["chapter_name"],
                "subchapter": sub,
                "tier": SUBCHAPTER_TO_TIER[sub],
            })
        by_url: dict[str, dict] = {}
        for r in resolved:
            if r["url"] not in by_url:
                by_url[r["url"]] = r
        resolved = list(by_url.values())
        print(f"[resolve] {len(resolved)} resolved, {len(url_misses)} url misses, "
              f"{len(subchapter_misses)} subchapter misses", file=sys.stderr)

    return resolved, url_misses, subchapter_misses


if __name__ == "__main__":
    main()
