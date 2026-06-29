#!/usr/bin/env python3
"""
Build tier-based directional classification dataset from AJO tutorials.

SEMANTIC-TEST VARIANT
---------------------
Identical to build_classification_data.py for the train and val splits, but the
TEST split is built by SEMANTIC SIMILARITY instead of a random direction
coin-flip:

  * For each tier-pair (t_low, t_hi) in the test split, every cross-tier
    chunk-pair (c_low, c_hi) is scored with cosine similarity using
    sentence-transformers/all-MiniLM-L6-v2 (mean-pooled HF embeddings).
  * The pairs are ranked by similarity. The MOST-similar pairs are emitted as
    label 0 (reverse direction, high-tier first); the LEAST-similar pairs are
    emitted as label 1 (forward direction, low-tier first).
  * The per-tier-pair count of label-0 and label-1 samples is taken from an
    existing reference test file (--match-counts) so the total number of 0s and
    1s — and their per-tier-pair distribution — exactly matches the current
    test set.

Pipeline (unchanged from the base builder):
  1. Parse TOC.md (fetched from Adobe's GitHub) into chapters + leaves.
  2. Map each leaf to its corpus URL (ancestor-anchor + slug rule).
  3. Assign each leaf to a sub-chapter code (2a, 9h, ...) per manual rules.
  4. Look up sub-chapter tier in the 15-tier table.
  5. Load the corpus and reconstruct each doc's text from its `chunks[]`.
  6. Re-chunk / re-segment each doc.
  7. train/val/test split.
  8. Emit pairs (train/val = base logic; test = semantic-similarity logic).
  9. Write JSONL: one file per split in OUT_DIR.

Outputs (in OUT_DIR):
  directional_train.jsonl
  directional_val.jsonl
  directional_test.jsonl
  directional_manifest.json   (counts + decisions)
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
from typing import Iterable


# ---------------------------------------------------------------------------
# Semantic similarity (all-MiniLM-L6-v2 via HF transformers, mean pooling)
# ---------------------------------------------------------------------------
SIM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class _Embedder:
    """Mean-pooled sentence embeddings using HF transformers (no
    sentence-transformers dependency). Embeddings are L2-normalised so cosine
    similarity is a plain dot product."""

    def __init__(self, model_name: str = SIM_MODEL_NAME, batch_size: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sim] loading {model_name} on {self.device}", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _mean_pool(self, last_hidden, attn_mask):
        torch = self._torch
        mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str]):
        """Return an (N, D) float32 normalised embedding tensor on CPU."""
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tok(
                    batch, padding=True, truncation=True, max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                emb = self._mean_pool(hidden, enc["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.append(emb.cpu())
                print(f"\r[sim] embedded {min(i + self.batch_size, len(texts))}/{len(texts)}",
                      end="", file=sys.stderr)
        print("", file=sys.stderr)
        return torch.cat(out, dim=0) if out else torch.empty(0)


def load_match_counts(path: Path) -> dict[tuple[int, int], dict[int, int]]:
    """Read a reference test JSONL and return per-tier-pair label counts.

    Key is (t_low, t_hi) with t_low < t_hi; value is {0: n0, 1: n1}.
    """
    counts: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lo, hi = sorted((int(d["tier_1"]), int(d["tier_2"])))
            counts[(lo, hi)][int(d["label"])] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# 15-tier table (sub_chapter_code -> tier index)
# ---------------------------------------------------------------------------
SUBCHAPTER_TO_TIER: dict[str, int] = {
    # T1 Product orientation
    "2a": 1, "2b": 1,
    # T2 Admin floor
    "17a": 2, "17c": 2,
    # T3 Data foundations
    "14a": 3, "14b": 3, "14c": 3,
    # T4 Profiles, audiences & subscriptions
    "8a": 4, "8b": 4, "8c": 4,
    # T5 Content authoring foundations
    "10a": 5, "10b": 5, "10c": 5, "10d": 5, "10e": 5, "10f": 5, "10g": 5,
    # T6 Channels + campaigns (first sends)
    "9a": 6, "9b": 6, "9c": 6, "9d": 6, "9e": 6,
    "9f": 6, "9g": 6, "9h": 6, "9i": 6, "9j": 6, "9k": 6,
    "4a": 6, "4b": 6, "4c": 6,
    # T7 Journeys (core)
    "5a": 7, "5b": 7,
    # T8 Personalization & decisioning
    "11a": 8, "11b": 8, "13a": 8, "13b": 8, "13c": 8,
    # T9 Advanced journey patterns & experimentation
    "5c": 9, "12a": 9, "12b": 9,
    # T10 Multi-journey orchestration
    "7a": 10, "7b": 10,
    # T11 Admin configuration
    "16a": 11, "16b": 11, "16c": 11, "17b": 11,
    # T12 Governance & privacy
    "18a": 12, "18b": 12, "18c": 12,
    # T13 Observability & reporting
    "15a": 13, "15b": 13, "15c": 13, "15d": 13, "15e": 13,
    # T14 AI agents & assistants
    "19a": 14, "19b": 14, "2c": 14,
    # T15 Use cases, labs, capstones
    "6": 15, "20a": 15, "20b": 15, "20c": 15, "20d": 15,
    "21a": 15, "21b": 15, "22": 15,
}


# ---------------------------------------------------------------------------
# Sub-chapter assignment rules
# ---------------------------------------------------------------------------
# For chapters whose TOC already names sub-chapters, the sub-chapter anchor
# resolves directly to a sub-chapter code. For flat chapters, we map by slug.
#
# Keys below are the TOC chapter's top-level anchor (from `{#...}`).
# Values are either:
#   - {"by_sub_anchor": {sub_anchor: code, ...}, "by_direct_slug": {slug: code, ...}}
#     (for chapters with TOC-named sub-chapters + some direct leaves)
#   - {"by_slug": {slug: code, ...}}
#     (for flat chapters where leaves are directly under the chapter)

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
            # Two of Ch 8's leaves are external (web SDK, evaluate-on-demand)
            # and resolve to non-/tutorials/ URLs; handled via absolute-url rules
            # below if they land in the corpus.
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
        # All 4 leaves are absolute external URLs; handled via BY_ABSOLUTE_URL below.
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
        # Orphan leaf at chapter level (Mobile App Opt overview)
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
        # Orphan web-offers tutorial is an external absolute URL -> handled below.
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

# Absolute-URL overrides (TOC entries that are full http(s) URLs, not /help/*.md)
BY_ABSOLUTE_URL: dict[str, str] = {
    # Ch 4 Campaigns -> Action Campaigns (one absolute link to a code-based campaign)
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/channels/code-based-experience-channel/create-a-code-based-experience-campaign": "4a",

    # Ch 8 Profiles — web SDK tutorial bundle + audience on-demand
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/create-audiences-using-web-sdk/introduction": "8b",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/audiences/audience-builder/evaluate-audiences-on-demand": "8b",

    # Ch 10 Content Mgmt — fragments leaf
    # (all fragment leaves are relative — no absolute URL cases)

    # Ch 13 Decision Capabilities — external leaves
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/trigger-journey-on-form-submission/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-real-time-weather-data/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-ranking-formulas-based-on-user-zip-code-and-income/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/use-decisioning-in-email-channel/introduction": "13a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/use-decisioning-to-personalize-web-offers/introduction": "13c",
    # decisioning-in-push-notifications, use-decisioning-in-an-sms-message are relative

    # Ch 19 AI Assistant — all 4 are absolute tutorial URLs
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/introduction-to-journey-optimizer/ai-assistant": "19a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/content-management/ai-assistant/ai-assistant-for-content-generation-overview": "19a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/journeys/journey-agent-overview": "19b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/tutorials/experimentation/experimentation-agent-overview": "19b",

    # Ch 20 Use cases — external
    "https://experienceleague.adobe.com/en/docs/experience-platform/rtcdp/use-cases/personalization-insights-engagement/use-cases-luma": "20a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-real-time-weather-data/introduction": "20b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/personalizing-offers-with-ranking-formulas-based-on-user-zip-code-and-income/introduction": "20b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/scaling-orchestration-to-omnichannel-engagement/introduction.md": "20c",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/use-case-playbooks/overview": "20d",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/use-case-playbooks/configure-a-playbook-sandbox": "20d",
    "https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/use-case-playbooks/create-and-publish-a-playbook-instance": "20d",

    # Ch 21 Exercises
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/configure-a-training-sandbox/introduction-and-prerequisites": "21a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/challenges/introduction-and-prerequisites": "21a",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/build-personalized-mobile-moments/lab-overview": "21b",
    "https://experienceleague.adobe.com/en/docs/journey-optimizer-learn/scaling-orchestration-to-omnichannel-engagement/introduction": "21b",
}


# ---------------------------------------------------------------------------
# TOC parsing
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
    # Indent-based tree walker
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
    stack: list[tuple[int, str, str | None]] = []  # (indent, name, anchor)
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
# URL resolution
# ---------------------------------------------------------------------------
def resolve_url(leaf: dict) -> str | None:
    """Return the expected corpus URL for a leaf, or None if absolute or unresolvable."""
    if leaf["absolute"]:
        return leaf["absolute"]
    ch = leaf["chapter_anchor"]
    slug = leaf["slug"]
    if not ch or not slug:
        return None
    parts: list[str] = [ch]
    if leaf["sub_anchor"]:
        parts.append(leaf["sub_anchor"])
    if leaf["sub_sub_anchor"]:
        parts.append(leaf["sub_sub_anchor"])
    parts.append(slug)
    return f"{BASE_URL}/{'/'.join(parts)}"


def url_candidates(leaf: dict) -> list[str]:
    """Candidate URLs, longest path first, to try against the corpus set."""
    if leaf["absolute"]:
        return [leaf["absolute"], leaf["absolute"].rstrip("/")]
    ch = leaf["chapter_anchor"]
    slug = leaf["slug"]
    if not ch or not slug:
        return []
    depths: list[list[str]] = []
    # Most specific first
    full = [ch]
    if leaf["sub_anchor"]:
        full.append(leaf["sub_anchor"])
    if leaf["sub_sub_anchor"]:
        full.append(leaf["sub_sub_anchor"])
    for L in range(len(full), 0, -1):
        depths.append(full[:L] + [slug])
    return [f"{BASE_URL}/{'/'.join(p)}" for p in depths]


# ---------------------------------------------------------------------------
# Sub-chapter assignment
# ---------------------------------------------------------------------------
def assign_subchapter(leaf: dict) -> str | None:
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
# Corpus loading
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")


def _url_fraction(text: str) -> float:
    """Return the fraction of characters in *text* that belong to URLs."""
    url_chars = sum(len(m.group()) for m in _URL_RE.finditer(text))
    return url_chars / max(len(text), 1)


def load_corpus(path: Path, max_url_fraction: float = 0.3) -> dict[str, dict]:
    """Map sourceUrl -> {'text': concatenated chunks, 'title': ..., 'n_chunks': int}.

    Chunks where URLs account for more than *max_url_fraction* of the text are
    skipped — they are raw schema/reference tables (e.g. XDM field-dictionary),
    not prose, and produce noisy training segments.
    """
    with path.open() as f:
        data = json.load(f)
    out: dict[str, dict] = {}
    for group in data:
        if not group:
            continue
        # Each group is a list of chunk-entries sharing a sourceUrl
        url = group[0]["metadata"].get("sourceUrl")
        if not url:
            continue
        title = group[0]["metadata"].get("title", "")
        # Corpus chunks[] holds chunk texts. Concat in the order stored.
        texts: list[str] = []
        for entry in group:
            for ch in entry.get("chunks", []) or []:
                # Corpus schema: chunk prose lives in `data`. `text`/`content` not used.
                t = ch.get("data") or ch.get("text") or ch.get("content") or ""
                if t and _url_fraction(t) <= max_url_fraction:
                    texts.append(t)
        body = "\n\n".join(texts).strip()
        if not body:
            # Fallback: stitch from top-level title
            body = title
        if url in out:
            # Duplicate URL across groups — keep the longer body
            if len(body) > len(out[url]["text"]):
                out[url] = {"text": body, "title": title, "n_chunks": len(texts)}
        else:
            out[url] = {"text": body, "title": title, "n_chunks": len(texts)}
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, tokenizer, window: int, stride: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(ids):
        window_ids = ids[i : i + window]
        chunk = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if i + window >= len(ids):
            break
        i += stride
    return chunks


def split_paragraphs(text: str, tokenizer, min_tokens: int = 20) -> list[str]:
    """Split text on blank lines (\\n\\n+), keep paragraphs with >= min_tokens."""
    raw = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        n_tok = len(tokenizer.encode(p, add_special_tokens=False))
        if n_tok >= min_tokens:
            paragraphs.append(p)
    return paragraphs


# Lightweight sentence splitter — handles common abbreviations and single-letter
# initials (e.g. "U.S.") to avoid false splits.  Uses a simple approach: split
# on ". " (or "! " / "? ") where the next character is uppercase.
_SENT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)


def split_sentences(
    text: str,
    min_sentences: int = 2,
    max_sentences: int = 5,
) -> list[str]:
    """Split text into segments of consecutive sentences.

    1. Sentence-tokenize the full text with a regex splitter.
    2. Greedily pack consecutive sentences into a segment up to *max_sentences*.
    3. If the final segment has fewer than *min_sentences* sentences, merge it
       into the previous segment (avoids tiny trailing fragments).

    Returns a list of segment strings (joined with spaces).
    """
    # First flatten newlines so sentence splitting works on a clean stream.
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in _SENT_RE.split(clean) if s.strip()]
    if not sentences:
        return []

    # Pack into segments of up to max_sentences consecutive sentences.
    segments: list[list[str]] = []
    buf: list[str] = []
    for s in sentences:
        buf.append(s)
        if len(buf) >= max_sentences:
            segments.append(buf)
            buf = []
    if buf:
        segments.append(buf)

    # Merge a short trailing segment into the previous one.
    if len(segments) > 1 and len(segments[-1]) < min_sentences:
        segments[-2].extend(segments.pop())

    return [" ".join(seg) for seg in segments]


# ---------------------------------------------------------------------------
# Doc-level split
# ---------------------------------------------------------------------------
def split_docs(
    urls: list[str], val_frac: float, test_frac: float, seed: int
) -> dict[str, str]:
    """Return url -> split_name ('train'|'val'|'test'). Stratified within caller."""
    rng = random.Random(seed)
    shuffled = list(urls)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n - n_test >= 2 else 0
    test_urls = set(shuffled[:n_test])
    val_urls = set(shuffled[n_test : n_test + n_val])
    out: dict[str, str] = {}
    for u in shuffled:
        if u in test_urls:
            out[u] = "test"
        elif u in val_urls:
            out[u] = "val"
        else:
            out[u] = "train"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="aep_docs_collection_v_24-03-2026.json")
    ap.add_argument("--assignments", default=None,
                    help="Path to a pre-resolved assignments JSON "
                         "({url, subchapter, tier} list, or wrapped "
                         "{'assignments': [...]} envelope). "
                         "When provided, the TOC fetch/parse step is skipped.")
    ap.add_argument("--out-dir", default="classification/data")
    ap.add_argument("--tokenizer", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument(
        "--unit",
        choices=["chunk", "paragraph", "sentence"],
        default="chunk",
        help="Segmentation unit. "
             "chunk = fixed-size sliding window (controlled by --window/--stride). "
             "paragraph = split on blank lines; keeps paragraphs with >= "
             "--min-paragraph-tokens tokens. "
             "sentence = pack consecutive sentences into segments of "
             "--min-sentences to --max-sentences sentences.",
    )
    ap.add_argument(
        "--min-paragraph-tokens",
        type=int,
        default=20,
        help="Minimum token count to keep a paragraph (only used with --unit paragraph).",
    )
    ap.add_argument(
        "--min-sentences",
        type=int,
        default=2,
        help="Minimum sentences per segment (only used with --unit sentence). "
             "A trailing segment shorter than this is merged into the previous one.",
    )
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="Maximum sentences per segment (only used with --unit sentence).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument(
        "--pair-scope",
        choices=["adjacent", "all"],
        default="adjacent",
        help="adjacent = only tier i<->i+1; all = every cross-tier pair.",
    )
    ap.add_argument(
        "--single-direction",
        action="store_true",
        default=False,
        help="For each segment pair, randomly emit only ONE direction (forward or "
             "reverse) instead of both. Halves dataset size and prevents the model "
             "from seeing every (A,B) with a guaranteed (B,A) in the same split.",
    )
    ap.add_argument(
        "--split-level",
        choices=["sub_chapter", "tier", "chunk_tier"],
        default="sub_chapter",
        help="Splitting unit for 80/10/10. "
             "sub_chapter = doc-level split within each sub-chapter. "
             "tier = doc-level split pooled per tier. "
             "chunk_tier = CHUNK-level split pooled per tier (every doc's "
             "chunks distributed across train/val/test). Requires non-overlapping "
             "chunks (stride >= window) to avoid token leakage; a warning is "
             "emitted otherwise.",
    )
    ap.add_argument(
        "--max-doc-pairs-per-tier-pair",
        type=int,
        default=None,
        help="Cap on doc-pairs per (tier_i, tier_j) per split (doc-level "
             "splits only). Each surviving doc-pair still contributes its full "
             "chunk cartesian product. None = uncapped.",
    )
    ap.add_argument(
        "--max-chunk-pairs-per-tier-pair",
        type=int,
        default=None,
        help="Cap on chunk-pair SAMPLES per (tier_i, tier_j) per split "
             "(chunk_tier split only). Samples include both forward (label=1) "
             "and reverse (label=0) directions, so cap=3000 means at most 1500 "
             "(c_low, c_hi) tuples survive per tier-pair. None = uncapped.",
    )
    ap.add_argument(
        "--cap-splits",
        choices=["train", "all"],
        default="train",
        help="Which splits to apply the tier-pair cap to. "
             "train = only cap the train split (val/test uncapped). "
             "all = cap train, val, and test with the same cap value.",
    )
    ap.add_argument("--cap-seed", type=int, default=42,
                    help="Seed for deterministic doc-pair subsampling when cap bites.")
    ap.add_argument(
        "--semantic-test",
        action="store_true",
        default=False,
        help="Build the TEST split by semantic similarity instead of the random "
             "direction coin-flip. Requires --match-counts. Most-similar "
             "cross-tier chunk-pairs become label 0 (reverse); least-similar "
             "become label 1 (forward). Only valid with --split-level chunk_tier "
             "and --single-direction.",
    )
    ap.add_argument(
        "--match-counts",
        default=None,
        help="Path to a reference test JSONL whose per-tier-pair (label0, label1) "
             "counts the semantic test split must reproduce exactly. "
             "Required when --semantic-test is set.",
    )
    ap.add_argument(
        "--sim-model",
        default=SIM_MODEL_NAME,
        help="HF model id used for semantic-similarity scoring of the test split.",
    )
    args = ap.parse_args()

    if args.semantic_test:
        if args.split_level != "chunk_tier":
            ap.error("--semantic-test requires --split-level chunk_tier")
        if not args.single_direction:
            ap.error("--semantic-test requires --single-direction")
        if not args.match_counts:
            ap.error("--semantic-test requires --match-counts <reference test jsonl>")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- load tokenizer lazily (allows --help to work without transformers) --
    from transformers import AutoTokenizer
    print(f"[tokenizer] loading {args.tokenizer}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # -- corpus --
    corpus_path = Path(args.corpus)
    print(f"[corpus] loading {corpus_path}", file=sys.stderr)
    corpus = load_corpus(corpus_path)
    print(f"[corpus] {len(corpus)} unique URLs", file=sys.stderr)

    # -- resolve docs: assignments file (skip TOC) or AJO TOC --
    resolved: list[dict] = []
    url_misses: list[dict] = []
    subchapter_misses: list[dict] = []

    if args.assignments:
        # ----- pre-resolved assignments path (AEP and other products) -----
        print(f"[assignments] loading {args.assignments}", file=sys.stderr)
        with open(args.assignments) as _f:
            _data = json.load(_f)
        _items = _data["assignments"] if isinstance(_data, dict) and "assignments" in _data else _data
        _seen: set[str] = set()
        for it in _items:
            # Normalise /./ path segments that crawlers sometimes leave in URLs
            raw_url: str = it["url"]
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
            if not sub or tier is None:
                subchapter_misses.append({"url": raw_url, "reason": "missing subchapter/tier"})
                continue
            resolved.append({
                "url": raw_url,
                "title": corpus[raw_url]["title"] or it.get("title", ""),
                "subchapter": sub,
                "tier": int(tier),
            })
        print(f"[assignments] {len(resolved)} docs resolved, "
              f"{len(url_misses)} URL misses, {len(subchapter_misses)} subchapter misses",
              file=sys.stderr)
    else:
        # ----- AJO TOC path (original behaviour) -----
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

        # Deduplicate URLs (e.g., same URL referenced from two TOC locations)
        by_url: dict[str, dict] = {}
        for r in resolved:
            if r["url"] not in by_url:
                by_url[r["url"]] = r
        resolved = list(by_url.values())

        print(f"[resolve] {len(resolved)} docs resolved, {len(url_misses)} URL misses, "
              f"{len(subchapter_misses)} subchapter misses", file=sys.stderr)

    # Summary per tier
    tier_docs: dict[int, list[dict]] = defaultdict(list)
    for r in resolved:
        tier_docs[r["tier"]].append(r)
    print("[resolve] docs per tier:", file=sys.stderr)
    for t in sorted(tier_docs):
        subs = defaultdict(int)
        for d in tier_docs[t]:
            subs[d["subchapter"]] += 1
        subs_str = ", ".join(f"{k}:{v}" for k, v in sorted(subs.items()))
        print(f"           T{t:>2}  n={len(tier_docs[t]):>3}  [{subs_str}]", file=sys.stderr)

    # -- segment every doc (chunk, paragraph, or sentence) --
    doc_chunks: dict[str, list[str]] = {}
    if args.unit == "sentence":
        print(f"[segment] unit=sentence  min_sentences={args.min_sentences}  max_sentences={args.max_sentences}", file=sys.stderr)
        for r in resolved:
            body = corpus[r["url"]]["text"]
            doc_chunks[r["url"]] = split_sentences(body, args.min_sentences, args.max_sentences)
    elif args.unit == "paragraph":
        print(f"[segment] unit=paragraph  min_tokens={args.min_paragraph_tokens}", file=sys.stderr)
        for r in resolved:
            body = corpus[r["url"]]["text"]
            doc_chunks[r["url"]] = split_paragraphs(body, tokenizer, args.min_paragraph_tokens)
    else:
        print(f"[segment] unit=chunk  window={args.window} stride={args.stride}", file=sys.stderr)
        if args.split_level == "chunk_tier" and args.stride < args.window:
            print(f"[segment] WARNING split_level=chunk_tier with stride {args.stride} < "
                  f"window {args.window}: adjacent chunks from the same doc share "
                  f"{args.window - args.stride} tokens, which will cause token-level "
                  f"leakage across splits. Use --stride == --window for clean splits.",
                  file=sys.stderr)
        for r in resolved:
            body = corpus[r["url"]]["text"]
            doc_chunks[r["url"]] = chunk_text(body, tokenizer, args.window, args.stride)
    total_chunks = sum(len(c) for c in doc_chunks.values())
    print(f"[segment] produced {total_chunks} segments across {len(doc_chunks)} docs", file=sys.stderr)

    # -- split assignment --
    # sub_chapter, tier: doc-level split (every chunk of a doc inherits the doc's split)
    # chunk_tier: chunk-level split, pooled per tier
    doc_splits: dict[str, str] = {}  # populated only in doc-level modes
    split_counts: dict[str, int] = defaultdict(int)

    # chunk_entries[tier] = list of {"text", "url", "chunk_idx", "subchapter", "split"}
    chunk_entries_by_tier: dict[int, list[dict]] = defaultdict(list)

    if args.split_level in ("sub_chapter", "tier"):
        by_group: dict[str, list[str]] = defaultdict(list)
        for r in resolved:
            key = r["subchapter"] if args.split_level == "sub_chapter" else f"T{r['tier']}"
            by_group[key].append(r["url"])
        print(f"[split] split_level={args.split_level}, {len(by_group)} groups", file=sys.stderr)
        for key, urls in by_group.items():
            s = split_docs(urls, args.val_frac, args.test_frac, args.seed + hash(key) % 10000)
            doc_splits.update(s)
        for sp in doc_splits.values():
            split_counts[sp] += 1
        print(f"[split] docs per split: {dict(split_counts)}", file=sys.stderr)
    else:
        # chunk_tier: pool chunks per tier, shuffle, split 80/10/10
        print(f"[split] split_level=chunk_tier, splitting chunks per tier", file=sys.stderr)
        chunk_split_counts_per_tier: dict[int, dict[str, int]] = {}
        for t in sorted(tier_docs.keys()):
            entries: list[dict] = []
            for r in tier_docs[t]:
                url = r["url"]
                for i, ct in enumerate(doc_chunks[url]):
                    entries.append({
                        "text": ct, "url": url, "chunk_idx": i,
                        "subchapter": r["subchapter"],
                    })
            rng = random.Random(args.seed + hash(f"T{t}") % 10000)
            rng.shuffle(entries)
            n = len(entries)
            n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
            n_val = max(1, round(n * args.val_frac)) if n - n_test >= 2 else 0
            for i, e in enumerate(entries):
                if i < n_test:
                    e["split"] = "test"
                elif i < n_test + n_val:
                    e["split"] = "val"
                else:
                    e["split"] = "train"
            chunk_entries_by_tier[t] = entries
            chunk_split_counts_per_tier[t] = {
                "train": sum(1 for e in entries if e["split"] == "train"),
                "val":   sum(1 for e in entries if e["split"] == "val"),
                "test":  sum(1 for e in entries if e["split"] == "test"),
            }
            for e in entries:
                split_counts[e["split"]] += 1
        print(f"[split] chunks per split: {dict(split_counts)}", file=sys.stderr)
        print(f"[split] chunks per tier per split:", file=sys.stderr)
        for t in sorted(chunk_split_counts_per_tier):
            c = chunk_split_counts_per_tier[t]
            print(f"           T{t:>2}  train={c['train']:>4d}  val={c['val']:>3d}  test={c['test']:>3d}", file=sys.stderr)

    # -- emit pairs --
    tiers_sorted = sorted(tier_docs.keys())
    if args.pair_scope == "adjacent":
        tier_pair_iter = [(tiers_sorted[i], tiers_sorted[i + 1]) for i in range(len(tiers_sorted) - 1)]
    else:
        tier_pair_iter = [
            (tiers_sorted[i], tiers_sorted[j])
            for i in range(len(tiers_sorted))
            for j in range(i + 1, len(tiers_sorted))
        ]
    print(f"[pairs] emitting for {len(tier_pair_iter)} tier-pair(s) in {args.pair_scope} mode", file=sys.stderr)

    # Cap-seed utility: deterministic per-key RNG so re-runs with the same
    # cap_seed give identical selections.
    cap_splits = {"train"} if args.cap_splits == "train" else {"train", "val", "test"}

    # Common state for manifest.
    pre_cap_counts: dict[str, int] = {}
    post_cap_counts: dict[str, int] = {}
    chunk_pairs_by_key: dict[str, int] = defaultdict(int)

    writers = {sp: (out_dir / f"directional_{sp}.jsonl").open("w") for sp in ("train", "val", "test")}
    counts: dict[str, int] = defaultdict(int)
    direction_rng = random.Random(args.seed + 7)  # separate RNG for direction coin-flip

    def _write_pair(split: str, c_low_text: str, c_hi_text: str,
                    t_low: int, t_hi: int,
                    sub_low: str, sub_hi: str,
                    url_low: str, url_hi: str, tag: str) -> None:
        if args.single_direction:
            # Randomly pick one direction
            if direction_rng.random() < 0.5:
                # Forward: low-tier first -> label 1
                writers[split].write(json.dumps({
                    "text_1": c_low_text, "text_2": c_hi_text, "label": 1,
                    "tier_1": t_low, "tier_2": t_hi,
                    "sub_1": sub_low, "sub_2": sub_hi,
                    "url_1": url_low, "url_2": url_hi,
                }) + "\n")
            else:
                # Reverse: high-tier first -> label 0
                writers[split].write(json.dumps({
                    "text_1": c_hi_text, "text_2": c_low_text, "label": 0,
                    "tier_1": t_hi, "tier_2": t_low,
                    "sub_1": sub_hi, "sub_2": sub_low,
                    "url_1": url_hi, "url_2": url_low,
                }) + "\n")
            counts[split] += 1
            chunk_pairs_by_key[tag] += 1
        else:
            # Both directions (original behavior)
            writers[split].write(json.dumps({
                "text_1": c_low_text, "text_2": c_hi_text, "label": 1,
                "tier_1": t_low, "tier_2": t_hi,
                "sub_1": sub_low, "sub_2": sub_hi,
                "url_1": url_low, "url_2": url_hi,
            }) + "\n")
            writers[split].write(json.dumps({
                "text_1": c_hi_text, "text_2": c_low_text, "label": 0,
                "tier_1": t_hi, "tier_2": t_low,
                "sub_1": sub_hi, "sub_2": sub_low,
                "url_1": url_hi, "url_2": url_low,
            }) + "\n")
            counts[split] += 2
            chunk_pairs_by_key[tag] += 2

    try:
        if args.split_level in ("sub_chapter", "tier"):
            # ---- doc-level split: cap on doc-pairs, then chunk cartesian product ----
            doc_pairs_by_key: dict[tuple[int, int, str], list[tuple[dict, dict]]] = defaultdict(list)
            for t_low, t_hi in tier_pair_iter:
                for d_low in tier_docs[t_low]:
                    sp_low = doc_splits[d_low["url"]]
                    if not doc_chunks.get(d_low["url"]):
                        continue
                    for d_hi in tier_docs[t_hi]:
                        sp_hi = doc_splits[d_hi["url"]]
                        if sp_low != sp_hi:
                            continue
                        if not doc_chunks.get(d_hi["url"]):
                            continue
                        doc_pairs_by_key[(t_low, t_hi, sp_low)].append((d_low, d_hi))

            for (tl, th, sp), lst in doc_pairs_by_key.items():
                pre_cap_counts[f"T{tl}-T{th}/{sp}"] = len(lst)

            cap = args.max_doc_pairs_per_tier_pair
            if cap is not None:
                for key, pairs in list(doc_pairs_by_key.items()):
                    t_low, t_hi, split = key
                    if split not in cap_splits:
                        continue
                    if len(pairs) > cap:
                        local_rng = random.Random((args.cap_seed, t_low, t_hi, split).__hash__())
                        shuf = list(pairs)
                        local_rng.shuffle(shuf)
                        doc_pairs_by_key[key] = shuf[:cap]

            for (tl, th, sp), lst in doc_pairs_by_key.items():
                post_cap_counts[f"T{tl}-T{th}/{sp}"] = len(lst)

            print("[pairs] doc-pair counts per tier-pair per split (pre -> post cap):", file=sys.stderr)
            for key in sorted(doc_pairs_by_key.keys()):
                tl, th, sp = key
                tag = f"T{tl}-T{th}/{sp}"
                pre = pre_cap_counts.get(tag, 0)
                post = post_cap_counts.get(tag, 0)
                marker = "*" if pre != post else " "
                print(f"        {marker} {tag:>14s}  pre={pre:>5d}  post={post:>5d}", file=sys.stderr)

            for (t_low, t_hi, split), pairs in doc_pairs_by_key.items():
                tag = f"T{t_low}-T{t_hi}/{split}"
                for d_low, d_hi in pairs:
                    chunks_low = doc_chunks[d_low["url"]]
                    chunks_hi = doc_chunks[d_hi["url"]]
                    for c_low in chunks_low:
                        for c_hi in chunks_hi:
                            _write_pair(split, c_low, c_hi, t_low, t_hi,
                                        d_low["subchapter"], d_hi["subchapter"],
                                        d_low["url"], d_hi["url"], tag)

        else:
            # ---- chunk_tier: cap on chunk-pair SAMPLES (fwd+rev counted together) ----
            # For each adjacent tier-pair and split, take cartesian product of
            # same-split chunks, optionally cap, then emit fwd+rev pairs.
            chunk_pairs_by_key_list: dict[tuple[int, int, str], list[tuple[dict, dict]]] = defaultdict(list)
            # Index chunks per (tier, split) for quick lookup.
            chunks_by_ts: dict[tuple[int, str], list[dict]] = defaultdict(list)
            for t, entries in chunk_entries_by_tier.items():
                for e in entries:
                    chunks_by_ts[(t, e["split"])].append(e)

            for t_low, t_hi in tier_pair_iter:
                for sp in ("train", "val", "test"):
                    lows = chunks_by_ts.get((t_low, sp), [])
                    highs = chunks_by_ts.get((t_hi, sp), [])
                    if not lows or not highs:
                        continue
                    for c_low in lows:
                        for c_hi in highs:
                            chunk_pairs_by_key_list[(t_low, t_hi, sp)].append((c_low, c_hi))

            samples_per_tuple = 1 if args.single_direction else 2

            for (tl, th, sp), lst in chunk_pairs_by_key_list.items():
                pre_cap_counts[f"T{tl}-T{th}/{sp}"] = len(lst) * samples_per_tuple

            cap = args.max_chunk_pairs_per_tier_pair
            if cap is not None:
                # cap is on SAMPLES; convert to max tuples — same cap for all splits
                target_tuples = cap if args.single_direction else cap // 2
                for key, pairs in list(chunk_pairs_by_key_list.items()):
                    t_low, t_hi, split = key
                    if split not in cap_splits:
                        continue
                    if len(pairs) > target_tuples:
                        local_rng = random.Random((args.cap_seed, t_low, t_hi, split).__hash__())
                        shuf = list(pairs)
                        local_rng.shuffle(shuf)
                        chunk_pairs_by_key_list[key] = shuf[:target_tuples]

            for (tl, th, sp), lst in chunk_pairs_by_key_list.items():
                post_cap_counts[f"T{tl}-T{th}/{sp}"] = len(lst) * samples_per_tuple

            dir_label = "single-dir" if args.single_direction else "fwd+rev"
            print(f"[pairs] chunk-pair sample counts per tier-pair per split "
                  f"(pre -> post cap, {dir_label}):", file=sys.stderr)
            for key in sorted(chunk_pairs_by_key_list.keys()):
                tl, th, sp = key
                tag = f"T{tl}-T{th}/{sp}"
                pre = pre_cap_counts.get(tag, 0)
                post = post_cap_counts.get(tag, 0)
                marker = "*" if pre != post else " "
                print(f"        {marker} {tag:>14s}  pre={pre:>6d}  post={post:>6d}", file=sys.stderr)

            # ---- load semantic-test machinery once, if requested ----
            match_counts: dict[tuple[int, int], dict[int, int]] = {}
            embedder = None
            sim_emit_log: list[str] = []
            if args.semantic_test:
                match_counts = load_match_counts(Path(args.match_counts))
                want0 = sum(c[0] for c in match_counts.values())
                want1 = sum(c[1] for c in match_counts.values())
                print(f"[sim] match-counts loaded from {args.match_counts}: "
                      f"{len(match_counts)} tier-pairs, want label0={want0} label1={want1}",
                      file=sys.stderr)
                embedder = _Embedder(args.sim_model)
                # Embed every test chunk once, cache by id(text-object) -> row idx.
                test_texts: list[str] = []
                test_index: dict[str, int] = {}
                for (t_low, t_hi, split), pairs in chunk_pairs_by_key_list.items():
                    if split != "test":
                        continue
                    for c_low, c_hi in pairs:
                        for c in (c_low, c_hi):
                            if c["text"] not in test_index:
                                test_index[c["text"]] = len(test_texts)
                                test_texts.append(c["text"])
                print(f"[sim] embedding {len(test_texts)} unique test chunks", file=sys.stderr)
                test_emb = embedder.encode(test_texts)

            def _emit_semantic_test(t_low: int, t_hi: int,
                                    pairs: list[tuple[dict, dict]]) -> None:
                """Rank cross-tier chunk-pairs by cosine sim; assign the n0
                most-similar to label 0 (reverse) and the n1 least-similar to
                label 1 (forward), matching the reference counts."""
                want = match_counts.get((t_low, t_hi), {0: 0, 1: 0})
                n0, n1 = want[0], want[1]
                tag = f"T{t_low}-T{t_hi}/test"
                if n0 + n1 == 0 or not pairs:
                    sim_emit_log.append(f"{tag:>16s}  pairs={len(pairs):>5d}  n0=0 n1=0 (skip)")
                    return
                # Cosine sim via cached normalised embeddings (dot product).
                idx_low = [test_index[c_low["text"]] for c_low, _ in pairs]
                idx_hi = [test_index[c_hi["text"]] for _, c_hi in pairs]
                e_low = test_emb[idx_low]
                e_hi = test_emb[idx_hi]
                sims = (e_low * e_hi).sum(dim=1)  # (len(pairs),)
                order = sims.argsort(descending=True).tolist()  # most -> least similar
                # Most-similar -> label 0 ; least-similar -> label 1.
                take0 = order[:n0]
                take1 = order[len(order) - n1:] if n1 > 0 else []
                for j in take0:
                    c_low, c_hi = pairs[j]
                    # label 0 == reverse direction (high tier first)
                    writers["test"].write(json.dumps({
                        "text_1": c_hi["text"], "text_2": c_low["text"], "label": 0,
                        "tier_1": t_hi, "tier_2": t_low,
                        "sub_1": c_hi["subchapter"], "sub_2": c_low["subchapter"],
                        "url_1": c_hi["url"], "url_2": c_low["url"],
                    }) + "\n")
                    counts["test"] += 1
                    chunk_pairs_by_key[tag] += 1
                for j in take1:
                    c_low, c_hi = pairs[j]
                    # label 1 == forward direction (low tier first)
                    writers["test"].write(json.dumps({
                        "text_1": c_low["text"], "text_2": c_hi["text"], "label": 1,
                        "tier_1": t_low, "tier_2": t_hi,
                        "sub_1": c_low["subchapter"], "sub_2": c_hi["subchapter"],
                        "url_1": c_low["url"], "url_2": c_hi["url"],
                    }) + "\n")
                    counts["test"] += 1
                    chunk_pairs_by_key[tag] += 1
                sim_emit_log.append(
                    f"{tag:>16s}  pairs={len(pairs):>5d}  n0={n0:>4d} n1={n1:>4d}  "
                    f"sim[min={sims.min():.3f} max={sims.max():.3f}]"
                )

            for (t_low, t_hi, split), pairs in chunk_pairs_by_key_list.items():
                tag = f"T{t_low}-T{t_hi}/{split}"
                if args.semantic_test and split == "test":
                    _emit_semantic_test(t_low, t_hi, pairs)
                    continue
                for c_low, c_hi in pairs:
                    _write_pair(split, c_low["text"], c_hi["text"], t_low, t_hi,
                                c_low["subchapter"], c_hi["subchapter"],
                                c_low["url"], c_hi["url"], tag)

            if args.semantic_test:
                print("[sim] per-tier-pair test emission:", file=sys.stderr)
                for ln in sorted(sim_emit_log):
                    print(f"        {ln}", file=sys.stderr)
    finally:
        for w in writers.values():
            w.close()

    # -- manifest --
    manifest = {
        "tier_table": SUBCHAPTER_TO_TIER,
        "unit": args.unit,
        "window": args.window if args.unit == "chunk" else None,
        "stride": args.stride if args.unit == "chunk" else None,
        "min_paragraph_tokens": args.min_paragraph_tokens if args.unit == "paragraph" else None,
        "min_sentences": args.min_sentences if args.unit == "sentence" else None,
        "max_sentences": args.max_sentences if args.unit == "sentence" else None,
        "tokenizer": args.tokenizer,
        "pair_scope": args.pair_scope,
        "single_direction": args.single_direction,
        "split_level": args.split_level,
        "max_doc_pairs_per_tier_pair": args.max_doc_pairs_per_tier_pair,
        "max_chunk_pairs_per_tier_pair": args.max_chunk_pairs_per_tier_pair,
        "cap_splits": args.cap_splits,
        "cap_seed": args.cap_seed,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_docs_resolved": len(resolved),
        "n_url_misses": len(url_misses),
        "n_subchapter_misses": len(subchapter_misses),
        "docs_per_tier": {t: len(tier_docs[t]) for t in tiers_sorted},
        "split_doc_counts": dict(split_counts),
        "sample_counts": dict(counts),
        # For doc-level splits these are doc-pair counts; for chunk_tier they are
        # chunk-pair sample counts (fwd+rev combined).
        "pre_cap_counts": pre_cap_counts,
        "post_cap_counts": post_cap_counts,
        "chunk_pairs_per_tier_pair": dict(chunk_pairs_by_key),
        "url_misses": url_misses,
        "subchapter_misses": subchapter_misses,
    }
    (out_dir / "directional_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote samples: {dict(counts)}", file=sys.stderr)
    print(f"[done] manifest: {out_dir/'directional_manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
