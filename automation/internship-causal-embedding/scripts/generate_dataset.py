#!/usr/bin/env python3
"""Automated tier-based directional classification dataset builder for ANY Adobe product.

End-to-end pipeline that turns a product's documentation (TOC + scraped corpus
JSON) into the final `directional_{train,val,test}.jsonl` dataset, using Claude
to do the *reasoning* part that used to be hand-written for Adobe Journey
Optimizer (the tier hierarchy + per-doc tier assignment).

Two phases (see DATASET_GENERATION_SYSTEM_PROMPT.md for the full spec):

  PHASE A  (Claude)  — infer the tier hierarchy and assign every corpus doc to a
                       tier + sub-chapter. Output: assignments.json + tier_hierarchy.md
  PHASE B  (code)    — drive the existing `build-classification-data` builder with
                       those assignments to segment / split / pair / write JSONL.

Usage:
  export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
  export AWS_BEDROCK_REGION=...  AWS_BEDROCK_ENDPOINT_URL=...  AWS_BEARER_TOKEN_BEDROCK=...

  python3 scripts/generate_dataset.py \
      --product "Adobe Express" \
      --corpus  path/to/express_docs_collection.json \
      --toc-file path/to/express_TOC.md \
      --out-dir data/express_causal_classification

  # TOC can also be fetched from a URL:
  #   --toc-url https://raw.githubusercontent.com/AdobeDocs/.../TOC.md

  # Stop after Phase A (inspect tiers before building the full dataset):
  #   --phase-a-only

Bedrock auth env vars (same as evaluation/llm_baseline.py):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BEDROCK_REGION,
  AWS_BEDROCK_ENDPOINT_URL, AWS_BEARER_TOKEN_BEDROCK
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# --- reuse the in-repo builder's TOC parser + corpus loader ---
# Load it by file path so we don't trigger the heavy followup_data package
# __init__ (which imports every dataset loader and their deps). The builder
# module itself only needs the stdlib at import time.
import importlib.util  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILDERS_DIR = _REPO_ROOT / "src" / "followup_data" / "builders"
_BUILDER_PATH = _BUILDERS_DIR / "build_classification_data.py"
_BUILDER_FROM_ASSIGNMENTS = _BUILDERS_DIR / "build_dataset_from_assignments.py"
_spec = importlib.util.spec_from_file_location("_bcd_builder", _BUILDER_PATH)
_bcd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bcd)
load_corpus = _bcd.load_corpus
parse_toc = _bcd.parse_toc

SYSTEM_PROMPT_FILE = _REPO_ROOT / "TIER_GENERATION_SYSTEM_PROMPT.md"

# The production ("v6") build configuration, proven on AJO. See README.
V6_BUILD_FLAGS = [
    "--unit", "sentence", "--min-sentences", "2", "--max-sentences", "5",
    "--pair-scope", "all", "--single-direction",
    "--max-chunk-pairs-per-tier-pair", "1500", "--cap-splits", "train",
    "--seed", "42",
]


# ---------------------------------------------------------------------------
# Bedrock client (same SigV4 + bearer pattern as evaluation/llm_baseline.py)
# ---------------------------------------------------------------------------
def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v or v == "unconfigured":
        raise EnvironmentError(f"missing required env var: {name}")
    return v


def call_claude(prompt: str, system: str, model_id: str,
                temperature: float = 0.0, max_tokens: int = 32000) -> str:
    import requests
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    region = _env("AWS_BEDROCK_REGION")
    endpoint = _env("AWS_BEDROCK_ENDPOINT_URL")
    bearer = _env("AWS_BEARER_TOKEN_BEDROCK")
    url = f"{endpoint}/model/{model_id}/invoke"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    })
    creds = Credentials(_env("AWS_ACCESS_KEY_ID"), _env("AWS_SECRET_ACCESS_KEY"))
    aws_request = AWSRequest(method="POST", url=url, data=body, headers={
        "Content-Type": "application/json", "Accept": "application/json",
    })
    SigV4Auth(creds, "bedrock", region).add_auth(aws_request)
    headers = dict(aws_request.headers)
    headers["x-api-key"] = bearer
    resp = requests.post(url, headers=headers, data=body)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _extract_json(text: str) -> dict:
    """Pull the first top-level JSON object out of an LLM response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Otherwise take from first { to its matching last }.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# Product-agnostic leaf -> corpus URL resolution (no BASE_URL assumptions)
# ---------------------------------------------------------------------------
def _last_segment(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def resolve_leaves(leaves: list[dict], corpus: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Match each TOC leaf to a corpus sourceUrl by slug suffix (or exact absolute).

    Returns (resolved_docs, url_misses). Each resolved doc carries the corpus
    URL, title, chapter, sub-anchor, and slug — everything Claude needs to reason
    about placement. This is intentionally BASE_URL-free so it works for any
    product whose corpus URLs end in the doc slug.
    """
    by_slug: dict[str, list[str]] = defaultdict(list)
    for u in corpus:
        by_slug[_last_segment(u)].append(u)

    resolved: list[dict] = []
    url_misses: list[dict] = []
    seen: set[str] = set()
    for leaf in leaves:
        url_found = None
        if leaf["absolute"]:
            for cand in (leaf["absolute"], leaf["absolute"].rstrip("/")):
                if cand in corpus:
                    url_found = cand
                    break
            if not url_found:
                # absolute URL might still match by its last segment
                cands = by_slug.get(_last_segment(leaf["absolute"]), [])
                url_found = cands[0] if len(cands) == 1 else None
        else:
            slug = leaf["slug"]
            cands = by_slug.get(slug or "", [])
            if len(cands) == 1:
                url_found = cands[0]
            elif len(cands) > 1:
                # disambiguate by chapter / sub anchor appearing in the path
                anchors = [a for a in (leaf["chapter_anchor"], leaf["sub_anchor"],
                                       leaf["sub_sub_anchor"]) if a]
                ranked = sorted(cands, key=lambda u: -sum(a in u for a in anchors))
                url_found = ranked[0]

        if not url_found:
            url_misses.append({"title": leaf["title"], "target": leaf["target"],
                               "chapter": leaf["chapter_name"]})
            continue
        if url_found in seen:
            continue
        seen.add(url_found)
        resolved.append({
            "url": url_found,
            "title": leaf["title"] or corpus[url_found]["title"],
            "chapter": leaf["chapter_name"],
            "chapter_anchor": leaf["chapter_anchor"],
            "sub_anchor": leaf["sub_anchor"],
            "slug": leaf["slug"],
        })
    return resolved, url_misses


# ---------------------------------------------------------------------------
# Phase A — Claude infers tiers + assignments
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    """The tier-generation spec (authoritative) + a short API-mode reminder.

    The full task and output schema live in TIER_GENERATION_SYSTEM_PROMPT.md so
    there is a single source of truth; we only append a brief note that this is a
    programmatic, single-shot call.
    """
    base = SYSTEM_PROMPT_FILE.read_text() if SYSTEM_PROMPT_FILE.exists() else ""
    api_note = """

---

# API mode

You are being called programmatically in a single shot — not as an interactive
agent. Do not use tools or ask questions. Return only the single JSON object
specified above: no prose, no explanation, no code fences.
"""
    return base + api_note


def build_user_message(product: str, docs: list[dict]) -> str:
    doc_lines = [
        {"url": d["url"], "title": d["title"], "chapter": d["chapter"],
         "sub_anchor": d["sub_anchor"], "slug": d["slug"]}
        for d in docs
    ]
    return (
        f"Product: {product}\n\n"
        f"Documents ({len(doc_lines)} total) as JSON:\n"
        f"{json.dumps(doc_lines, ensure_ascii=False, indent=0)}\n\n"
        f"Return the single JSON object per the OUTPUT CONTRACT."
    )


def assign_docs(product: str, docs: list[dict], model_id: str,
                max_tokens: int, temperature: float) -> dict:
    """Single Claude call: infer tiers + assign all docs. Returns parsed JSON."""
    user = build_user_message(product, docs)
    raw = call_claude(user, build_system_prompt(), model_id, temperature, max_tokens)
    return _extract_json(raw)


def validate_phase_a(config: dict, resolved: list[dict]) -> list[str]:
    """Return a list of problems (empty == valid)."""
    problems: list[str] = []
    tiers = {t["id"] for t in config.get("tiers", [])}
    if not tiers:
        problems.append("no tiers returned")
    elif sorted(tiers) != list(range(1, len(tiers) + 1)):
        problems.append(f"tier ids not contiguous 1..N: {sorted(tiers)}")

    sub_to_tier = {s["code"]: s["tier"] for s in config.get("subchapters", [])}
    assigned_urls = set()
    for a in config.get("assignments", []):
        assigned_urls.add(a["url"])
        sub, tier = a.get("subchapter"), a.get("tier")
        if sub not in sub_to_tier:
            problems.append(f"assignment uses unknown subchapter {sub!r} ({a['url']})")
        elif sub_to_tier[sub] != tier:
            problems.append(
                f"assignment tier {tier} != subchapter {sub} tier {sub_to_tier[sub]} ({a['url']})")
        if tier not in tiers:
            problems.append(f"assignment tier {tier} not in tier table ({a['url']})")

    resolved_urls = {d["url"] for d in resolved}
    missing = resolved_urls - assigned_urls
    extra = assigned_urls - resolved_urls
    if missing:
        problems.append(f"{len(missing)} docs were not assigned (coverage gap): "
                        f"{list(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    if extra:
        problems.append(f"{len(extra)} assignments reference unknown urls: "
                        f"{list(extra)[:5]}{' ...' if len(extra) > 5 else ''}")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", required=True, help='e.g. "Adobe Express"')
    ap.add_argument("--corpus", required=True, help="scraped corpus JSON")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--toc-file", help="local TOC.md path")
    src.add_argument("--toc-url", help="raw URL to fetch TOC.md")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="us.anthropic.claude-sonnet-4-20250514-v1:0")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--phase-a-only", action="store_true",
                    help="stop after writing assignments + tier_hierarchy (no JSONL build)")
    ap.add_argument("--allow-coverage-gaps", action="store_true",
                    help="proceed to Phase B even if some docs went unassigned")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble and dump the exact system prompt + user message + "
                         "Bedrock request body that WOULD be sent to Claude, then exit. "
                         "Needs no AWS credentials — for inspecting Phase A prompt generation.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- load inputs --
    print(f"[corpus] loading {args.corpus}", file=sys.stderr)
    corpus = load_corpus(Path(args.corpus))
    print(f"[corpus] {len(corpus)} unique URLs", file=sys.stderr)

    if args.toc_file:
        toc_text = Path(args.toc_file).read_text()
    else:
        print(f"[toc] fetching {args.toc_url}", file=sys.stderr)
        with urllib.request.urlopen(args.toc_url, timeout=30) as r:
            toc_text = r.read().decode("utf-8")
    leaves = parse_toc(toc_text)
    print(f"[toc] parsed {len(leaves)} leaves", file=sys.stderr)

    # -- resolve leaves -> corpus docs --
    resolved, url_misses = resolve_leaves(leaves, corpus)
    print(f"[resolve] {len(resolved)} docs matched to corpus, "
          f"{len(url_misses)} url misses", file=sys.stderr)
    if not resolved:
        sys.exit("[fatal] no TOC leaves resolved to corpus URLs — check inputs.")

    # -- DRY RUN: dump the prompt + request that WOULD be sent, then stop --
    if args.dry_run:
        system = build_system_prompt()
        user = build_user_message(args.product, resolved)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        (out_dir / "phase_a_system_prompt.txt").write_text(system)
        (out_dir / "phase_a_user_message.txt").write_text(user)
        (out_dir / "phase_a_request_body.json").write_text(json.dumps(body, indent=2, ensure_ascii=False))
        approx_tok = (len(system) + len(user)) // 4
        print(f"\n[dry-run] model            : {args.model_id}", file=sys.stderr)
        print(f"[dry-run] endpoint         : {{AWS_BEDROCK_ENDPOINT_URL}}/model/{args.model_id}/invoke", file=sys.stderr)
        print(f"[dry-run] system prompt    : {len(system):>7} chars", file=sys.stderr)
        print(f"[dry-run] user message     : {len(user):>7} chars  ({len(resolved)} docs)", file=sys.stderr)
        print(f"[dry-run] ~input tokens    : {approx_tok:>7} (rough, chars/4)", file=sys.stderr)
        print(f"[dry-run] wrote phase_a_system_prompt.txt / _user_message.txt / _request_body.json to {out_dir}", file=sys.stderr)
        print(f"[dry-run] (no Bedrock call made; set AWS_BEDROCK_* env vars and drop --dry-run to run for real)", file=sys.stderr)
        return

    # -- PHASE A: Claude infers tiers + assignments --
    print(f"[phase-a] asking {args.model_id} to infer tiers for {len(resolved)} docs",
          file=sys.stderr)
    config = assign_docs(args.product, resolved, args.model_id,
                         args.max_tokens, args.temperature)

    problems = validate_phase_a(config, resolved)
    (out_dir / "phase_a_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (out_dir / "tier_hierarchy.md").write_text(config.get("tier_hierarchy_md", ""))

    print(f"\n[phase-a] inferred {len(config.get('tiers', []))} tiers, "
          f"{len(config.get('subchapters', []))} sub-chapters, "
          f"{len(config.get('assignments', []))} assignments", file=sys.stderr)
    for t in config.get("tiers", []):
        n = sum(1 for a in config.get("assignments", []) if a["tier"] == t["id"])
        print(f"        T{t['id']:>2}  n={n:>4}  {t['title']}", file=sys.stderr)

    if problems:
        print("\n[phase-a] VALIDATION PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"        - {p}", file=sys.stderr)
        coverage_only = all("coverage gap" in p for p in problems)
        if not (coverage_only and args.allow_coverage_gaps):
            sys.exit("[fatal] Phase A validation failed. Inspect phase_a_config.json. "
                     "Re-run, or pass --allow-coverage-gaps to drop unassigned docs.")

    # -- write assignments.json (Phase B input) --
    title_by_url = {d["url"]: d["title"] for d in resolved}
    assignments = [
        {"url": a["url"], "subchapter": a["subchapter"], "tier": int(a["tier"]),
         "title": title_by_url.get(a["url"], "")}
        for a in config.get("assignments", [])
        if a["url"] in title_by_url  # drop hallucinated urls
    ]
    assignments_path = out_dir / "assignments.json"
    assignments_path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False))
    print(f"\n[phase-a] wrote {assignments_path} ({len(assignments)} docs)", file=sys.stderr)
    print(f"[phase-a] wrote {out_dir / 'tier_hierarchy.md'}", file=sys.stderr)

    if args.phase_a_only:
        print("[done] --phase-a-only: stopping before the JSONL build.", file=sys.stderr)
        return

    # -- PHASE B: drive the deterministic builder with these assignments --
    cmd = [
        sys.executable, str(_BUILDER_FROM_ASSIGNMENTS),
        "--corpus", args.corpus,
        "--out-dir", str(out_dir),
        "--assignments", str(assignments_path),
        *V6_BUILD_FLAGS,
    ]
    print(f"\n[phase-b] running: {' '.join(cmd)}\n", file=sys.stderr)
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        sys.exit(f"[fatal] build-classification-data failed (exit {result.returncode}).")

    # -- summary --
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"[done] {args.product} dataset written to {out_dir}", file=sys.stderr)
    for fn in ("directional_train.jsonl", "directional_val.jsonl",
               "directional_test.jsonl"):
        fp = out_dir / fn
        if fp.exists():
            n = sum(1 for _ in fp.open())
            print(f"        {fn:<28} {n:>8} rows", file=sys.stderr)
    print(f"        tier_hierarchy.md, phase_a_config.json, "
          f"directional_manifest.json also written.", file=sys.stderr)


if __name__ == "__main__":
    main()
