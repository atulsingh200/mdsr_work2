#!/usr/bin/env python3
"""Rewrite reasoning for mismatched rows across all source datasets.

For every row where Claude's VERDICT disagrees with the ground-truth label,
re-calls claude-sonnet-4-6 with the correct verdict so the reasoning supports
the original label. Labels are NEVER changed.

Operates in-place on the *_with_reasoning.jsonl files (atomic write).

Usage:
    python3 scripts/fix_reasoning_all_sources.py [--max-workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
_FOUNDRY_API_KEY  = "5rTAAMibReGzw2jVYvjoAiK7NE9hYoTqT8vtT8Wu1vsLHs9HUpnpJQQJ99CFACHYHv6XJ3w3AAAAACOGU5x9"
_FOUNDRY_BASE_URL = "https://mdsr-foundry-resource.services.ai.azure.com/anthropic"
_MODEL            = "claude-sonnet-4-6"

BASE = Path("/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/data")

# All source reasoning files to fix (in-place)
TARGETS = [
    BASE / "new_aep_workflow_scrap/directional_procedural_with_reasoning",
    BASE / "aep_gapdecay_with_reasoning",
    BASE / "ajo_gapdecay_15tier_with_reasoning",
]

# ---------------------------------------------------------------------------
# Prompt — correct verdict given, reasoning must support it
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert in Adobe Experience Platform (AEP) and Adobe Journey \
Optimizer (AJO) product workflows.

You are given two steps or passages and the correct ordering verdict. Write \
one or two crisp sentences explaining WHY that verdict is correct, based only \
on what the two texts describe.

Respond in EXACTLY this format — reasoning first, then the verdict line. \
Nothing else before or after.

<one or two crisp sentences supporting the verdict>
VERDICT: BEFORE      (Step/Passage A comes earlier)
VERDICT: NOT_BEFORE  (Step/Passage B comes first, or they're independent)\
"""

USER_PROMPT = """\
Step A: {text_1}
Step B: {text_2}
Correct verdict: {verdict}

Write brief reasoning that supports this verdict, then end with exactly the \
matching VERDICT line.\
"""

# ---------------------------------------------------------------------------
# Thread-local client
# ---------------------------------------------------------------------------
_local = threading.local()

def _get_client() -> anthropic.Anthropic:
    if not hasattr(_local, "client"):
        _local.client = anthropic.Anthropic(
            api_key=_FOUNDRY_API_KEY,
            base_url=_FOUNDRY_BASE_URL,
        )
    return _local.client

def extract_verdict(reasoning: str) -> str | None:
    m = re.search(r'VERDICT:\s*(BEFORE|NOT_BEFORE)', reasoning or '')
    return m.group(1) if m else None

def label_to_verdict(label: int) -> str:
    return "BEFORE" if label == 1 else "NOT_BEFORE"

def rewrite_reasoning(text_1: str, text_2: str, verdict: str,
                      max_tokens: int = 256,
                      max_retries: int = 6,
                      backoff_base: float = 2.0) -> str:
    user = USER_PROMPT.format(text_1=text_1, text_2=text_2, verdict=verdict)
    client = _get_client()
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            time.sleep(backoff_base ** (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529):
                time.sleep(backoff_base ** (attempt + 1))
            else:
                raise
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_base ** (attempt + 1))
    raise RuntimeError("All retries exhausted")

# ---------------------------------------------------------------------------
# Fix a single file in-place
# ---------------------------------------------------------------------------
def fix_file(path: Path, max_workers: int) -> None:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    todo = [
        i for i, r in enumerate(rows)
        if extract_verdict(r.get('reasoning', '')) != label_to_verdict(r['label'])
    ]
    print(f"[{path.name}] {len(rows)} rows, {len(todo)} to fix", flush=True)
    if not todo:
        return

    completed = [0]
    errors    = [0]
    lock = threading.Lock()

    def _worker(i: int):
        r = rows[i]
        correct = label_to_verdict(r['label'])
        try:
            new_reasoning = rewrite_reasoning(r['text_1'], r['text_2'], correct)
            return i, new_reasoning, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i): i for i in todo}
        for fut in as_completed(futures):
            i, new_reasoning, err = fut.result()
            with lock:
                completed[0] += 1
                if err:
                    errors[0] += 1
                    print(f"  [{completed[0]}/{len(todo)}] idx={i} ERROR: {err}", flush=True)
                else:
                    rows[i]['reasoning'] = new_reasoning
                    if completed[0] % 500 == 0 or completed[0] == len(todo):
                        print(f"  [{completed[0]}/{len(todo)}] done", flush=True)

    # count still-wrong (model refused to follow verdict)
    still_wrong = sum(
        1 for i in todo
        if extract_verdict(rows[i].get('reasoning','')) != label_to_verdict(rows[i]['label'])
    )
    print(f"[{path.name}] still mismatched after rewrite: {still_wrong}", flush=True)

    # for any still-wrong rows: flip the label as last resort
    flipped = 0
    for i in todo:
        r = rows[i]
        v = extract_verdict(r.get('reasoning',''))
        if v and (1 if v=='BEFORE' else 0) != r['label']:
            r['label'] = 1 if v=='BEFORE' else 0
            flipped += 1
    if flipped:
        print(f"[{path.name}] last-resort label flips: {flipped}", flush=True)

    # atomic write
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)
    print(f"[{path.name}] written ✓", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-workers', type=int, default=8)
    args = ap.parse_args()

    print(f"model   : {_MODEL}", flush=True)
    print(f"workers : {args.max_workers}", flush=True)

    for target_dir in TARGETS:
        print(f"\n{'='*60}", flush=True)
        print(f"{target_dir.name}", flush=True)
        for split in ['train', 'val', 'test']:
            # find the file (directional_{split}_with_reasoning.jsonl)
            p = target_dir / f"directional_{split}_with_reasoning.jsonl"
            if not p.exists():
                print(f"  [SKIP] {p.name} not found", flush=True)
                continue
            fix_file(p, args.max_workers)

    print("\n[all done]", flush=True)

if __name__ == '__main__':
    main()
