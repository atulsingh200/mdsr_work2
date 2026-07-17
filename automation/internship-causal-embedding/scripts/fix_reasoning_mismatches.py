#!/usr/bin/env python3
"""Fix rows where Claude's VERDICT disagrees with the ground-truth label.

For each mismatched row, re-calls claude-sonnet-4-6 with the correct verdict
locked in, so the model writes reasoning that supports the right answer.

Overwrites each *_with_reasoning.jsonl in-place (atomic rename).

Usage:
    python3 scripts/fix_reasoning_mismatches.py \
        --data-dir data/new_aep_workflow_scrap/directional_procedural_with_reasoning \
        [--splits train val test] \
        [--max-workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
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

# ---------------------------------------------------------------------------
# Prompts — correct verdict is provided so reasoning must support it
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert in Adobe Experience Platform (AEP) and Adobe Journey Optimizer \
(AJO) product workflows.

You are given two steps from the same how-to procedure and the correct ordering \
verdict. Write one or two crisp sentences explaining WHY that verdict is correct, \
based only on what the two steps describe.

Respond in EXACTLY this format — reasoning first, then the verdict line. \
Nothing else before or after.

<one or two crisp sentences supporting the verdict>
VERDICT: BEFORE      (Step A must be completed before Step B)
VERDICT: NOT_BEFORE  (Step B comes first, or the two are independent)\
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

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def fix_reasoning(text_1: str, text_2: str, verdict: str,
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

def extract_verdict(reasoning: str) -> str | None:
    m = re.search(r'VERDICT:\s*(BEFORE|NOT_BEFORE)', reasoning or '')
    return m.group(1) if m else None

def label_to_verdict(label: int) -> str:
    return "BEFORE" if label == 1 else "NOT_BEFORE"

# ---------------------------------------------------------------------------
# Per-split fix
# ---------------------------------------------------------------------------
def fix_split(path: Path, max_workers: int) -> None:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # identify mismatches
    todo_indices: list[int] = []
    for i, r in enumerate(rows):
        verdict = extract_verdict(r.get('reasoning', ''))
        correct = label_to_verdict(r['label'])
        if verdict != correct:
            todo_indices.append(i)

    print(f"[{path.name}] {len(rows)} rows, {len(todo_indices)} mismatches to fix", flush=True)
    if not todo_indices:
        print(f"[{path.name}] nothing to do", flush=True)
        return

    completed = [0]
    errors    = [0]
    lock = threading.Lock()

    def _worker(i: int) -> tuple[int, str | None, str | None]:
        r = rows[i]
        correct_verdict = label_to_verdict(r['label'])
        try:
            new_reasoning = fix_reasoning(r['text_1'], r['text_2'], correct_verdict)
            return i, new_reasoning, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i): i for i in todo_indices}
        for fut in as_completed(futures):
            i, new_reasoning, err = fut.result()
            with lock:
                completed[0] += 1
                if err:
                    errors[0] += 1
                    print(f"  [{completed[0]}/{len(todo_indices)}] idx={i} ERROR: {err}", flush=True)
                else:
                    rows[i]['reasoning'] = new_reasoning
                    if completed[0] % 500 == 0 or completed[0] == len(todo_indices):
                        print(f"  [{completed[0]}/{len(todo_indices)}] fixed", flush=True)

    # verify fix quality
    still_wrong = sum(
        1 for i in todo_indices
        if extract_verdict(rows[i].get('reasoning', '')) != label_to_verdict(rows[i]['label'])
    )
    print(f"[{path.name}] after fix: {still_wrong} still mismatched (model overrode verdict)", flush=True)

    # atomic write
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)
    print(f"[{path.name}] written → {path}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data-dir',
                    default='data/new_aep_workflow_scrap/directional_procedural_with_reasoning')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                    choices=['train', 'val', 'test'])
    ap.add_argument('--max-workers', type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"data_dir : {data_dir}", flush=True)
    print(f"model    : {_MODEL}", flush=True)
    print(f"workers  : {args.max_workers}", flush=True)

    for split in args.splits:
        path = data_dir / f'directional_{split}_with_reasoning.jsonl'
        if not path.exists():
            print(f"[SKIP] {path} not found", flush=True)
            continue
        fix_split(path, args.max_workers)

    print("\n[all splits done]", flush=True)

if __name__ == '__main__':
    main()
