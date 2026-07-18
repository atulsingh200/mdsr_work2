#!/usr/bin/env python3
"""Fix mismatched reasoning in gapdecay datasets.

Makes a COPY of each *_with_reasoning.jsonl into a *_with_reasoning_fixed.jsonl,
then re-generates reasoning only for rows where the VERDICT disagrees with the
ground-truth label — telling Claude the correct verdict so reasoning aligns.

Original files are NEVER modified.

Usage:
    python3 scripts/fix_reasoning_mismatches_gapdecay.py \
        --data-dirs \
            data/ajo_gapdecay_15tier_with_reasoning \
            data/aep_gapdecay_notier11_with_reasoning \
        [--splits train val test] \
        [--max-workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Credentials — NOT exported to the environment
# ---------------------------------------------------------------------------
_FOUNDRY_API_KEY  = "5rTAAMibReGzw2jVYvjoAiK7NE9hYoTqT8vtT8Wu1vsLHs9HUpnpJQQJ99CFACHYHv6XJ3w3AAAAACOGU5x9"
_FOUNDRY_BASE_URL = "https://mdsr-foundry-resource.services.ai.azure.com/anthropic"
_MODEL            = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Prompts — correct verdict is provided, reasoning must support it
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert in both Adobe Experience Platform (AEP) and Adobe Journey \
Optimizer (AJO) who knows the order in which these products' capabilities are \
learned and built on each other.

You are given two short passages and the correct ordering verdict. Write one or \
two crisp sentences explaining WHY that verdict is correct, based only on what \
the two passages describe and where each topic sits in the product's learning \
or implementation sequence.

Respond in EXACTLY this format — reasoning first, then the verdict line. \
Nothing else before or after.

<one or two crisp sentences supporting the verdict>
VERDICT: BEFORE      (Passage A's topic comes earlier)
VERDICT: NOT_BEFORE  (Passage B's topic comes earlier, or they're independent)\
"""

USER_PROMPT = """\
Passage A: {text_1}
Passage B: {text_2}
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
# Helpers
# ---------------------------------------------------------------------------
def extract_verdict(reasoning: str) -> str | None:
    m = re.search(r'VERDICT:\s*(BEFORE|NOT_BEFORE)', reasoning or '')
    return m.group(1) if m else None

def label_to_verdict(label: int) -> str:
    return "BEFORE" if label == 1 else "NOT_BEFORE"

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

# ---------------------------------------------------------------------------
# Per-file fix — operates on the *_fixed.jsonl copy
# ---------------------------------------------------------------------------
def fix_split(src_path: Path, dst_path: Path, max_workers: int) -> None:
    # copy original -> fixed if not already there
    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"[{src_path.name}] copied -> {dst_path.name}", flush=True)

    rows: list[dict] = []
    with dst_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    todo_indices = [
        i for i, r in enumerate(rows)
        if extract_verdict(r.get('reasoning', '')) != label_to_verdict(r['label'])
    ]
    print(f"[{dst_path.name}] {len(rows)} rows, {len(todo_indices)} mismatches to fix", flush=True)
    if not todo_indices:
        print(f"[{dst_path.name}] nothing to do", flush=True)
        return

    completed = [0]
    errors    = [0]
    lock = threading.Lock()

    def _worker(i: int):
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

    still_wrong = sum(
        1 for i in todo_indices
        if extract_verdict(rows[i].get('reasoning', '')) != label_to_verdict(rows[i]['label'])
    )
    print(f"[{dst_path.name}] after fix: {still_wrong} still mismatched", flush=True)

    # atomic write
    tmp = dst_path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(dst_path)
    print(f"[{dst_path.name}] written -> {dst_path}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data-dirs', nargs='+', required=True,
                    help='One or more *_with_reasoning directories to fix')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                    choices=['train', 'val', 'test'])
    ap.add_argument('--max-workers', type=int, default=8)
    args = ap.parse_args()

    print(f"model   : {_MODEL}", flush=True)
    print(f"workers : {args.max_workers}", flush=True)

    for data_dir in args.data_dirs:
        data_dir = Path(data_dir)
        print(f"\n{'='*60}", flush=True)
        print(f"data_dir: {data_dir}", flush=True)
        for split in args.splits:
            src = data_dir / f'directional_{split}_with_reasoning.jsonl'
            dst = data_dir / f'directional_{split}_with_reasoning_fixed.jsonl'
            if not src.exists():
                print(f"[SKIP] {src} not found", flush=True)
                continue
            fix_split(src, dst, args.max_workers)

    print("\n[all done]", flush=True)

if __name__ == '__main__':
    main()
