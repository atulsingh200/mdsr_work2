#!/usr/bin/env python3
"""Generate LLM reasoning for ajo_gapdecay_15tier directional pairs.

Uses the AJO syllabus-ordering prompt (tier-gap / gap-decay dataset).
Rows are written to disk immediately as they complete.
Supports --resume to skip already-done rows.

Usage:
    python3 scripts/generate_reasoning_gapdecay.py \
        --data-dir data/ajo_gapdecay_15tier \
        --out-dir  data/ajo_gapdecay_15tier_with_reasoning \
        [--splits train val test] \
        [--max-workers 8] \
        [--max-rows N] \
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import threading
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
# Prompts (AJO syllabus ordering)
# ---------------------------------------------------------------------------
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

Respond in EXACTLY this format — one or two crisp sentences, then the verdict. \
Nothing else before or after.

<one or two crisp sentences: what each passage is about and which comes earlier>
VERDICT: BEFORE      (Passage A's topic comes earlier)
VERDICT: NOT_BEFORE  (Passage B's topic comes earlier, or they're independent)\
"""

USER_PROMPT = """\
Passage A: {text_1}
Passage B: {text_2}

Which passage's topic would a learner reach earlier in Adobe Journey Optimizer, \
and why? Be brief, then end with exactly one VERDICT line.\
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
# Single-row inference
# ---------------------------------------------------------------------------
def generate_reasoning(text_1: str, text_2: str,
                        max_tokens: int = 256,
                        max_retries: int = 6,
                        backoff_base: float = 2.0) -> str:
    user = USER_PROMPT.format(text_1=text_1, text_2=text_2)
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
# Per-split processing — writes rows immediately as they complete
# ---------------------------------------------------------------------------
def process_split(
    in_path: Path,
    out_path: Path,
    max_workers: int,
    max_rows: int | None,
    resume: bool,
    max_tokens: int,
) -> None:
    rows: list[dict] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_rows is not None:
        rows = rows[:max_rows]
    total = len(rows)
    print(f"\n[{in_path.name}] {total} rows", flush=True)

    # resume: collect already-done indices
    done_indices: set[int] = set()
    if resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if "_idx" in r:
                            done_indices.add(r["_idx"])
                    except json.JSONDecodeError:
                        pass
        print(f"[{in_path.name}] resume: {len(done_indices)} done, "
              f"{total - len(done_indices)} remaining", flush=True)

    todo = [(i, rows[i]) for i in range(total) if i not in done_indices]
    if not todo:
        print(f"[{in_path.name}] nothing to do", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("a", buffering=1)
    write_lock = threading.Lock()
    completed = [0]
    errors    = [0]

    def _worker(i: int, row: dict):
        try:
            reasoning = generate_reasoning(row["text_1"], row["text_2"],
                                           max_tokens=max_tokens)
            out_row = {**row, "reasoning": reasoning, "_idx": i}
            return i, out_row, None
        except Exception as e:
            out_row = {**row, "reasoning": None, "_idx": i}
            return i, out_row, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i, row): i for i, row in todo}
        for fut in as_completed(futures):
            i, out_row, err = fut.result()
            with write_lock:
                clean = {k: v for k, v in out_row.items() if k != "_idx"}
                out_file.write(json.dumps(clean, ensure_ascii=False) + "\n")
                completed[0] += 1
                if err:
                    errors[0] += 1
                    print(f"  [{completed[0]}/{len(todo)}] idx={i} ERROR: {err}", flush=True)
                elif completed[0] % 500 == 0 or completed[0] == len(todo):
                    print(f"  [{completed[0]}/{len(todo)}] written so far", flush=True)

    out_file.close()
    print(f"[{in_path.name}] done — {completed[0] - errors[0]} ok, "
          f"{errors[0]} failed → {out_path}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate claude-sonnet-4-6 reasoning for ajo_gapdecay pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir",    default="data/ajo_gapdecay_15tier")
    ap.add_argument("--out-dir",     default=None,
                    help="Defaults to <data-dir>_with_reasoning")
    ap.add_argument("--splits",      nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-rows",    type=int, default=None)
    ap.add_argument("--resume",      action="store_true")
    ap.add_argument("--max-tokens",  type=int, default=256)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) if args.out_dir \
               else Path(str(data_dir) + "_with_reasoning")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"data_dir : {data_dir}", flush=True)
    print(f"out_dir  : {out_dir}", flush=True)
    print(f"model    : {_MODEL}", flush=True)
    print(f"workers  : {args.max_workers}", flush=True)

    for split in args.splits:
        in_path  = data_dir / f"directional_{split}.jsonl"
        out_path = out_dir  / f"directional_{split}_with_reasoning.jsonl"
        if not in_path.exists():
            print(f"[SKIP] {in_path} not found", flush=True)
            continue
        process_split(
            in_path=in_path,
            out_path=out_path,
            max_workers=args.max_workers,
            max_rows=args.max_rows,
            resume=args.resume,
            max_tokens=args.max_tokens,
        )

    print("\n[all splits done]", flush=True)

if __name__ == "__main__":
    main()
