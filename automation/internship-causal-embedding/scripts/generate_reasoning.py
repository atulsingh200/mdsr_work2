#!/usr/bin/env python3
"""Generate LLM reasoning for directional procedural pairs.

Reads directional_{train,val,test}.jsonl, calls claude-sonnet-4-6 via Azure
Foundry with the canonical SYSTEM_PROMPT / USER_PROMPT, and writes
*_with_reasoning.jsonl files that add a `reasoning` field to each row.

Rows are written to disk immediately as they complete (not batched at the end),
so you can watch progress with `wc -l` or `tail -f` on the output file.

The output file is append-only while running; on --resume it skips any row
whose original index (_idx) is already present in the file.

Usage:
    python3 scripts/generate_reasoning.py \
        --data-dir data/new_aep_workflow_scrap/directional_procedural \
        --out-dir  data/new_aep_workflow_scrap/directional_procedural_with_reasoning \
        [--splits train val test] \
        [--max-workers 8] \
        [--max-rows N]   # smoke-test: only process first N rows per split
        [--resume]       # skip rows already written to the output file

Credentials are kept inside the script — not exported to the shell environment —
to avoid conflicts with other API keys in the environment.
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Azure Foundry credentials — NOT exported to the environment
# ---------------------------------------------------------------------------
_FOUNDRY_API_KEY  = "5rTAAMibReGzw2jVYvjoAiK7NE9hYoTqT8vtT8Wu1vsLHs9HUpnpJQQJ99CFACHYHv6XJ3w3AAAAACOGU5x9"
_FOUNDRY_BASE_URL = "https://mdsr-foundry-resource.services.ai.azure.com/anthropic"
_MODEL            = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert in Adobe Experience Platform (AEP) and Adobe Journey Optimizer \
(AJO) product workflows.

You are given two steps from the same how-to procedure. Decide which must come \
first based solely on what the steps describe.

Respond in EXACTLY this format — one or two crisp sentences of reasoning, \
then the verdict. Nothing else before or after.

<one or two sentences — direct and to the point>
VERDICT: BEFORE      (Step A must be completed before Step B)
VERDICT: NOT_BEFORE  (Step B comes first, or the two are independent)\
"""

USER_PROMPT = """\
Step A: {text_1}
Step B: {text_2}

Which must come first and why? Be brief, then end with exactly one VERDICT line.\
"""

# ---------------------------------------------------------------------------
# Thread-local Anthropic client
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
# Single-row inference with retry
# ---------------------------------------------------------------------------
def generate_reasoning(text_1: str, text_2: str,
                        max_tokens: int = 512,
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
    raise RuntimeError(f"All {max_retries} retries exhausted")


# ---------------------------------------------------------------------------
# Per-split processing  — writes rows to disk immediately as they finish
# ---------------------------------------------------------------------------
def process_split(
    in_path: Path,
    out_path: Path,
    max_workers: int,
    max_rows: int | None,
    resume: bool,
    max_tokens: int,
) -> None:
    # -- load source rows --
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

    # -- resume: collect indices already written --
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
        print(f"[{in_path.name}] resume: {len(done_indices)} already done, "
              f"{total - len(done_indices)} remaining", flush=True)

    todo = [(i, rows[i]) for i in range(total) if i not in done_indices]
    if not todo:
        print(f"[{in_path.name}] nothing to do", flush=True)
        return

    # -- open output file for appending --
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("a", buffering=1)   # line-buffered → flush each write
    write_lock = threading.Lock()

    completed = [0]
    errors    = [0]

    def _worker(i: int, row: dict):
        try:
            reasoning = generate_reasoning(row["text_1"], row["text_2"],
                                           max_tokens=max_tokens)
            out_row = {**row, "reasoning": reasoning, "_idx": i}
            err = None
        except Exception as e:
            out_row = {**row, "reasoning": None, "_idx": i}
            err = str(e)
        return i, out_row, err

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i, row): i for i, row in todo}
        for fut in as_completed(futures):
            i, out_row, err = fut.result()
            with write_lock:
                # strip _idx before writing to keep output clean
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
        description="Generate claude-sonnet-4-6 reasoning for directional pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir",
                    default="data/new_aep_workflow_scrap/directional_procedural")
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to <data-dir>_with_reasoning")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-rows",    type=int, default=None)
    ap.add_argument("--resume",      action="store_true",
                    help="Skip rows already written to the output file")
    ap.add_argument("--max-tokens",  type=int, default=512)
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
