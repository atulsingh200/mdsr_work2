"""Comprehensive smoke test for every registered dataset.

For each dataset:
  1. Try to download (idempotent — no-op if already present).
  2. Iterate up to N pairs, report counts + length stats.
  3. Sample 2 triples to confirm negative sampling works end-to-end.

Manual stubs are skipped with a clear note. Use --include-stubs to attempt
them anyway (they will raise ManualDownloadRequired).

Usage:
    uv run python scripts/test_all_loaders.py
    uv run python scripts/test_all_loaders.py --limit 200 --skip multiwoz_v22 movielens_1m
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path

from followup_data import default_root, get, list_datasets
from followup_data.base import DatasetNotDownloaded
from followup_data.loaders._manual import ManualDownloadRequired
from followup_data.negatives import with_random_negatives


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def test_one(name: str, root: Path, limit: int) -> dict:
    cls = get(name)
    split = cls.splits[0]
    ds = cls(root=root, split=split)

    result = {
        "name": name,
        "split": split,
        "homepage": cls.homepage,
        "stub": False,
        "downloaded": False,
        "pairs": 0,
        "anchor_mean": 0.0,
        "positive_mean": 0.0,
        "triples": 0,
        "duration_s": 0.0,
        "error": None,
    }

    t0 = time.time()
    try:
        ds.download()
        result["downloaded"] = ds.is_downloaded()
    except ManualDownloadRequired as e:
        result["stub"] = True
        result["error"] = "manual download required"
        result["duration_s"] = time.time() - t0
        return result
    except Exception as e:
        result["error"] = f"download failed: {type(e).__name__}: {e}"
        result["duration_s"] = time.time() - t0
        return result

    pairs = []
    try:
        for ex in ds:
            pairs.append(ex)
            if len(pairs) >= limit:
                break
    except DatasetNotDownloaded as e:
        result["error"] = f"DatasetNotDownloaded: {e}"
        result["duration_s"] = time.time() - t0
        return result
    except Exception as e:
        result["error"] = f"iter failed: {type(e).__name__}: {e}"
        result["duration_s"] = time.time() - t0
        return result

    result["pairs"] = len(pairs)
    if pairs:
        result["anchor_mean"] = statistics.mean(len(p.anchor) for p in pairs)
        result["positive_mean"] = statistics.mean(len(p.positive) for p in pairs)

    if pairs:
        try:
            triples = list(with_random_negatives(pairs[: max(64, limit)], k=2, seed=0))
            result["triples"] = len(triples)
        except Exception as e:
            result["error"] = f"triples failed: {type(e).__name__}: {e}"

    result["duration_s"] = time.time() - t0
    return result


def fmt(r: dict) -> str:
    if r["stub"]:
        status = f"{YELLOW}stub{RESET}"
    elif r["error"]:
        status = f"{RED}FAIL{RESET}"
    else:
        status = f"{GREEN}ok  {RESET}"
    pairs = f"{r['pairs']:>5}" if r["pairs"] else f"{DIM}    -{RESET}"
    triples = f"{r['triples']:>4}" if r["triples"] else f"{DIM}   -{RESET}"
    a = f"{r['anchor_mean']:>4.0f}" if r["anchor_mean"] else "   -"
    p = f"{r['positive_mean']:>4.0f}" if r["positive_mean"] else "   -"
    t = f"{r['duration_s']:>5.1f}s"
    err = (r["error"] or "")[:60]
    return (
        f"  {status}  {r['name']:<18}  pairs={pairs}  triples={triples}  "
        f"anc={a}  pos={p}  {t}  {DIM}{err}{RESET}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(default_root()))
    p.add_argument("--limit", type=int, default=200, help="pairs to iterate per dataset")
    p.add_argument("--include-stubs", action="store_true")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=[])
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    names = list_datasets()
    if args.only:
        names = [n for n in names if n in args.only]
    if args.skip:
        names = [n for n in names if n not in args.skip]

    print(f"{BOLD}Testing {len(names)} dataset(s) with limit={args.limit} root={root}{RESET}")
    print()

    results = []
    for n in names:
        cls = get(n)
        # Heuristic: skip manual-stub classes unless requested.
        is_stub = "_ManualDataset" in [b.__name__ for b in cls.__mro__]
        if is_stub and not args.include_stubs:
            results.append({
                "name": n, "split": cls.splits[0], "homepage": cls.homepage,
                "stub": True, "downloaded": False, "pairs": 0,
                "anchor_mean": 0.0, "positive_mean": 0.0, "triples": 0,
                "duration_s": 0.0,
                "error": "stub (use --include-stubs to test)",
            })
            print(fmt(results[-1]))
            continue
        try:
            r = test_one(n, root, args.limit)
        except Exception as e:
            r = {
                "name": n, "split": cls.splits[0], "homepage": cls.homepage,
                "stub": False, "downloaded": False, "pairs": 0,
                "anchor_mean": 0.0, "positive_mean": 0.0, "triples": 0,
                "duration_s": 0.0,
                "error": f"unexpected: {type(e).__name__}: {e}",
            }
            traceback.print_exc(file=sys.stderr)
        results.append(r)
        print(fmt(r))

    print()
    ok = sum(1 for r in results if not r["stub"] and not r["error"])
    failed = sum(1 for r in results if not r["stub"] and r["error"])
    stubs = sum(1 for r in results if r["stub"])
    print(f"{BOLD}Summary:{RESET}  {GREEN}{ok} ok{RESET}  {RED}{failed} failed{RESET}  {YELLOW}{stubs} stubs{RESET}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
