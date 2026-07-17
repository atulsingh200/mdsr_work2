#!/usr/bin/env python3
"""Emit the 15-tier AJO resolved-assignments file.

The canonical AJO ordering is the 15-tier SUBCHAPTER_TO_TIER table in
build_classification_data.py, derived from the AJO TOC (TOC_URL on GitHub) via
assign_subchapter. This script reruns that resolution against our corpus and
dumps {assignments: [{url, subchapter, tier}]}, so the downstream gap-decay
builder can consume it exactly like ajo_doc_resolved_assignments.json (which
used the WRONG 10-tier scheme).

Mirrors build_classification_data.main() lines ~794-829 (AJO TOC path).

Usage (from this dir):
    python3 make_ajo_15tier_assignments.py \
        --corpus aep_docs_collection_v_24-03-2026.json \
        --out ajo_doc_resolved_assignments_15tier.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bcd", _HERE / "build_classification_data.py")
_bcd = importlib.util.module_from_spec(_spec)
sys.modules["_bcd"] = _bcd
_spec.loader.exec_module(_bcd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="aep_docs_collection_v_24-03-2026.json")
    ap.add_argument("--out", default="ajo_doc_resolved_assignments_15tier.json")
    args = ap.parse_args()

    corpus = _bcd.load_corpus(Path(args.corpus))
    print(f"[corpus] {len(corpus)} URLs", file=sys.stderr)

    print(f"[toc] fetching {_bcd.TOC_URL}", file=sys.stderr)
    toc = _bcd.fetch_toc()
    leaves = _bcd.parse_toc(toc)
    print(f"[toc] parsed {len(leaves)} leaves", file=sys.stderr)

    S2T = _bcd.SUBCHAPTER_TO_TIER
    resolved: list[dict] = []
    url_miss = sub_miss = 0
    for leaf in leaves:
        sub = _bcd.assign_subchapter(leaf)
        url_found = None
        for u in _bcd.url_candidates(leaf):
            if u in corpus:
                url_found = u
                break
        if not url_found:
            url_miss += 1
            continue
        if not sub or sub not in S2T:
            sub_miss += 1
            continue
        resolved.append({
            "url": url_found,
            "subchapter": sub,
            "tier": int(S2T[sub]),
        })

    # dedupe by URL (a URL may appear under two TOC locations)
    by_url: dict[str, dict] = {}
    for r in resolved:
        by_url.setdefault(r["url"], r)
    resolved = list(by_url.values())

    tier_docs: dict[int, int] = defaultdict(int)
    for r in resolved:
        tier_docs[r["tier"]] += 1
    print(f"[resolve] {len(resolved)} docs  (url_miss={url_miss}, sub_miss={sub_miss})",
          file=sys.stderr)
    for t in sorted(tier_docs):
        print(f"           T{t:>2}: {tier_docs[t]}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"assignments": resolved}, indent=2, ensure_ascii=False))
    print(f"[done] wrote {len(resolved)} assignments -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
