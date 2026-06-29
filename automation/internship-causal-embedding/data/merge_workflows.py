#!/usr/bin/env python3
"""
Merge two directional workflow datasets into one combined dataset:

  new_ajo_workflows      (sentence-level, fields: tier_*, sub_*, url_1, url_2)
  new_aep_workflow_scrap (phase-level,    fields: step_*, url, title)

Output: merged_workflows/directional_{train,val,test}.jsonl

Both inputs are already split per-document (no intra-source leakage). We merge
split-by-split, normalise to one union schema (nothing dropped), tag each row
with `source` for provenance, then shuffle. Cross-source URL overlap across the
train/test boundary is checked and reported.
"""
import json
import random
import sys
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent
SOURCES = {
    "new_ajo_workflows": DATA / "new_ajo_workflows",
    "new_aep_workflow_scrap": DATA / "new_aep_workflow_scrap",
}
OUT = DATA / "merged_workflows"
SPLITS = ("train", "val", "test")

# union schema: every row carries the same keys (null where a source lacks it)
UNION_KEYS = ["text_1", "text_2", "label", "source",
              "url_1", "url_2", "tier_1", "tier_2", "sub_1", "sub_2",
              "step_1", "step_2", "title"]


def normalise(rec: dict, source: str) -> dict:
    out = {k: None for k in UNION_KEYS}
    out["text_1"] = rec["text_1"]
    out["text_2"] = rec["text_2"]
    out["label"] = rec["label"]
    out["source"] = source
    # urls: scrap has single `url`; ajo has url_1/url_2
    if "url" in rec and rec.get("url") is not None:
        out["url_1"] = out["url_2"] = rec["url"]
    out["url_1"] = rec.get("url_1", out["url_1"])
    out["url_2"] = rec.get("url_2", out["url_2"])
    for k in ("tier_1", "tier_2", "sub_1", "sub_2", "step_1", "step_2", "title"):
        if k in rec:
            out[k] = rec[k]
    return out


def main():
    rng = random.Random(42)
    OUT.mkdir(exist_ok=True)

    skipped = Counter()
    by_split = {}
    for sp in SPLITS:
        rows = []
        for source, d in SOURCES.items():
            path = d / f"directional_{sp}.jsonl"
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped[f"{source}/{sp}"] += 1   # pre-existing corrupt line
                    continue
                rows.append(normalise(rec, source))
        by_split[sp] = rows

    # Global dedup of identical (text_1, text_2, label). The AJO source has the
    # same pair across its own splits; iterating train first resolves every
    # collision in favour of train, so no eval pair is ever also in train.
    seen = set()
    deduped = Counter()
    for sp in SPLITS:
        kept = []
        for r in by_split[sp]:
            key = (r["text_1"], r["text_2"], r["label"])
            if key in seen:
                deduped[sp] += 1
                continue
            seen.add(key)
            kept.append(r)
        by_split[sp] = kept

    grand_total = 0
    for sp in SPLITS:
        rows = by_split[sp]
        rng.shuffle(rows)
        with open(OUT / f"directional_{sp}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_source = Counter(r["source"] for r in rows)
        labels = Counter(r["label"] for r in rows)
        grand_total += len(rows)
        print(f"[{sp}] total={len(rows)} per_source={dict(per_source)} "
              f"labels={dict(labels)} deduped={deduped[sp]}", file=sys.stderr)

    # ---- cross-source URL leakage check across train/test & train/val ----
    def urls(sp):
        s = set()
        for r in (json.loads(l) for l in open(OUT / f"directional_{sp}.jsonl")):
            for k in ("url_1", "url_2"):
                if r.get(k):
                    s.add(r[k])
        return s
    tr, va, te = urls("train"), urls("val"), urls("test")
    print(f"\n[leakage] URL overlap train/test={len(tr & te)} "
          f"train/val={len(tr & va)} val/test={len(va & te)}", file=sys.stderr)
    if skipped:
        print(f"[skipped] unparseable source lines: {dict(skipped)}", file=sys.stderr)
    print(f"[done] grand total pairs: {grand_total} -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
