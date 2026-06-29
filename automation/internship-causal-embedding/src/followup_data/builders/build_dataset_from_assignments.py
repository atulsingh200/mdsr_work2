#!/usr/bin/env python3
"""Build a tier-based directional classification dataset from PRE-RESOLVED assignments.

Product-agnostic counterpart to `build_classification_data.py`. That script is
AJO-specific: it fetches the AJO TOC and uses hand-written tier/sub-chapter tables.
This script takes those decisions as *input* — a JSON list of
`{url, subchapter, tier}` produced by `scripts/generate_dataset.py` (Claude does
the reasoning) — and runs the identical deterministic mechanics: reconstruct text,
segment, split, emit directional pairs, write JSONL + manifest.

The pure segmentation / chunking helpers are reused verbatim from
`build_classification_data.py` (imported by file path so the heavy followup_data
package __init__ is not triggered). Nothing in that file is modified.

Defaults reproduce the production ("v6") configuration: sentence segments (2-5),
all tier-pairs, single-direction, chunk-level split pooled per tier, capped at
1500 samples per tier-pair on train.

Invoke:
  python3 build_dataset_from_assignments.py \
      --corpus corpus.json --assignments assignments.json --out-dir out/

# ── AJO-doc dataset (ajo_doc_dataset) ─────────────────────────────────────────
# Reproducible command used to generate the AJO documentation tier dataset.
# Source : ajo-doc-tier-hierarchy.json  (10 tiers, 613 resolved docs)
# Corpus : aep_docs_collection_v_24-03-2026.json
# Pre-processing: ambiguous-slug (123) and missing (89) entries were skipped;
#   exact corpus URLs written to ajo_doc_resolved_assignments.json via:
#
#   python3 -c "
#   import json
#   with open('aep_docs_collection_v_24-03-2026.json') as f: data = json.load(f)
#   corpus_urls = set()
#   for group in data:
#       if group:
#           url = group[0]['metadata'].get('sourceUrl')
#           if url: corpus_urls.add(url)
#   slug_to_corpus = {}
#   for u in corpus_urls:
#       if '/docs/journey-optimizer/' in u and 'journey-optimizer-' not in u:
#           slug = u.rstrip('/').split('/')[-1]
#           slug_to_corpus.setdefault(slug, []).append(u)
#   with open('ajo-doc-tier-hierarchy.json') as f: hier = json.load(f)
#   resolved = []
#   for a in hier['assignments']:
#       url = a['url']; slug = url.rstrip('/').split('/')[-1]
#       if url in corpus_urls: corpus_url = url
#       elif slug in slug_to_corpus and len(slug_to_corpus[slug]) == 1: corpus_url = slug_to_corpus[slug][0]
#       else: continue
#       resolved.append({'url': corpus_url, 'subchapter': a['subchapter'], 'tier': a['tier']})
#   import json; open('ajo_doc_resolved_assignments.json','w').write(json.dumps({'assignments': resolved}, indent=2))
#   "
#
# Dataset build command (run from this directory):
#
# Step 1 — build train (capped) + raw val/test:
#   python3 build_dataset_from_assignments.py \
#       --corpus      ../../data/comparison_test/aep_docs_collection_v_24-03-2026.json \
#       --assignments ajo_doc_resolved_assignments.json \
#       --out-dir     ../../data/ajo_doc_dataset \
#       --unit chunk --window 256 --stride 128 \
#       --pair-scope adjacent \
#       --both-directions \
#       --max-chunk-pairs-per-tier-pair 11000 \
#       --cap-splits train \
#       --seed 42 --val-frac 0.10 --test-frac 0.10
#
# Step 2 — cap val/test to 1400 samples per tier-pair (80/10/10 by sample count):
#   python3 -c "
#   import json, random
#   from collections import defaultdict
#   from pathlib import Path
#   OUT = Path('../../data/ajo_doc_dataset')
#   CAP_PER_TIER_PAIR = 1400
#   SEED = 42
#   for split in ('val', 'test'):
#       rows = [json.loads(l) for l in open(OUT / f'directional_{split}.jsonl')]
#       by_pair = defaultdict(list)
#       for r in rows:
#           key = (r['tier_1'], r['tier_2']) if r['label'] == 1 else (r['tier_2'], r['tier_1'])
#           by_pair[key].append(r)
#       capped = []
#       rng = random.Random(SEED)
#       for key, lst in by_pair.items():
#           if len(lst) > CAP_PER_TIER_PAIR:
#               rng.shuffle(lst); lst = lst[:CAP_PER_TIER_PAIR]
#           capped.extend(lst)
#       rng.shuffle(capped)
#       open(OUT / f'directional_{split}.jsonl', 'w').writelines(json.dumps(r)+'\n' for r in capped)
#       print(f'{split}: {len(rows)} -> {len(capped)}')
#   "
#
# Result: train=99000, val=12256, test=12256  (≈ 80/10/10 by sample count)
#         val ∩ test = 0 text leaks; val/test capped deterministically (seed=42)
# ─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# --- reuse the original builder's pure helpers without importing the package ---
_HERE = Path(__file__).resolve().parent
_ORIG = _HERE / "build_classification_data.py"
_spec = importlib.util.spec_from_file_location("_orig_builder", _ORIG)
_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig)

load_corpus = _orig.load_corpus
split_sentences = _orig.split_sentences
split_paragraphs = _orig.split_paragraphs
chunk_text = _orig.chunk_text


def load_assignments(path: Path, corpus: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Turn the assignments file into the `resolved` list the pipeline expects."""
    with path.open() as f:
        data = json.load(f)
    # Support both a flat list and the wrapped {"assignments": [...]} format
    items = data["assignments"] if isinstance(data, dict) and "assignments" in data else data
    resolved: list[dict] = []
    misses: list[dict] = []
    for it in items:
        url = it["url"]
        if url not in corpus:
            misses.append({"title": it.get("title", ""), "url": url, "reason": "not in corpus"})
            continue
        sub, tier = it.get("subchapter"), it.get("tier")
        if not sub or tier is None:
            misses.append({"title": it.get("title", ""), "url": url, "reason": "missing subchapter/tier"})
            continue
        resolved.append({
            "url": url,
            "title": corpus[url]["title"] or it.get("title", ""),
            "chapter": it.get("chapter", sub),
            "subchapter": sub,
            "tier": int(tier),
        })
    return resolved, misses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--unit", choices=["sentence", "paragraph", "chunk"], default="sentence")
    ap.add_argument("--min-sentences", type=int, default=2)
    ap.add_argument("--max-sentences", type=int, default=5)
    ap.add_argument("--min-paragraph-tokens", type=int, default=20)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--tokenizer", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--pair-scope", choices=["adjacent", "all"], default="all")
    ap.add_argument("--single-direction", action="store_true", default=True)
    ap.add_argument("--both-directions", dest="single_direction", action="store_false")
    ap.add_argument("--max-chunk-pairs-per-tier-pair", type=int, default=1500)
    ap.add_argument("--cap-splits", choices=["train", "all"], default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer only needed for paragraph/chunk units (sentence is regex-based)
    tokenizer = None
    if args.unit in ("chunk", "paragraph"):
        from transformers import AutoTokenizer
        print(f"[tokenizer] loading {args.tokenizer}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"[corpus] loading {args.corpus}", file=sys.stderr)
    corpus = load_corpus(Path(args.corpus))
    print(f"[corpus] {len(corpus)} unique URLs", file=sys.stderr)

    resolved, url_misses = load_assignments(Path(args.assignments), corpus)
    print(f"[assignments] {len(resolved)} docs resolved, {len(url_misses)} misses",
          file=sys.stderr)
    if not resolved:
        sys.exit("[fatal] no assignments resolved against the corpus.")

    subchapter_to_tier = {r["subchapter"]: r["tier"] for r in resolved}
    tier_docs: dict[int, list[dict]] = defaultdict(list)
    for r in resolved:
        tier_docs[r["tier"]].append(r)
    print("[resolve] docs per tier:", file=sys.stderr)
    for t in sorted(tier_docs):
        print(f"           T{t:>2}  n={len(tier_docs[t]):>4}", file=sys.stderr)

    # -- segment --
    doc_chunks: dict[str, list[str]] = {}
    for r in resolved:
        body = corpus[r["url"]]["text"]
        if args.unit == "sentence":
            doc_chunks[r["url"]] = split_sentences(body, args.min_sentences, args.max_sentences)
        elif args.unit == "paragraph":
            doc_chunks[r["url"]] = split_paragraphs(body, tokenizer, args.min_paragraph_tokens)
        else:
            doc_chunks[r["url"]] = chunk_text(body, tokenizer, args.window, args.stride)
    total = sum(len(c) for c in doc_chunks.values())
    print(f"[segment] unit={args.unit}: {total} segments across {len(doc_chunks)} docs",
          file=sys.stderr)

    # -- chunk-level split pooled per tier (the v6 default) --
    chunk_entries_by_tier: dict[int, list[dict]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)
    for t in sorted(tier_docs):
        entries: list[dict] = []
        for r in tier_docs[t]:
            for i, ct in enumerate(doc_chunks[r["url"]]):
                entries.append({"text": ct, "url": r["url"], "chunk_idx": i,
                                "subchapter": r["subchapter"]})
        rng = random.Random(args.seed + hash(f"T{t}") % 10000)
        rng.shuffle(entries)
        n = len(entries)
        n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * args.val_frac)) if n - n_test >= 2 else 0
        for i, e in enumerate(entries):
            e["split"] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            split_counts[e["split"]] += 1
        chunk_entries_by_tier[t] = entries
    print(f"[split] chunk_tier — segments per split: {dict(split_counts)}", file=sys.stderr)

    # -- tier-pair iteration --
    tiers_sorted = sorted(tier_docs)
    if args.pair_scope == "adjacent":
        tier_pairs = [(tiers_sorted[i], tiers_sorted[i + 1]) for i in range(len(tiers_sorted) - 1)]
    else:
        tier_pairs = [(tiers_sorted[i], tiers_sorted[j])
                      for i in range(len(tiers_sorted)) for j in range(i + 1, len(tiers_sorted))]

    chunks_by_ts: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for t, entries in chunk_entries_by_tier.items():
        for e in entries:
            chunks_by_ts[(t, e["split"])].append(e)

    pairs_by_key: dict[tuple[int, int, str], list[tuple[dict, dict]]] = defaultdict(list)
    for t_low, t_hi in tier_pairs:
        for sp in ("train", "val", "test"):
            lows = chunks_by_ts.get((t_low, sp), [])
            highs = chunks_by_ts.get((t_hi, sp), [])
            for c_low in lows:
                for c_hi in highs:
                    pairs_by_key[(t_low, t_hi, sp)].append((c_low, c_hi))

    samples_per_tuple = 1 if args.single_direction else 2
    pre_cap: dict[str, int] = {}
    post_cap: dict[str, int] = {}
    for (tl, th, sp), lst in pairs_by_key.items():
        pre_cap[f"T{tl}-T{th}/{sp}"] = len(lst) * samples_per_tuple

    cap = args.max_chunk_pairs_per_tier_pair
    cap_splits = {"train"} if args.cap_splits == "train" else {"train", "val", "test"}
    if cap is not None:
        target = cap if args.single_direction else cap // 2
        for key, lst in list(pairs_by_key.items()):
            if key[2] in cap_splits and len(lst) > target:
                lrng = random.Random((args.cap_seed, *key).__hash__())
                shuf = list(lst)
                lrng.shuffle(shuf)
                pairs_by_key[key] = shuf[:target]
    for (tl, th, sp), lst in pairs_by_key.items():
        post_cap[f"T{tl}-T{th}/{sp}"] = len(lst) * samples_per_tuple

    # -- emit --
    writers = {sp: (out_dir / f"directional_{sp}.jsonl").open("w") for sp in ("train", "val", "test")}
    counts: dict[str, int] = defaultdict(int)
    dir_rng = random.Random(args.seed + 7)

    def emit(sp, low, hi, tl, th):
        if args.single_direction:
            if dir_rng.random() < 0.5:
                row = {"text_1": low["text"], "text_2": hi["text"], "label": 1,
                       "tier_1": tl, "tier_2": th, "sub_1": low["subchapter"], "sub_2": hi["subchapter"],
                       "url_1": low["url"], "url_2": hi["url"]}
            else:
                row = {"text_1": hi["text"], "text_2": low["text"], "label": 0,
                       "tier_1": th, "tier_2": tl, "sub_1": hi["subchapter"], "sub_2": low["subchapter"],
                       "url_1": hi["url"], "url_2": low["url"]}
            writers[sp].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[sp] += 1
        else:
            writers[sp].write(json.dumps({
                "text_1": low["text"], "text_2": hi["text"], "label": 1,
                "tier_1": tl, "tier_2": th, "sub_1": low["subchapter"], "sub_2": hi["subchapter"],
                "url_1": low["url"], "url_2": hi["url"]}, ensure_ascii=False) + "\n")
            writers[sp].write(json.dumps({
                "text_1": hi["text"], "text_2": low["text"], "label": 0,
                "tier_1": th, "tier_2": tl, "sub_1": hi["subchapter"], "sub_2": low["subchapter"],
                "url_1": hi["url"], "url_2": low["url"]}, ensure_ascii=False) + "\n")
            counts[sp] += 2

    try:
        for (t_low, t_hi, sp), lst in pairs_by_key.items():
            for c_low, c_hi in lst:
                emit(sp, c_low, c_hi, t_low, t_hi)
    finally:
        for w in writers.values():
            w.close()

    manifest = {
        "mode": "assignments",
        "unit": args.unit,
        "min_sentences": args.min_sentences if args.unit == "sentence" else None,
        "max_sentences": args.max_sentences if args.unit == "sentence" else None,
        "window": args.window if args.unit == "chunk" else None,
        "stride": args.stride if args.unit == "chunk" else None,
        "pair_scope": args.pair_scope,
        "single_direction": args.single_direction,
        "split_level": "chunk_tier",
        "max_chunk_pairs_per_tier_pair": args.max_chunk_pairs_per_tier_pair,
        "cap_splits": args.cap_splits,
        "seed": args.seed,
        "cap_seed": args.cap_seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "subchapter_to_tier": subchapter_to_tier,
        "n_docs_resolved": len(resolved),
        "docs_per_tier": {t: len(tier_docs[t]) for t in tiers_sorted},
        "segment_split_counts": dict(split_counts),
        "sample_counts": dict(counts),
        "pre_cap_counts": pre_cap,
        "post_cap_counts": post_cap,
        "url_misses": url_misses,
    }
    (out_dir / "directional_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[done] wrote samples: {dict(counts)}", file=sys.stderr)
    print(f"[done] manifest: {out_dir / 'directional_manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
