#!/usr/bin/env python3
"""Build a tier-directional classification dataset with GAP-DECAY downsampling.

Same mechanics as ``build_dataset_from_assignments.py`` (resolve docs -> segment
-> pool per tier -> doc/chunk split -> pair across tiers), but the per-tier-pair
sample cap DECAYS with the tier gap instead of being uniform:

    cap(g) = max(far_gap_floor, round(base_cap * (1 - gap_decay * (g - 1))))

where ``g`` is the RANK gap between the two tiers (adjacent tiers in sorted
order -> g=1 -> maximum samples; the cap shrinks linearly as the gap grows).
This is the "maximum adjacent pairs, fewer as the gap increases" strategy, using
the exact linear-decay formula from ``build_classification_data_workflow.py``.

The gap-decay cap is applied to the TRAIN split only. ``--target-train`` auto-
tunes ``base_cap`` (binary search) so the emitted train row count lands on a
target (e.g. 6300 for AJO, 13000 for AEP). VAL/TEST use a uniform per-tier-pair
cap (``--valtest-cap``).

Hermetic: reuses only the dependency-free helpers from build_classification_data
(``load_corpus``, ``load_assignments``, ``split_sentences``, ``split_docs``).
The ``chunk`` unit uses a self-contained WORD-window chunker (no tokenizer /
no network), so ``--window``/``--stride`` are counted in WORDS in chunk mode.

Example (AJO, ~6.3k train):
    python3 build_gapdecay_dataset.py \
        --corpus      aep_docs_collection_v_24-03-2026.json \
        --assignments ajo_doc_resolved_assignments.json \
        --out-dir     ../../data/ajo_gapdecay \
        --unit chunk --window 200 --stride 100 \
        --gap-decay 0.055 --far-gap-floor 20 --target-train 6300 \
        --valtest-cap 1400 --seed 42

Example (AEP, ~13k train):
    python3 build_gapdecay_dataset.py \
        --corpus      aep_docs_collection_v_24-03-2026.json \
        --assignments aep_doc_resolved_assignments.json \
        --out-dir     ../../data/aep_gapdecay \
        --unit sentence --min-sentences 2 --max-sentences 5 \
        --gap-decay 0.055 --far-gap-floor 20 --target-train 13000 \
        --valtest-cap 1400 --seed 42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# --- reuse the original builder's pure (dependency-free) helpers ---
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_bcd", _HERE / "build_classification_data.py"
)
_bcd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bcd)

load_corpus = _bcd.load_corpus
load_assignments = _bcd.load_assignments if hasattr(_bcd, "load_assignments") else None
split_sentences = _bcd.split_sentences
split_docs = _bcd.split_docs


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def word_chunk(text: str, window: int, stride: int) -> list[str]:
    """Tokenizer-free sliding-window chunker (window/stride counted in WORDS)."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        seg = words[i : i + window]
        if seg:
            chunks.append(" ".join(seg))
        if i + window >= len(words):
            break
        i += stride
    return chunks


def _load_assignments_local(path: Path, corpus: dict) -> tuple[list[dict], list[dict]]:
    """Resolve a {url,subchapter,tier} assignments file against the corpus.

    Uses the original builder's ``load_assignments`` when available; otherwise a
    faithful local reimplementation (exact-URL match only — callers should pass a
    pre-resolved assignments file whose urls are exact corpus keys).
    """
    if load_assignments is not None:
        return load_assignments(path, corpus)
    data = json.loads(path.read_text())
    items = data["assignments"] if isinstance(data, dict) and "assignments" in data else data
    resolved, misses = [], []
    for it in items:
        url = it["url"]
        if url not in corpus:
            misses.append({"url": url, "reason": "not in corpus"})
            continue
        sub, tier = it.get("subchapter"), it.get("tier")
        if not sub or tier is None:
            misses.append({"url": url, "reason": "missing subchapter/tier"})
            continue
        resolved.append({
            "url": url,
            "title": corpus[url]["title"] or it.get("title", ""),
            "subchapter": sub,
            "tier": int(tier),
        })
    return resolved, misses


# ---------------------------------------------------------------------------
# Gap-decay cap
# ---------------------------------------------------------------------------
def gap_decay_cap(g: int, base_cap: float, gap_decay: float, floor: int) -> int:
    """cap(g) = max(floor, round(base*(1 - decay*(g-1)))), clamped to >= 0."""
    return max(floor, round(base_cap * (1.0 - gap_decay * (g - 1))))


def sample_index_pairs(n_low: int, n_high: int, k: int, rng: random.Random):
    """Sample up to ``k`` distinct (i, j) index pairs from the n_low x n_high grid."""
    total = n_low * n_high
    if total == 0:
        return []
    if k >= total:
        return [(i, j) for i in range(n_low) for j in range(n_high)]
    picks = rng.sample(range(total), k)
    return [(p // n_high, p % n_high) for p in picks]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--assignments", required=True,
                    help="pre-resolved {url,subchapter,tier} JSON (urls = exact corpus keys)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--unit", choices=["sentence", "chunk"], default="sentence")
    ap.add_argument("--min-sentences", type=int, default=2)
    ap.add_argument("--max-sentences", type=int, default=5)
    ap.add_argument("--window", type=int, default=200, help="chunk window in WORDS")
    ap.add_argument("--stride", type=int, default=100, help="chunk stride in WORDS")
    # gap-decay cap (train)
    ap.add_argument("--base-cap", type=float, default=180.0,
                    help="cap at gap=1 (adjacent tiers). Ignored if --target-train set.")
    ap.add_argument("--gap-decay", type=float, default=0.055)
    ap.add_argument("--far-gap-floor", type=int, default=20)
    ap.add_argument("--target-train", type=int, default=None,
                    help="auto-tune --base-cap (binary search) to hit this train row count")
    ap.add_argument("--target-val", type=int, default=None,
                    help="val row target (default: derived from --target-train via fracs)")
    ap.add_argument("--target-test", type=int, default=None,
                    help="test row target (default: derived from --target-train via fracs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    # tier / gap restriction
    ap.add_argument("--drop-tiers", default="",
                    help="comma-separated tiers to exclude, e.g. '1,15'")
    ap.add_argument("--max-gap", type=int, default=None,
                    help="only pair tiers whose RANK gap <= this (e.g. 4). None = all gaps.")
    # repetition-aware sampling (overrides the per-tier-pair gap-decay cap when set)
    ap.add_argument("--max-text-repeat", type=int, default=None,
                    help="cap the number of pairs each segment appears in (e.g. 8). Enables "
                         "repetition-aware sampling: process gaps 1..max_gap in order, accept a "
                         "pair only if both segments are below the cap.")
    ap.add_argument("--gap-weights", default=None,
                    help="comma-separated per-gap weights for the rep-aware path (len == max_gap), "
                         "e.g. '1.0,0.65,0.4,0.22'. Default: geometric 0.62 decay.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] loading {args.corpus}", file=sys.stderr)
    corpus = load_corpus(Path(args.corpus))
    print(f"[corpus] {len(corpus)} unique URLs", file=sys.stderr)

    resolved, url_misses = _load_assignments_local(Path(args.assignments), corpus)
    print(f"[assignments] {len(resolved)} docs resolved, {len(url_misses)} misses",
          file=sys.stderr)
    if not resolved:
        sys.exit("[fatal] no assignments resolved against the corpus.")

    drop_tiers = {int(t) for t in args.drop_tiers.split(",") if t.strip()}
    if drop_tiers:
        before = len(resolved)
        resolved = [r for r in resolved if r["tier"] not in drop_tiers]
        print(f"[drop-tiers] removed tiers {sorted(drop_tiers)}: {before} -> {len(resolved)} docs",
              file=sys.stderr)

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
        else:
            doc_chunks[r["url"]] = word_chunk(body, args.window, args.stride)
    total = sum(len(c) for c in doc_chunks.values())
    print(f"[segment] unit={args.unit}: {total} segments across {len(doc_chunks)} docs",
          file=sys.stderr)

    # -- global text dedup: keep each unique text once (first occurrence, by
    #    sorted url/idx) so repetition is bounded per UNIQUE TEXT and identical
    #    boilerplate can't leak across splits or blow past the repeat cap. --
    seen_text: set[str] = set()
    dup_dropped = 0

    # -- chunk-level split pooled per tier (deterministic per-tier seed) --
    chunk_entries_by_tier: dict[int, list[dict]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)
    for t in sorted(tier_docs):
        entries: list[dict] = []
        for r in tier_docs[t]:
            for i, ct in enumerate(doc_chunks[r["url"]]):
                norm = " ".join(ct.split())
                if norm in seen_text:
                    dup_dropped += 1
                    continue
                seen_text.add(norm)
                entries.append({"text": ct, "url": r["url"], "chunk_idx": i,
                                "subchapter": r["subchapter"]})
        rng = random.Random((args.seed, "tier", t).__hash__())
        rng.shuffle(entries)
        n = len(entries)
        n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * args.val_frac)) if n - n_test >= 2 else 0
        for i, e in enumerate(entries):
            e["split"] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            split_counts[e["split"]] += 1
        chunk_entries_by_tier[t] = entries
    print(f"[dedup] dropped {dup_dropped} duplicate-text segments; "
          f"{sum(split_counts.values())} unique texts kept", file=sys.stderr)
    print(f"[split] segments per split: {dict(split_counts)}", file=sys.stderr)

    # -- tier-pairs + rank gap (optionally restricted to gap <= max_gap) --
    tiers_sorted = sorted(tier_docs)
    rank = {t: i for i, t in enumerate(tiers_sorted)}
    tier_pairs = [(tiers_sorted[i], tiers_sorted[j])
                  for i in range(len(tiers_sorted)) for j in range(i + 1, len(tiers_sorted))
                  if args.max_gap is None or (j - i) <= args.max_gap]
    if args.max_gap is not None:
        print(f"[gap] restricting to rank gap <= {args.max_gap}: {len(tier_pairs)} tier-pairs",
              file=sys.stderr)

    chunks_by_ts: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for t, entries in chunk_entries_by_tier.items():
        for e in entries:
            chunks_by_ts[(t, e["split"])].append(e)

    # availability per tier-pair per split (single-direction => rows == pairs)
    avail: dict[tuple[int, int, str], int] = {}
    for t_low, t_hi in tier_pairs:
        for sp in ("train", "val", "test"):
            avail[(t_low, t_hi, sp)] = (
                len(chunks_by_ts.get((t_low, sp), [])) * len(chunks_by_ts.get((t_hi, sp), []))
            )

    # -- per-split target row counts --
    train_frac = max(1e-9, 1.0 - args.val_frac - args.test_frac)
    targets = {
        "train": args.target_train,
        "val": args.target_val if args.target_val is not None else (
            round(args.target_train * args.val_frac / train_frac)
            if args.target_train is not None else None),
        "test": args.target_test if args.target_test is not None else (
            round(args.target_train * args.test_frac / train_frac)
            if args.target_train is not None else None),
    }

    def predict_split(sp: str, bc: float) -> int:
        tot = 0
        for t_low, t_hi in tier_pairs:
            g = rank[t_hi] - rank[t_low]
            cap = gap_decay_cap(g, bc, args.gap_decay, args.far_gap_floor)
            tot += min(avail[(t_low, t_hi, sp)], cap)
        return tot

    def tune_base_cap(sp: str, target: int) -> float:
        lo, hi = 0.0, 1.0
        while predict_split(sp, hi) < target and hi < 1e8:
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            if predict_split(sp, mid) < target:
                lo = mid
            else:
                hi = mid
        return round(hi, 3)

    # -- writers / emit --
    writers = {sp: (out_dir / f"directional_{sp}.jsonl").open("w") for sp in ("train", "val", "test")}
    counts: dict[str, int] = defaultdict(int)
    dir_rng = random.Random(args.seed + 7)
    pre_cap: dict[str, int] = {}
    post_cap: dict[str, int] = {}
    base_cap_by_split: dict[str, float] = {}
    gap_hist: dict[str, dict[int, int]] = {sp: defaultdict(int) for sp in ("train", "val", "test")}
    rep_count: dict[str, dict] = {sp: defaultdict(int) for sp in ("train", "val", "test")}

    def emit(sp, low, hi, tl, th):
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

    def seg_key(e):
        return (e["url"], e["chunk_idx"])

    try:
        if args.max_text_repeat is not None:
            # ---- repetition-aware gap-decay sampling ----
            max_gap = args.max_gap if args.max_gap is not None else (len(tiers_sorted) - 1)
            if args.gap_weights:
                weights = [float(x) for x in args.gap_weights.split(",")][:max_gap]
            else:
                weights = [0.62 ** (g - 1) for g in range(1, max_gap + 1)]
            wsum = sum(weights) or 1.0
            R = args.max_text_repeat
            print(f"[rep-aware] max_gap={max_gap} rep_cap={R} weights={[round(w,3) for w in weights]}",
                  file=sys.stderr)
            for sp in ("train", "val", "test"):
                target = targets[sp] or 0
                rc = rep_count[sp]
                for gi, g in enumerate(range(1, max_gap + 1)):
                    gap_target = round(target * weights[gi] / wsum) if target else None
                    cands = []
                    for t_low, t_hi in tier_pairs:
                        if rank[t_hi] - rank[t_low] != g:
                            continue
                        lows = chunks_by_ts.get((t_low, sp), [])
                        highs = chunks_by_ts.get((t_hi, sp), [])
                        for li in range(len(lows)):
                            for hj in range(len(highs)):
                                cands.append((t_low, t_hi, li, hj))
                    crng = random.Random((args.cap_seed, "gap", g, sp).__hash__())
                    crng.shuffle(cands)
                    pre_cap[f"gap{g}/{sp}"] = len(cands)
                    taken = 0
                    for t_low, t_hi, li, hj in cands:
                        if gap_target is not None and taken >= gap_target:
                            break
                        low = chunks_by_ts[(t_low, sp)][li]
                        hi = chunks_by_ts[(t_hi, sp)][hj]
                        kl, kh = seg_key(low), seg_key(hi)
                        if rc[kl] >= R or rc[kh] >= R:
                            continue
                        emit(sp, low, hi, t_low, t_hi)
                        rc[kl] += 1
                        rc[kh] += 1
                        gap_hist[sp][g] += 1
                        taken += 1
                    post_cap[f"gap{g}/{sp}"] = taken
        else:
            # ---- legacy per-tier-pair gap-decay cap ----
            for sp in ("train", "val", "test"):
                if targets[sp] is not None:
                    bc = tune_base_cap(sp, targets[sp])
                    base_cap_by_split[sp] = bc
                    print(f"[tune] {sp}: target={targets[sp]} -> base_cap={bc} "
                          f"(predicted={predict_split(sp, bc)})", file=sys.stderr)
                else:
                    base_cap_by_split[sp] = args.base_cap
            for t_low, t_hi in tier_pairs:
                g = rank[t_hi] - rank[t_low]
                for sp in ("train", "val", "test"):
                    lows = chunks_by_ts.get((t_low, sp), [])
                    highs = chunks_by_ts.get((t_hi, sp), [])
                    key = f"T{t_low}-T{t_hi}/{sp}"
                    pre_cap[key] = len(lows) * len(highs)
                    if not lows or not highs:
                        post_cap[key] = 0
                        continue
                    cap = gap_decay_cap(g, base_cap_by_split[sp], args.gap_decay, args.far_gap_floor)
                    srng = random.Random((args.cap_seed, t_low, t_hi, sp).__hash__())
                    idx_pairs = sample_index_pairs(len(lows), len(highs), cap, srng)
                    post_cap[key] = len(idx_pairs)
                    for i, j in idx_pairs:
                        emit(sp, lows[i], highs[j], t_low, t_hi)
                        gap_hist[sp][g] += 1
    finally:
        for w in writers.values():
            w.close()

    # repetition histogram per split
    def rep_stats(rc: dict) -> dict:
        vals = sorted(rc.values())
        if not vals:
            return {"n_texts": 0}
        n = len(vals)
        return {
            "n_texts": n,
            "max": vals[-1],
            "mean": round(sum(vals) / n, 3),
            "p50": vals[n // 2],
            "p95": vals[min(n - 1, int(0.95 * n))],
            "hist": {k: sum(1 for v in vals if v == k) for k in range(1, max(vals) + 1)},
        }
    repetition_stats = {sp: rep_stats(rep_count[sp]) for sp in ("train", "val", "test")}

    manifest = {
        "builder": "build_gapdecay_dataset.py",
        "unit": args.unit,
        "min_sentences": args.min_sentences if args.unit == "sentence" else None,
        "max_sentences": args.max_sentences if args.unit == "sentence" else None,
        "window_words": args.window if args.unit == "chunk" else None,
        "stride_words": args.stride if args.unit == "chunk" else None,
        "pair_scope": "all" if args.max_gap is None else f"gap<={args.max_gap}",
        "gap_metric": "rank",
        "single_direction": True,
        "drop_tiers": sorted(drop_tiers),
        "max_gap": args.max_gap,
        "max_text_repeat": args.max_text_repeat,
        "gap_weights": args.gap_weights,
        "tiers_kept": tiers_sorted,
        "repetition_stats": repetition_stats,
        "gap_decay": args.gap_decay,
        "far_gap_floor": args.far_gap_floor,
        "targets": targets,
        "base_cap_by_split": base_cap_by_split,
        "seed": args.seed,
        "cap_seed": args.cap_seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "subchapter_to_tier": subchapter_to_tier,
        "n_docs_resolved": len(resolved),
        "docs_per_tier": {t: len(tier_docs[t]) for t in tiers_sorted},
        "tier_rank": rank,
        "segment_split_counts": dict(split_counts),
        "sample_counts": dict(counts),
        "gap_histogram": {sp: dict(sorted(gap_hist[sp].items())) for sp in gap_hist},
        "pre_cap_counts": pre_cap,
        "post_cap_counts": post_cap,
        "n_url_misses": len(url_misses),
    }
    (out_dir / "directional_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[done] wrote samples: {dict(counts)}", file=sys.stderr)
    print(f"[done] train gap histogram: {dict(sorted(gap_hist['train'].items()))}", file=sys.stderr)
    print(f"[done] manifest: {out_dir / 'directional_manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
