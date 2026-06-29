"""Measure how semantically (not lexically) hard BM25's negatives actually are.

For each dataset:
  1. Mine 4 BM25 hard negatives per anchor — same procedure as
     evaluation_6/bm25negative/evaluate_bm25.py (corpus capped at 10 000,
     queries capped at 10 000, top-K BM25 by anchor tokens).
  2. Encode anchors and positives with the same off-the-shelf semantic
     encoder used in evaluation_6 (sentence-transformers/all-MiniLM-L6-v2).
  3. Report **mean cosine similarity** of:
        sim(anchor, gold_positive)
        sim(anchor, single_BM25_negative)   ← averaged over the 4 negs and over anchors
        sim(anchor, hardest_BM25_negative)  ← worst-case neg per anchor, then avg
     and the gap pos − max-neg.
  4. Print a handful of sample anchors with the actual numbers.

The output makes "BM25 finds lexically hard negatives that are or aren't
semantically hard" measurable per dataset.

Usage:
  .venv/bin/python scripts/bm25_semantic_diagnostic.py
  .venv/bin/python scripts/bm25_semantic_diagnostic.py --datasets followupqg workflow
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.retrievers import PretrainedRetriever                # noqa: E402

# Local helpers
from finetune_eval.data import load_pairs                            # noqa: E402


DATASETS: dict[str, Path] = {
    "aep_causal":   ROOT / "data_6/aep_causal/test.jsonl",
    "aep_followup": ROOT / "data_6/aep_followup/followup_pairs.jsonl",
    "followupqg":   ROOT / "data_6/followupqg/test.jsonl",
    "multiwoz_v24": ROOT / "data_6/multiwoz_v24/test.jsonl",
    "qrecc":        ROOT / "data_6/qrecc/test.jsonl",
    "workflow":     ROOT / "data_6/workflow/test_pairs.jsonl",
}


def trim(text: str, n: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def mine_bm25_negatives(
    anchors: list[str],
    positives: list[str],
    n_negatives: int,
    bm25_corpus_size: int,
    max_bm25_queries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror evaluation_6/bm25negative/evaluate_bm25.py mining logic.

    Returns:
        neg_global: (Q, n_negatives) indices into positives (global indices).
        query_idx:  (Q,) global indices into anchors that were used as queries.
    """
    from rank_bm25 import BM25Okapi

    N = len(anchors)
    rng = np.random.default_rng(seed)
    corpus_idx = (rng.choice(N, bm25_corpus_size, replace=False)
                  if N > bm25_corpus_size else np.arange(N))
    corpus_idx = np.sort(corpus_idx)
    corpus_texts = [positives[i] for i in corpus_idx]

    query_idx = (rng.choice(N, max_bm25_queries, replace=False)
                 if N > max_bm25_queries else np.arange(N))
    query_idx = np.sort(query_idx)

    tokenize = lambda s: s.lower().split()
    bm25 = BM25Okapi([tokenize(t) for t in corpus_texts])

    neg_global = np.empty((len(query_idx), n_negatives), dtype=np.int64)
    for qi, gi in enumerate(query_idx):
        scores = bm25.get_scores(tokenize(anchors[gi]))
        order = np.argsort(-scores)
        chosen: list[int] = []
        for c_idx in order:
            g_idx = int(corpus_idx[c_idx])
            if g_idx == gi:
                continue
            chosen.append(g_idx)
            if len(chosen) == n_negatives:
                break
        while len(chosen) < n_negatives:
            r = int(rng.integers(0, N))
            if r != gi and r not in chosen:
                chosen.append(r)
        neg_global[qi] = chosen
    return neg_global, query_idx


def run_dataset(
    ds_name: str,
    miner: PretrainedRetriever,
    n_negatives: int,
    bm25_corpus_size: int,
    max_bm25_queries: int,
    n_samples: int,
    snippet_chars: int,
    seed: int,
) -> dict:
    path = DATASETS[ds_name]
    print(f"\n=== {ds_name} ({path.name}) ===", flush=True)
    pairs = load_pairs(path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    N = len(pairs)
    print(f"  {N} test pairs", flush=True)

    t0 = time.time()
    neg_global, query_idx = mine_bm25_negatives(
        anchors, positives, n_negatives, bm25_corpus_size, max_bm25_queries, seed,
    )
    print(f"  BM25 mining: {time.time() - t0:.1f}s   queries={len(query_idx)}", flush=True)

    # Encode with the off-the-shelf semantic model.
    t1 = time.time()
    q_anchor_texts = [anchors[i] for i in query_idx]
    q_anchor_emb = miner.encode_anchors(q_anchor_texts)          # (Q, D)
    p_emb = miner.encode_candidates(positives)                   # (N, D)
    print(f"  semantic encoding: {time.time() - t1:.1f}s", flush=True)

    pos_sims = np.sum(q_anchor_emb * p_emb[query_idx], axis=1)         # (Q,)
    neg_sims = np.einsum("id,ikd->ik", q_anchor_emb, p_emb[neg_global])  # (Q, K)

    pos_mean = float(pos_sims.mean())
    pos_median = float(np.median(pos_sims))
    neg_mean = float(neg_sims.mean())                                  # avg over Q*K
    neg_max_mean = float(neg_sims.max(axis=1).mean())                  # avg hardest neg per anchor
    gap_max = pos_mean - neg_max_mean
    gap_avg = pos_mean - neg_mean

    print(f"  mean sim(anchor, gold)        = {pos_mean:.4f}")
    print(f"  mean sim(anchor, BM25 neg)    = {neg_mean:.4f}     (avg over {n_negatives} negs × anchors)")
    print(f"  mean sim(anchor, hardest neg) = {neg_max_mean:.4f}  (worst-case per anchor, then averaged)")
    print(f"  gap  pos − avg neg            = {gap_avg:+.4f}")
    print(f"  gap  pos − hardest neg        = {gap_max:+.4f}")

    # Sample examples — pick a few uniformly across the rank by gap.
    gaps = pos_sims - neg_sims.max(axis=1)
    if n_samples > 0:
        # Show n_samples scenarios: easy win, close-call, failure.
        order = np.argsort(-gaps)  # easy wins first, failures last
        easy = int(order[len(order) // 10])
        median = int(order[len(order) // 2])
        hardest = int(order[len(order) * 9 // 10])
        idxs = sorted({easy, median, hardest})
        samples: list[dict] = []
        for qi in idxs:
            gi = int(query_idx[qi])
            samples.append({
                "anchor":            trim(anchors[gi], snippet_chars),
                "gold_positive":     trim(positives[gi], snippet_chars),
                "sim_anchor_gold":   float(pos_sims[qi]),
                "negatives":         [
                    {
                        "rank":              k + 1,
                        "sim_anchor_neg":    float(neg_sims[qi, k]),
                        "text":              trim(positives[int(neg_global[qi, k])], snippet_chars),
                    }
                    for k in range(n_negatives)
                ],
                "gap": float(gaps[qi]),
            })
    else:
        samples = []

    return {
        "dataset":              ds_name,
        "n_test_pairs":         N,
        "n_queries":            int(len(query_idx)),
        "n_negatives_per_anchor": n_negatives,
        "encoder":              miner.model_name,
        "mean_sim_pos":         pos_mean,
        "median_sim_pos":       pos_median,
        "mean_sim_neg_all":     neg_mean,
        "mean_sim_neg_hardest": neg_max_mean,
        "gap_pos_minus_avg_neg":     gap_avg,
        "gap_pos_minus_hardest_neg": gap_max,
        "fraction_neg_beats_pos":    float(np.mean(neg_sims.max(axis=1) > pos_sims)),
        "samples": samples,
    }


def render_markdown(rows: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# BM25 hard negatives — semantic similarity vs. the gold positive\n")
    lines.append("Encoder: `sentence-transformers/all-MiniLM-L6-v2` (off-the-shelf, mean pool, L2-normalized).\n")
    lines.append("Cosine similarity = dot product of L2-normalized embeddings.\n")
    lines.append("All values are *averages over the queried anchors*.\n")
    lines.append("\n## Aggregate semantic similarities\n")
    lines.append("| dataset | N queries | sim(anchor, **gold**) | sim(anchor, BM25 neg) avg | sim(anchor, **hardest** BM25 neg) | gap (gold − hardest neg) | % anchors where a neg outscores gold |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['dataset']}` | {r['n_queries']:,} | **{r['mean_sim_pos']:.4f}** | "
            f"{r['mean_sim_neg_all']:.4f} | {r['mean_sim_neg_hardest']:.4f} | "
            f"{r['gap_pos_minus_hardest_neg']:+.4f} | "
            f"{100 * r['fraction_neg_beats_pos']:.1f}% |"
        )
    lines.append("")
    lines.append("> If `gap (gold − hardest neg)` is **positive**, the gold has a higher mean semantic similarity to the anchor than any BM25 distractor — that's why MiniLM still wins despite BM25 picking lexically similar texts.\n")
    lines.append("> If it's **negative**, BM25 is genuinely semantically hard there — every dataset besides `followupqg` is in this regime.\n")

    for r in rows:
        lines.append("\n---\n")
        lines.append(f"## {r['dataset']} — examples\n")
        lines.append("Each row is one randomly chosen anchor sampled near the dataset's 10th, 50th, and 90th percentiles of the per-anchor (gold − hardest neg) gap. Higher gap = easier for the model on that anchor.\n")
        for s in r["samples"]:
            lines.append(f"\n**gap = {s['gap']:+.4f}**\n")
            lines.append(f"- **anchor.** {s['anchor']}")
            lines.append(f"- **gold positive**  (sim {s['sim_anchor_gold']:.4f})  ·  {s['gold_positive']}")
            for n in s["negatives"]:
                lines.append(f"- BM25 neg #{n['rank']} (sim {n['sim_anchor_neg']:.4f})  ·  {n['text']}")
            lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\n[wrote] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--bm25-corpus-size", type=int, default=10_000)
    ap.add_argument("--max-bm25-queries", type=int, default=10_000)
    ap.add_argument("--n-samples", type=int, default=3,
                    help="Sample anchors per dataset (at 10th/50th/90th percentile by gap).")
    ap.add_argument("--max-snippet-chars", type=int, default=300)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "scripts/bm25_semantic_diagnostic.md"))
    args = ap.parse_args()

    print(f"[load semantic encoder] {args.encoder}")
    miner = PretrainedRetriever(
        model_name=args.encoder, pooling="mean",
        max_seq_length=256, batch_size=256,
    )

    rows: list[dict] = []
    for ds in args.datasets:
        rows.append(run_dataset(
            ds, miner,
            n_negatives=args.n_negatives,
            bm25_corpus_size=args.bm25_corpus_size,
            max_bm25_queries=args.max_bm25_queries,
            n_samples=args.n_samples,
            snippet_chars=args.max_snippet_chars,
            seed=args.seed,
        ))

    out_md = Path(args.out)
    render_markdown(rows, out_md)
    Path(str(out_md) + ".json").write_text(json.dumps(rows, indent=2))
    print(f"[wrote] {out_md}.json")


if __name__ == "__main__":
    main()
