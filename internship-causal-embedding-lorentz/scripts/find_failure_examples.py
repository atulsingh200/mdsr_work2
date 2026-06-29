"""Surface concrete failure examples — cases where a negative beats the positive.

Reports two failure modes for the followupqg and workflow datasets:

  (a) BM25 hard negatives, off-the-shelf all-MiniLM-L6-v2 encoder
      (mirrors Experiment 2 from ppt.md).
  (b) Semantic kNN hard negatives, fine-tuned BERT bi-encoder
      (mirrors Experiment 6 from ppt.md).

For each mode we pick the worst k examples by gap = max_neg_sim − pos_sim and
print the anchor / positive / negatives texts so the user can inspect what
the model is being fooled by.

Usage:
  .venv/bin/python scripts/find_failure_examples.py
  .venv/bin/python scripts/find_failure_examples.py \\
      --datasets followupqg workflow --n-examples 3 --max-snippet-chars 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from biencoder.model import BiEncoder, get_tokenizer, tokenize_texts        # noqa: E402

from evaluation.retrievers import PretrainedRetriever                       # noqa: E402

from finetune_eval.data import load_pairs                                   # noqa: E402
from finetune_eval.datasets import split_path                               # noqa: E402


def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trim(text: str, n_chars: int) -> str:
    """Single-line, ellipsis-truncated text for log output."""
    flat = " ".join(text.split())
    if len(flat) <= n_chars:
        return flat
    return flat[: n_chars - 1] + "…"


# ---------------------------------------------------------------------------
# (a) BM25 hard-negative failures with off-the-shelf MiniLM
# ---------------------------------------------------------------------------
def bm25_failures(
    dataset: str,
    n_examples: int,
    n_negatives: int,
    bm25_corpus_size: int,
    max_bm25_queries: int,
    snippet_chars: int,
) -> list[dict]:
    from rank_bm25 import BM25Okapi
    print(f"\n=== BM25 failures on {dataset} (MiniLM-L6-v2, off-the-shelf) ===")

    test_path = split_path(dataset, "test")
    pairs = load_pairs(test_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    print(f"  loaded {len(pairs)} test pairs from {test_path.name}")

    # Same corpus/query selection as evaluation_6/bm25negative/evaluate_bm25.py
    rng = np.random.default_rng(0)
    N = len(pairs)
    if N > bm25_corpus_size:
        corpus_idx = rng.choice(N, size=bm25_corpus_size, replace=False)
        corpus_idx = np.sort(corpus_idx)
    else:
        corpus_idx = np.arange(N)
    corpus_texts = [positives[i] for i in corpus_idx]
    print(f"  BM25 corpus size: {len(corpus_texts)}", flush=True)

    if N > max_bm25_queries:
        query_idx = rng.choice(N, size=max_bm25_queries, replace=False)
        query_idx = np.sort(query_idx)
    else:
        query_idx = np.arange(N)
    print(f"  query count: {len(query_idx)}", flush=True)

    tokenize = lambda s: s.lower().split()
    bm25 = BM25Okapi([tokenize(t) for t in corpus_texts])
    print(f"  BM25 index built", flush=True)

    # corpus_idx[i] maps the BM25 corpus row i to the GLOBAL positive index.
    corpus_to_global = corpus_idx

    neg_global = np.empty((len(query_idx), n_negatives), dtype=np.int64)
    for qi, gi in enumerate(query_idx):
        scores = bm25.get_scores(tokenize(anchors[gi]))
        order = np.argsort(-scores)
        chosen: list[int] = []
        for c_idx in order:
            g_idx = int(corpus_to_global[c_idx])
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
    print(f"  BM25 negatives mined")

    # Encode with MiniLM
    print(f"  encoding with MiniLM-L6-v2…")
    miner = PretrainedRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="mean", max_seq_length=256, batch_size=128,
    )
    a_emb = miner.encode_anchors([anchors[i] for i in query_idx])     # (Q, D)
    p_emb = miner.encode_candidates(positives)                         # (N, D)

    pos_sims = np.sum(a_emb * p_emb[query_idx], axis=1)
    neg_sims = np.einsum("id,ikd->ik", a_emb, p_emb[neg_global])
    max_neg = neg_sims.max(axis=1)
    gap = max_neg - pos_sims         # positive when negative beats positive
    fails = np.where(gap > 0)[0]
    print(f"  failures: {len(fails)} / {len(query_idx)} "
          f"({100 * len(fails) / max(len(query_idx), 1):.1f} %)")

    # Top-k worst by gap
    worst = fails[np.argsort(-gap[fails])][:n_examples]
    out: list[dict] = []
    for qi in worst:
        gi = int(query_idx[qi])
        neg_ids = neg_global[qi].tolist()
        winning_neg = int(neg_sims[qi].argmax())
        out.append({
            "dataset":      dataset,
            "mode":         "bm25",
            "anchor":       trim(anchors[gi], snippet_chars),
            "positive":     trim(positives[gi], snippet_chars),
            "positive_sim": float(pos_sims[qi]),
            "negatives":    [
                {
                    "rank":          k + 1,
                    "sim":           float(neg_sims[qi, k]),
                    "wins":          k == winning_neg,
                    "is_max_winner": k == winning_neg,
                    "text":          trim(positives[neg_ids[k]], snippet_chars),
                }
                for k in range(n_negatives)
            ],
            "gap": float(gap[qi]),
        })
    return out


# ---------------------------------------------------------------------------
# (b) Semantic-hard-negative failures with the fine-tuned BERT bi-encoder
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_with_biencoder(
    model: BiEncoder, tokenizer, texts: list[str], which: str,
    max_length: int, batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        ids, mask = tokenize_texts(tokenizer, chunk, max_length)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        emb = (model.encode_anchor(ids, mask) if which == "anchor"
               else model.encode_positive(ids, mask))
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


def semantic_hardneg_failures(
    dataset: str,
    results_dir: Path,
    n_examples: int,
    snippet_chars: int,
) -> list[dict]:
    print(f"\n=== Semantic hard-neg failures on {dataset} (fine-tuned BERT) ===")
    ds_dir = results_dir / dataset
    ckpt_path = ds_dir / "checkpoint_best.pt"
    pairs_path = ds_dir / "test_pairs.jsonl"
    hn_path = ds_dir / "test_hard_negatives.npy"
    if not (ckpt_path.exists() and pairs_path.exists() and hn_path.exists()):
        print(f"  SKIP: missing artifacts in {ds_dir}")
        return []

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = auto_device()
    tokenizer = get_tokenizer(cfg["backbone"])
    model = BiEncoder(
        model_name=cfg["backbone"], pooling_strategy=cfg["pooling"],
        anchor_prefix="", positive_prefix="",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pairs = load_pairs(pairs_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    hn_idx = np.load(hn_path)
    print(f"  {len(pairs)} pairs · K={hn_idx.shape[1]} sem hard negs each")

    a_emb = encode_with_biencoder(model, tokenizer, anchors,   "anchor",   cfg["max_seq_length"], 128, device)
    p_emb = encode_with_biencoder(model, tokenizer, positives, "positive", cfg["max_seq_length"], 128, device)

    pos_sims = np.sum(a_emb * p_emb, axis=1)
    neg_sims = np.einsum("id,ikd->ik", a_emb, p_emb[hn_idx])
    max_neg = neg_sims.max(axis=1)
    gap = max_neg - pos_sims
    fails = np.where(gap > 0)[0]
    print(f"  failures: {len(fails)} / {len(pairs)} "
          f"({100 * len(fails) / max(len(pairs), 1):.1f} %)")

    worst = fails[np.argsort(-gap[fails])][:n_examples]
    out: list[dict] = []
    for i in worst:
        winning_neg = int(neg_sims[i].argmax())
        neg_ids = hn_idx[i].tolist()
        out.append({
            "dataset":      dataset,
            "mode":         "semantic_hard",
            "anchor":       trim(anchors[i], snippet_chars),
            "positive":     trim(positives[i], snippet_chars),
            "positive_sim": float(pos_sims[i]),
            "negatives": [
                {
                    "rank":          k + 1,
                    "sim":           float(neg_sims[i, k]),
                    "wins":          k == winning_neg,
                    "is_max_winner": k == winning_neg,
                    "text":          trim(positives[neg_ids[k]], snippet_chars),
                }
                for k in range(hn_idx.shape[1])
            ],
            "gap": float(gap[i]),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def render_markdown(examples: list[dict], out_path: Path) -> None:
    """Pretty-print examples as a markdown report."""
    lines: list[str] = []
    lines.append("# Failure examples — where the model picks a negative over the positive\n")
    lines.append("For each example: the anchor, the gold positive (along with the model's "
                 "similarity score), and the 4 negatives (BM25 or semantic kNN, depending "
                 "on the experiment). The negative marked **(beats pos)** is the one whose "
                 "similarity to the anchor exceeded the gold positive's — that's what causes "
                 "the failure.\n")
    by_block: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        by_block.setdefault((ex["mode"], ex["dataset"]), []).append(ex)
    titles = {
        ("bm25",          "followupqg"): "## BM25 hard negatives · followupqg · MiniLM-L6-v2 (Exp 2)",
        ("bm25",          "workflow"):   "## BM25 hard negatives · workflow · MiniLM-L6-v2 (Exp 2)",
        ("semantic_hard", "followupqg"): "## Semantic kNN hard negatives · followupqg · fine-tuned BERT (Exp 6)",
        ("semantic_hard", "workflow"):   "## Semantic kNN hard negatives · workflow · fine-tuned BERT (Exp 6)",
    }
    order = [
        ("bm25", "followupqg"),
        ("bm25", "workflow"),
        ("semantic_hard", "followupqg"),
        ("semantic_hard", "workflow"),
    ]
    for key in order:
        block = by_block.get(key, [])
        if not block:
            continue
        lines.append("\n---\n")
        lines.append(titles[key])
        for k, ex in enumerate(block, 1):
            lines.append(f"\n### Example {k}  (gap = +{ex['gap']:.4f})\n")
            lines.append(f"**Anchor.** {ex['anchor']}\n")
            lines.append(f"**Gold positive  (sim = {ex['positive_sim']:.4f})**  \n> {ex['positive']}\n")
            lines.append("**Negatives:**\n")
            for n in ex["negatives"]:
                tag = "  **(beats pos)**" if n["is_max_winner"] else ""
                lines.append(f"- neg #{n['rank']}  sim = {n['sim']:.4f}{tag}  \n  > {n['text']}")
            lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"\n[wrote] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["followupqg", "workflow"])
    ap.add_argument("--n-examples", type=int, default=3, help="Worst-K examples per (mode, dataset).")
    ap.add_argument("--n-negatives", type=int, default=4, help="K for BM25 negatives.")
    ap.add_argument("--bm25-corpus-size", type=int, default=10_000)
    ap.add_argument("--max-bm25-queries", type=int, default=10_000)
    ap.add_argument("--max-snippet-chars", type=int, default=400,
                    help="Per-field truncation for the report.")
    ap.add_argument("--results-dir", default=str(ROOT / "finetune_eval/results"))
    ap.add_argument("--out", default=str(ROOT / "scripts/failure_examples.md"))
    args = ap.parse_args()

    examples: list[dict] = []
    for ds in args.datasets:
        examples += bm25_failures(
            ds, args.n_examples, args.n_negatives,
            args.bm25_corpus_size, args.max_bm25_queries, args.max_snippet_chars,
        )
        examples += semantic_hardneg_failures(
            ds, Path(args.results_dir), args.n_examples, args.max_snippet_chars,
        )

    out_path = Path(args.out)
    render_markdown(examples, out_path)
    Path(str(out_path) + ".json").write_text(json.dumps(examples, indent=2))
    print(f"[wrote] {out_path}.json")


if __name__ == "__main__":
    main()
