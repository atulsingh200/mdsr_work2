"""Minimal example: score (anchor, positive) and (anchor, negative) pairs.

The framing: a good causal embedding should give

    sim(anchor, positive) > sim(anchor, negative)

for most triples. This script measures that pairwise accuracy on a few
hundred triples sampled from a dataset, using a sentence-transformers
model as the scorer (any model with .encode() works).

Usage:
    uv run python examples/score_pairs.py --dataset clariq --k 4 -n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from followup_data import default_root, load
from followup_data.negatives import with_random_negatives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--root", default=str(default_root()))
    args = p.parse_args()

    try:
        ds = load(args.dataset, root=Path(args.root), split=args.split)
    except ValueError:
        ds = load(args.dataset, root=Path(args.root))
    if not ds.is_downloaded():
        ds.download()

    pairs = list(ds.head(args.n * 2))
    triples = list(with_random_negatives(pairs, k=args.k, seed=0))[: args.n]
    if not triples:
        print(f"no triples produced for {args.dataset}")
        return

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer(args.model)

    anchors = [t.anchor for t in triples]
    positives = [t.positive for t in triples]
    negatives_flat: list[str] = []
    for t in triples:
        negatives_flat.extend(t.negatives)

    a_emb = model.encode(anchors, convert_to_tensor=True, show_progress_bar=False)
    p_emb = model.encode(positives, convert_to_tensor=True, show_progress_bar=False)
    n_emb = model.encode(negatives_flat, convert_to_tensor=True, show_progress_bar=False)

    pos_sim = cos_sim(a_emb, p_emb).diagonal()
    correct = 0
    total = 0
    for i, t in enumerate(triples):
        for j, _ in enumerate(t.negatives):
            neg_idx = i * args.k + j
            ns = float(cos_sim(a_emb[i], n_emb[neg_idx]))
            if float(pos_sim[i]) > ns:
                correct += 1
            total += 1

    print(f"dataset      : {args.dataset}")
    print(f"model        : {args.model}")
    print(f"triples      : {len(triples)}")
    print(f"k negatives  : {args.k}")
    print(f"pairwise acc : {correct}/{total} = {correct / total:.3f}")
    print(f"mean pos sim : {float(pos_sim.mean()):.3f}")


if __name__ == "__main__":
    main()
