"""Generic retrieval evaluation harness.

One CLI (`evaluate-retrieval`) that subsumes three older script flavors:

  1) Test-split retrieval — rank each anchor against the positives of the
     same split. (Equivalent to the legacy `evaluate.py`.)

  2) Full-corpus retrieval — rank each anchor against the positives of all
     splits combined. (Equivalent to `evaluate_full_corpus.py`.) Harder
     because the candidate pool is larger.

  3) External candidate pool with multiple golds per query — e.g. the
     follow-up question task (100 queries × 300 candidates × 3 golds).
     Loaded from a JSON file. (Equivalent to `evaluate_followup_retrieval.py`.)

All three modes share the same code path: build (queries, candidates,
gold_indices) triples, encode via a `Retriever`, compute metrics. Multiple
retrievers can be evaluated in one invocation for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from followup_data import default_root, load
from followup_data.base import DatasetNotDownloaded

from .interfaces import Retriever
from .metrics import compute_metrics, random_baseline_recall
from .retrievers import PretrainedRetriever, TfidfRetriever


# ---------------------------------------------------------------------------
# Eval task builders
# ---------------------------------------------------------------------------
def build_task_from_dataset(
    dataset_name: str,
    split: str,
    candidate_pool: str,
    root: Path | None,
    max_queries: int | None,
) -> tuple[list[str], list[str], list[list[int]], dict]:
    """Construct (queries, candidates, gold_indices, info) from a registered dataset.

    candidate_pool:
        "same-split" — candidates are the positives of THIS split only.
                       Each query has exactly one gold (the diagonal partner).
        "all-splits" — candidates are the positives across all splits.
                       The query's gold remains its same-split partner; harder
                       because there are now far more distractors.
    """
    effective_root = root if root is not None else default_root()

    def _pairs_from(spl: str) -> list[tuple[str, str]]:
        ds = load(dataset_name, root=effective_root, split=spl)
        if not ds.is_downloaded():
            try:
                ds.download()
            except DatasetNotDownloaded as e:
                raise SystemExit(f"dataset not downloaded: {e}")
        out: list[tuple[str, str]] = []
        for ex in ds:
            a = (ex.anchor or "").strip()
            p = (ex.positive or "").strip()
            if a and p:
                out.append((a, p))
        return out

    pairs = _pairs_from(split)
    if max_queries is not None:
        pairs = pairs[:max_queries]
    queries = [a for a, _ in pairs]
    same_split_positives = [p for _, p in pairs]

    if candidate_pool == "same-split":
        candidates = same_split_positives
        gold_indices: list[list[int]] = [[i] for i in range(len(queries))]
    elif candidate_pool == "all-splits":
        # Try all known splits; positives in earlier splits come first so that
        # the query's same-split positive stays at a stable index.
        ds_class = type(load(dataset_name, root=effective_root, split=split))
        splits = list(ds_class.splits)
        # Always include the query's split first so we keep gold index alignment.
        ordered = [split] + [s for s in splits if s != split]
        seen: set[str] = set()
        all_candidates: list[str] = []
        # First add the query split's positives in order — indices 0..len(queries)-1
        for p in same_split_positives:
            if p not in seen:
                seen.add(p)
                all_candidates.append(p)
        # Then add positives from other splits
        for spl in ordered:
            if spl == split:
                continue
            for _, p in _pairs_from(spl):
                if p not in seen:
                    seen.add(p)
                    all_candidates.append(p)
        # Gold index of query i is the index of its same-split positive in all_candidates
        cand_index_of = {c: i for i, c in enumerate(all_candidates)}
        gold_indices = [[cand_index_of[same_split_positives[i]]] for i in range(len(queries))]
        candidates = all_candidates
    else:
        raise ValueError(f"unknown --candidate-pool: {candidate_pool!r}")

    info = {
        "task_type": "dataset_retrieval",
        "dataset": dataset_name,
        "split": split,
        "candidate_pool": candidate_pool,
        "n_queries": len(queries),
        "n_candidates": len(candidates),
        "n_gold_per_query": 1,
    }
    return queries, candidates, gold_indices, info


def build_task_from_followup_file(
    path: Path,
    max_queries: int | None,
) -> tuple[list[str], list[str], list[list[int]], dict]:
    """Load a follow-up benchmark JSON: 1 source question -> N follow-up golds.

    Expected format:
        {
          "metadata": {...},
          "queries": [
            {
              "query_id": 0,
              "topic": "...",
              "source_question": "...",
              "follow_ups": [
                {"id": 0, "text": "..."},
                ...
              ]
            },
            ...
          ]
        }

    All follow-ups from all queries are pooled into the candidate set. For
    each source question, its own follow-ups (and only those) are the golds.
    """
    data = json.load(open(path))
    all_queries = data["queries"]
    if max_queries is not None:
        all_queries = all_queries[:max_queries]

    # Pool: all follow-ups across all queries (always use the full set for
    # consistent comparison, even when --max-queries truncates the query list).
    pool_source = data["queries"]
    candidates: list[str] = []
    global_id_of = {}  # (query_id, local_id) -> global index in candidates
    for q in pool_source:
        for fu in q["follow_ups"]:
            global_id_of[(q["query_id"], fu["id"])] = len(candidates)
            candidates.append(fu["text"])

    queries: list[str] = []
    gold_indices: list[list[int]] = []
    for q in all_queries:
        queries.append(q["source_question"])
        gold_indices.append([global_id_of[(q["query_id"], fu["id"])] for fu in q["follow_ups"]])

    info = {
        "task_type": "followup_retrieval",
        "source_file": str(path),
        "n_queries": len(queries),
        "n_candidates": len(candidates),
        "n_gold_per_query": len(all_queries[0]["follow_ups"]) if all_queries else 0,
    }
    return queries, candidates, gold_indices, info


# ---------------------------------------------------------------------------
# Retriever spec parsing
# ---------------------------------------------------------------------------
def parse_retriever_spec(spec: str) -> Retriever:
    """Parse a `--retriever` CLI string into a Retriever instance.

    Recognized forms:
        biencoder:<checkpoint-path>        Trained bi-encoder
        pretrained[:<hf-model-id>]          Off-the-shelf HF encoder (default bge-small)
        tfidf                               TF-IDF baseline
    """
    if spec == "tfidf":
        return TfidfRetriever()
    if spec.startswith("pretrained"):
        _, _, model_name = spec.partition(":")
        model_name = model_name or "BAAI/bge-small-en-v1.5"
        return PretrainedRetriever(model_name=model_name)
    if spec.startswith("biencoder:"):
        _, _, path = spec.partition(":")
        if not path:
            raise ValueError("biencoder retriever needs a checkpoint path: biencoder:/path/to/best.pt")
        # Lazy import — avoids requiring torch at module-load time.
        from biencoder.evaluation_adapter import BiEncoderRetriever

        return BiEncoderRetriever(checkpoint_path=path)
    raise ValueError(f"unknown retriever spec: {spec!r}")


# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------
def evaluate(
    retriever: Retriever,
    queries: list[str],
    candidates: list[str],
    gold_indices: list[list[int]],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict:
    """Encode queries + candidates, build similarity matrix, compute metrics."""
    q = retriever.encode_anchors(queries)
    c = retriever.encode_candidates(candidates)
    similarity = q @ c.T
    return compute_metrics(similarity, gold_indices, k_values=k_values)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_metrics(name: str, metrics: dict, k_values: list[int]) -> None:
    print(f"  {name}")
    print(f"    MRR:        {metrics['mrr']:.4f}")
    print(f"    Mean rank:  {metrics['mean_rank']:.2f}")
    print(f"    Median rank:{metrics['median_rank']:.1f}")
    for k in k_values:
        print(f"    Recall@{k:<3}{metrics['recall_at_k'][k]:.4f}    Hit@{k:<3}{metrics['hit_at_k'][k]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate one or more retrievers on a retrieval task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- Task: pick exactly one ----
    task_group = ap.add_argument_group("task")
    task_group.add_argument("--dataset",
                            help="Registered dataset name (e.g. aep_causal).")
    task_group.add_argument("--split", default="test",
                            help="Dataset split to use as queries.")
    task_group.add_argument("--candidate-pool", default="same-split",
                            choices=["same-split", "all-splits"],
                            help="For --dataset mode: pool of candidates to rank against.")
    task_group.add_argument("--followup-file",
                            help="Path to a follow-up benchmark JSON (overrides --dataset).")
    task_group.add_argument("--root", default=None,
                            help="Override data root.")
    task_group.add_argument("--max-queries", type=int, default=None,
                            help="Truncate to first N queries (smoke test).")
    # ---- Retrievers ----
    ap.add_argument("--retriever", action="append", default=None,
                    help="A retriever spec. Pass multiple times to compare. "
                         "Forms: biencoder:<path>, pretrained[:<hf-id>], tfidf")
    ap.add_argument("--k", action="append", type=int, default=None,
                    help="Recall@K / Hit@K cut-off (repeatable, default 1,3,5,10).")
    # ---- Output ----
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: <dataset>_<split>_retrieval.json or "
                         "<followup>_retrieval.json.")
    args = ap.parse_args()

    if args.followup_file is None and args.dataset is None:
        raise SystemExit("must provide one of --dataset or --followup-file")
    if not args.retriever:
        raise SystemExit("must provide at least one --retriever")

    k_values = args.k or [1, 3, 5, 10]
    root = Path(args.root) if args.root else None

    # --- Build task ---
    if args.followup_file:
        path = Path(args.followup_file)
        queries, candidates, gold_indices, info = build_task_from_followup_file(path, args.max_queries)
        default_out = path.with_suffix(".retrieval_results.json")
    else:
        queries, candidates, gold_indices, info = build_task_from_dataset(
            args.dataset, args.split, args.candidate_pool, root, args.max_queries,
        )
        default_out = Path(f"{args.dataset}_{args.split}_retrieval_results.json")

    print(f"[task] {info['task_type']}  queries={info['n_queries']}  candidates={info['n_candidates']}")

    # --- Run each retriever ---
    all_results: dict = {"task": info, "retrievers": {}, "k_values": list(k_values)}
    for spec in args.retriever:
        print(f"\n[retriever] {spec}")
        retriever = parse_retriever_spec(spec)
        metrics = evaluate(retriever, queries, candidates, gold_indices, k_values=k_values)
        all_results["retrievers"][spec] = metrics
        _print_metrics(spec, metrics, k_values)

    # --- Random baseline (analytical) ---
    rand_recall = random_baseline_recall(info["n_candidates"], k_values, n_gold=info["n_gold_per_query"])
    all_results["retrievers"]["random"] = {
        "recall_at_k": rand_recall,
        "note": "analytical: K * n_gold / n_candidates (capped at 1)",
    }
    print("\n[retriever] random (analytical)")
    for k in k_values:
        print(f"    Recall@{k:<3}{rand_recall[k]:.4f}")

    # --- Save ---
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[done] results saved to {out_path}")


if __name__ == "__main__":
    main()
