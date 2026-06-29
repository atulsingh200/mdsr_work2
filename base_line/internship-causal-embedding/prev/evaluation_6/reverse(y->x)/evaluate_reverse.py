"""Reverse-pair evaluation: 1 negative per anchor, where the negative is the
*reverse* of the positive pair.

For each pair (a, p) in the test split:
  * positive score = sim( query_enc(a) , passage_enc(p) )      # (a -> p)
  * negative score = sim( query_enc(p) , passage_enc(a) )      # (p -> a)

This only makes sense for **asymmetric** encoders that apply different
transformations to the query side and the passage side (e.g. different
text prefixes or different prompts). For a symmetric encoder, the two
scores are identical and the task is trivially undefined.

Reported metrics per (model, dataset) — same set as evaluation_6/results.json,
except `n_negatives = 1` and the pool is {gold, reverse} so
`n_candidates_per_query = 2`:

  * auc              — ROC-AUC over (pos, neg) score pairs (label 1 vs 0)
  * precision_at_1   — fraction of pairs where pos > neg (strict, ties counted as loss)
  * mrr              — mean reciprocal rank of the gold (rank ∈ {1, 2})
  * mean_rank        — mean rank of the gold
  * median_rank      — median rank of the gold
  * recall_at_k      — {1, 3, 5, 10}; same convention as parent eval
  * hit_at_k         — same as recall_at_k (one gold per anchor)

Rank counting follows `evaluation_6.metrics_extra.random_pool_retrieval_metrics`:
`rank_i = #{distractors with sim STRICTLY > gold} + 1`. So a tie counts as
rank = 1 for retrieval (optimistic), but as a *loss* for precision_at_1
(strict `>`). This is the same inconsistency the parent eval has — kept
on purpose so the numbers are directly comparable.

Extra fields (not in parent) for diagnostic value:
  * mean_margin      — mean(pos - neg)
  * ties             — fraction where pos == neg
  * mean_pos_sim     — mean cosine of the forward direction
  * mean_neg_sim     — mean cosine of the reverse direction

Models (same as evaluation_6/results.json — symmetric, off-the-shelf):
  * all-MiniLM-L6-v2  — sentence-transformers/all-MiniLM-L6-v2  (mean pool, no prefix)
  * bert-base-uncased — google-bert/bert-base-uncased           (CLS pool,  no prefix)

NB: both encoders are symmetric — same encoder, no per-side prefix — so
`sim(a, p) ≡ sim(p, a)` exactly. The reverse-pair "negative" therefore
gets the same score as the positive on every pair: accuracy ≈ 0,
AUC = 0.5, ties = 1.0. The results are reported so that the symmetric
case is on the record, not because there's a real signal to measure.

Datasets (test split each, from data_6/):
  aep_causal, aep_followup, followupqg, multiwoz_v24, qrecc, workflow

Usage:
  .venv/bin/python 'evaluation_6/reverse(y->x)/evaluate_reverse.py'
  .venv/bin/python 'evaluation_6/reverse(y->x)/evaluate_reverse.py' \\
        --models bge-small --datasets aep_causal qrecc
  .venv/bin/python 'evaluation_6/reverse(y->x)/evaluate_reverse.py' --max-pairs 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "all-MiniLM-L6-v2": {
        "hf_id":           "sentence-transformers/all-MiniLM-L6-v2",
        "pooling":         "mean",
        "max_seq_length":  256,
        "anchor_prefix":   "",
        "positive_prefix": "",
    },
    "bert-base-uncased": {
        "hf_id":           "google-bert/bert-base-uncased",
        "pooling":         "cls",
        "max_seq_length":  512,
        "anchor_prefix":   "",
        "positive_prefix": "",
    },
}

DATASETS: dict[str, Path] = {
    "aep_causal":   ROOT / "data_6/aep_causal/test.jsonl",
    "aep_followup": ROOT / "data_6/aep_followup/followup_pairs.jsonl",
    "followupqg":   ROOT / "data_6/followupqg/test.jsonl",
    "multiwoz_v24": ROOT / "data_6/multiwoz_v24/test.jsonl",
    "qrecc":        ROOT / "data_6/qrecc/test.jsonl",
    "workflow":     ROOT / "data_6/workflow/test_pairs.jsonl",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = (row.get("anchor") or "").strip()
            p = (row.get("positive") or "").strip()
            if a and p:
                pairs.append((a, p))
    return pairs


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class PrefixedEncoder:
    """HF encoder that supports a separate prefix for queries vs. passages.

    We deliberately don't reuse `evaluation.retrievers.PretrainedRetriever`
    because it has no prefix handling; the reverse-pair eval is meaningless
    without that asymmetry.
    """

    def __init__(
        self,
        hf_id: str,
        pooling: str,
        max_seq_length: int,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).to(self.device)
        self.model.eval()
        self.pooling = pooling
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _pool(self, last_hidden, mask):
        if self.pooling == "cls":
            return last_hidden[:, 0]
        m = mask.unsqueeze(-1).float()
        s = (last_hidden * m).sum(dim=1)
        c = m.sum(dim=1).clamp(min=1e-9)
        return s / c

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                last_hidden = self.model(**enc).last_hidden_state
                embs = self._pool(last_hidden, enc["attention_mask"])
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            chunks.append(embs.cpu().numpy())
        if not chunks:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Per-(model, dataset) eval
# ---------------------------------------------------------------------------
def evaluate_on_dataset(
    model_key: str,
    model_cfg: dict,
    dataset_name: str,
    dataset_path: Path,
    encoder: PrefixedEncoder,
    max_pairs: int | None,
    k_values: list[int],
) -> dict:
    print(f"\n[{model_key} × {dataset_name}]")

    pairs = load_pairs(dataset_path)
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
        print(f"  truncated to first {max_pairs} pairs")

    if len(pairs) < 2:
        print(f"  SKIP: only {len(pairs)} valid pairs in {dataset_path}")
        return {"model": model_key, "dataset": dataset_name, "error": "not enough pairs"}

    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    print(f"  loaded {len(pairs)} pairs (file: {dataset_path.name})")

    qp = model_cfg["anchor_prefix"]
    pp = model_cfg["positive_prefix"]

    t0 = time.time()
    # Forward direction: a is query, p is passage
    a_q = encoder.encode([qp + a for a in anchors])
    p_d = encoder.encode([pp + p for p in positives])
    # Reverse direction: p is query, a is passage  (this is the "negative")
    p_q = encoder.encode([qp + p for p in positives])
    a_d = encoder.encode([pp + a for a in anchors])
    enc_time = time.time() - t0
    print(f"  encoded {len(pairs) * 4} texts in {enc_time:.1f}s (dim={a_q.shape[1]})")

    # ---- pair-level scoring ----
    pos_score = np.einsum("nd,nd->n", a_q, p_d)
    neg_score = np.einsum("nd,nd->n", p_q, a_d)
    margin = pos_score - neg_score

    precision_at_1 = float((pos_score > neg_score).mean())  # strict >
    ties = float((pos_score == neg_score).mean())
    scores = np.concatenate([pos_score, neg_score])
    labels = np.concatenate([np.ones(len(pairs)), np.zeros(len(pairs))])
    auc = float(roc_auc_score(labels, scores))

    # ---- ranking metrics over the 2-candidate pool (gold + reverse neg) ----
    # Same tie-breaking convention as evaluation_6/metrics_extra.random_pool_retrieval_metrics:
    # rank_i = #{distractors with sim STRICTLY > gold} + 1
    #   neg > pos       → rank = 2
    #   neg <= pos      → rank = 1  (ties count as rank-1, optimistic)
    t1 = time.time()
    ranks = 1 + (neg_score > pos_score).astype(np.int64)
    recall_at_k = {k: float((ranks <= k).mean()) for k in k_values}
    mrr = float((1.0 / ranks).mean())
    mean_rank = float(ranks.mean())
    median_rank = float(np.median(ranks))
    ret_time = time.time() - t1

    print(f"  AUC: {auc:.4f}    P@1 (vs 1 reverse neg): {precision_at_1:.4f}")
    print(f"  MRR: {mrr:.4f}    Mean rank: {mean_rank:.4f}    Median rank: {median_rank:.1f}")
    for k in k_values:
        print(f"  Recall@{k:<3}{recall_at_k[k]:.4f}    Hit@{k:<3}{recall_at_k[k]:.4f}")
    print(f"  ties: {ties:.4f}    "
          f"mean margin: {margin.mean():+.4f}    "
          f"mean pos sim: {pos_score.mean():.4f}    "
          f"mean neg sim: {neg_score.mean():.4f}")

    return {
        "model":                  model_key,
        "model_hf_id":            model_cfg["hf_id"],
        "dataset":                dataset_name,
        "n_queries":              len(pairs),
        "n_candidates_per_query": 2,   # gold + 1 reverse-pair negative
        "encoding_seconds":       enc_time,
        "ranking_seconds":        ret_time,
        "auc":                    auc,
        "precision_at_1":         precision_at_1,
        "n_negatives":            1,
        "mrr":                    mrr,
        "mean_rank":              mean_rank,
        "median_rank":            median_rank,
        "recall_at_k":            recall_at_k,
        "hit_at_k":               recall_at_k,  # one gold/query → coincide
        # Extras specific to the reverse-pair eval (not in parent results.json):
        "ties":                   ties,
        "mean_margin":            float(margin.mean()),
        "median_margin":          float(np.median(margin)),
        "mean_pos_sim":           float(pos_score.mean()),
        "mean_neg_sim":           float(neg_score.mean()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], k_values: list[int]) -> None:
    print("\n" + "=" * 130)
    print("SUMMARY  (1 negative per pair = reverse direction)")
    print("=" * 130)
    header = (
        f"{'model':<18} {'dataset':<14} {'N':>7} {'pool':>5} "
        f"{'AUC':>7} {'P@1':>7} {'MRR':>7} "
        f"{'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} "
        f"{'mean':>7} {'median':>7} {'ties':>7}"
    )
    print(header)
    print("-" * 130)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<18} {r['dataset']:<14}  ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:<18} {r['dataset']:<14} {r['n_queries']:>7d} "
            f"{r['n_candidates_per_query']:>5d} "
            f"{r['auc']:>7.4f} {r['precision_at_1']:>7.4f} {r['mrr']:>7.4f} "
            f"{r['recall_at_k'][1]:>7.4f} {r['recall_at_k'][3]:>7.4f} "
            f"{r['recall_at_k'][5]:>7.4f} {r['recall_at_k'][10]:>7.4f} "
            f"{r['mean_rank']:>7.4f} {r['median_rank']:>7.1f} {r['ties']:>7.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Cap pairs per dataset (for quick smoke runs).")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                    help="K values for Recall@K / Hit@K. Mirrors evaluation_6/.")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = ap.parse_args()
    k_values = sorted(set(args.k))

    print(f"Models:        {args.models}")
    print(f"Datasets:      {args.datasets}")
    print(f"Max pairs:     {args.max_pairs}")
    print(f"K values:      {k_values}")
    print(f"Output:        {args.out}")

    results: list[dict] = []
    for model_key in args.models:
        model_cfg = MODELS[model_key]
        print(f"\n[load] {model_key} ({model_cfg['hf_id']}, {model_cfg['pooling']} pool)")
        encoder = PrefixedEncoder(
            hf_id=model_cfg["hf_id"],
            pooling=model_cfg["pooling"],
            max_seq_length=model_cfg["max_seq_length"],
            batch_size=args.batch_size,
        )
        for ds_name in args.datasets:
            try:
                row = evaluate_on_dataset(
                    model_key=model_key,
                    model_cfg=model_cfg,
                    dataset_name=ds_name,
                    dataset_path=DATASETS[ds_name],
                    encoder=encoder,
                    max_pairs=args.max_pairs,
                    k_values=k_values,
                )
                results.append(row)
            except Exception as e:
                print(f"  ERROR ({type(e).__name__}): {e}")
                results.append({
                    "model": model_key,
                    "dataset": ds_name,
                    "error": f"{type(e).__name__}: {e}",
                })

    payload = {
        "config": {
            "models":     {k: MODELS[k] for k in args.models},
            "datasets":   {k: str(DATASETS[k]) for k in args.datasets},
            "batch_size": args.batch_size,
            "max_pairs":  args.max_pairs,
            "k_values":   k_values,
            "n_negatives_per_pair": 1,
            "negative_definition": "reverse pair (positive as query, anchor as passage)",
        },
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] saved to {out_path}")

    print_summary(results, k_values)


if __name__ == "__main__":
    main()
