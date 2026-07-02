"""metrics.py -- ordering / partial-order evaluation metrics.

Implements the "evaluate as a partial order, not an exact chain" idea (Idea 4 /
research 07 sec.4-5). A pairwise-then-aggregate ordering pipeline should not be
judged only on exact-permutation match (Perfect Match Rate): real workflows have
parallel steps and a different-but-valid linear extension should not be punished.

Metrics provided
----------------
- ``exact_match(pred, gold)``         : 1 if the permutations are identical, else 0.
- ``perfect_match_rate(preds, golds)``: fraction of workflows ordered exactly right (PMR).
- ``kendall_tau_b(pred, gold)``       : tie-aware Kendall tau-b (via scipy.stats.kendalltau).
- ``spearman(pred, gold)``            : Spearman rank correlation (scipy.stats.spearmanr).
- ``precedence_prf(pred, gold)``      : precision / recall / F1 over precedence pairs
                                        {(i,j): i precedes j}. Credits any correct
                                        precedence regardless of which linear extension
                                        was chosen.
- ``evaluate_orders(preds, golds)``   : aggregate dict of mean metrics over a dataset.

All inputs are orderings expressed as lists of item labels, e.g. ``["t1","t3","t2"]``.
``gold`` may also be a list-of-lists (a *partial order* given as ordered "buckets" /
antichains); precedence_prf then only requires the cross-bucket precedences and treats
within-bucket pairs as ties (not penalised).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.stats import kendalltau, spearmanr

Order = Sequence[str]
PartialOrder = Union[Sequence[str], Sequence[Sequence[str]]]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rank_vector(order: Order) -> Dict[str, int]:
    """Map item -> position index for a flat total order."""
    return {item: i for i, item in enumerate(order)}


def _precedence_set(order: PartialOrder) -> set:
    """Set of ordered precedence pairs {(a, b): a strictly precedes b}.

    Accepts a flat total order (list of labels) or a partial order given as a list
    of buckets (list of lists); within-bucket pairs are NOT added (they are ties).
    """
    if len(order) > 0 and isinstance(order[0], (list, tuple)):
        buckets = [list(b) for b in order]
    else:
        buckets = [[x] for x in order]  # total order: each item its own bucket
    flat_before: List[str] = []
    prec = set()
    for bucket in buckets:
        for earlier in flat_before:
            for item in bucket:
                prec.add((earlier, item))
        flat_before.extend(bucket)
    return prec


def _flatten(order: PartialOrder) -> List[str]:
    if len(order) > 0 and isinstance(order[0], (list, tuple)):
        return [x for b in order for x in b]
    return list(order)


# ---------------------------------------------------------------------------
# exact match / PMR
# ---------------------------------------------------------------------------
def exact_match(pred: Order, gold: Order) -> int:
    """1 if predicted permutation exactly equals gold permutation."""
    return int(list(pred) == list(gold))


def perfect_match_rate(preds: Sequence[Order], golds: Sequence[Order]) -> float:
    """Fraction of workflows whose entire order is exactly correct (PMR)."""
    if not preds:
        return 0.0
    return float(np.mean([exact_match(p, g) for p, g in zip(preds, golds)]))


# ---------------------------------------------------------------------------
# rank correlations (tie aware)
# ---------------------------------------------------------------------------
def _aligned_rank_arrays(pred: Order, gold: Order) -> Tuple[np.ndarray, np.ndarray]:
    gold_flat = _flatten(gold)
    pred_rank = _rank_vector(_flatten(pred))
    gold_rank = _rank_vector(gold_flat)
    items = [x for x in gold_flat if x in pred_rank]
    g = np.array([gold_rank[x] for x in items], dtype=float)
    p = np.array([pred_rank[x] for x in items], dtype=float)
    return p, g


def kendall_tau_b(pred: Order, gold: Order) -> float:
    """Tie-aware Kendall tau-b. Returns nan for degenerate (<2 item) cases."""
    p, g = _aligned_rank_arrays(pred, gold)
    if len(p) < 2:
        return float("nan")
    tau, _ = kendalltau(p, g)  # scipy default variant is tau-b
    return float(tau)


def spearman(pred: Order, gold: Order) -> float:
    """Spearman rank correlation between predicted and gold positions."""
    p, g = _aligned_rank_arrays(pred, gold)
    if len(p) < 2:
        return float("nan")
    rho, _ = spearmanr(p, g)
    return float(rho)


# ---------------------------------------------------------------------------
# precedence-pair precision / recall / F1
# ---------------------------------------------------------------------------
def precedence_prf(pred: PartialOrder, gold: PartialOrder) -> Dict[str, float]:
    """Precision / recall / F1 over precedence pairs.

    A precedence pair (a, b) means "a strictly precedes b". Predicted precedences are
    compared against gold precedences as a set-classification problem. Within-bucket
    (tied / parallel) gold pairs are excluded, so a correct ordering of genuinely
    parallel steps is neither rewarded nor punished.
    """
    pred_p = _precedence_set(pred)
    gold_p = _precedence_set(gold)
    if not pred_p and not gold_p:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_p & gold_p)
    precision = tp / len(pred_p) if pred_p else 0.0
    recall = tp / len(gold_p) if gold_p else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# dataset-level aggregation
# ---------------------------------------------------------------------------
def evaluate_orders(preds: Sequence[Order], golds: Sequence[PartialOrder]) -> Dict[str, float]:
    """Aggregate metrics over a dataset of (pred, gold) order pairs.

    Returns mean exact-match (== PMR), mean Kendall tau-b, mean Spearman, and mean
    precedence precision/recall/F1.
    """
    em, taus, rhos, precs, recs, f1s = [], [], [], [], [], []
    for pred, gold in zip(preds, golds):
        em.append(exact_match(_flatten(pred), _flatten(gold)))
        t = kendall_tau_b(pred, gold)
        r = spearman(pred, gold)
        if not np.isnan(t):
            taus.append(t)
        if not np.isnan(r):
            rhos.append(r)
        prf = precedence_prf(pred, gold)
        precs.append(prf["precision"])
        recs.append(prf["recall"])
        f1s.append(prf["f1"])
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "n": len(preds),
        "perfect_match_rate": mean(em),
        "kendall_tau_b": mean(taus),
        "spearman": mean(rhos),
        "precedence_precision": mean(precs),
        "precedence_recall": mean(recs),
        "precedence_f1": mean(f1s),
    }


if __name__ == "__main__":
    print("metrics.py demo")
    gold = ["t1", "t2", "t3", "t4"]
    perfect = ["t1", "t2", "t3", "t4"]
    swapped = ["t1", "t3", "t2", "t4"]   # one adjacent swap
    reversed_ = ["t4", "t3", "t2", "t1"]

    for name, pred in [("perfect", perfect), ("one-swap", swapped), ("reversed", reversed_)]:
        prf = precedence_prf(pred, gold)
        print(f"\n[{name}] pred={pred}")
        print(f"  exact_match     = {exact_match(pred, gold)}")
        print(f"  kendall_tau_b   = {kendall_tau_b(pred, gold):+.3f}")
        print(f"  spearman        = {spearman(pred, gold):+.3f}")
        print(f"  precedence P/R/F1 = {prf['precision']:.3f} / {prf['recall']:.3f} / {prf['f1']:.3f}")

    # partial-order gold: t2 and t3 are parallel (same bucket)
    print("\n[partial-order gold] gold buckets = [[t1],[t2,t3],[t4]]")
    pgold = [["t1"], ["t2", "t3"], ["t4"]]
    print("  pred t1,t3,t2,t4 precedence F1 =",
          round(precedence_prf(["t1", "t3", "t2", "t4"], pgold)["f1"], 3),
          "(parallel swap not penalised)")

    print("\n[dataset]", evaluate_orders([perfect, swapped], [gold, gold]))
