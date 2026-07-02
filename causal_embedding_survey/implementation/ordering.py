"""ordering.py -- robust rank aggregation: turning a noisy pairwise score matrix
into a global workflow order.

This is the core "0/30 fix" from research 07/08 (Idea 1). The current production
pipeline uses a naive tournament / topological sort, which *cannot survive a single
cycle* (A>B>C>A) in the predicted graph -- and noisy pairwise causal scores almost
always produce cycles -- so the global order collapses (the 0/30 failure mode).

We replace it with robust aggregators that minimise *weighted pairwise disagreement*
and degrade gracefully under cycles:

- ``topo_sort_baseline``    : reproduces the brittle failure mode (Copeland-tournament
                              ordering with greedy cycle handling) for comparison.
- ``mfas_order``            : weighted Minimum-Feedback-Arc-Set ordering (Kemeny /
                              minimum-violations). Edges weighted by log-odds of
                              P(i precedes j). Uses igraph.feedback_arc_set
                              (Eades-Lin-Smyth / exact IP); pure-numpy Eades fallback.
- ``bradley_terry_order``   : fit Bradley-Terry / Plackett-Luce latent strengths with
                              choix (MM / I-LSR), sort by strength. Numpy MM fallback.
- ``kemeny_order``          : exact minimum-violations by permutation search for small
                              n (<=8), else delegates to mfas_order.

ALL functions take ``items`` (list of labels) and ``score_matrix``, an NxN numpy array
where ``score_matrix[i, j] = P(item_i causally precedes item_j)``. Diagonal is ignored.
"""

from __future__ import annotations

from itertools import permutations
from typing import List, Optional, Sequence

import numpy as np

try:
    import igraph  # type: ignore
    _HAS_IGRAPH = True
except Exception:  # pragma: no cover
    _HAS_IGRAPH = False

try:
    import choix  # type: ignore
    _HAS_CHOIX = True
except Exception:  # pragma: no cover
    _HAS_CHOIX = False

_EPS = 1e-6


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _logodds(p: np.ndarray) -> np.ndarray:
    """Elementwise log-odds log(p/(1-p)), clipped for numerical stability."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _validate(items: Sequence[str], score_matrix: np.ndarray) -> np.ndarray:
    n = len(items)
    P = np.asarray(score_matrix, dtype=float)
    if P.shape != (n, n):
        raise ValueError(f"score_matrix shape {P.shape} != ({n},{n})")
    return P


def count_backward_weight(order: Sequence[int], P: np.ndarray) -> float:
    """Total log-odds weight of pairwise judgments VIOLATED by ``order`` (lower=better).

    For every ordered pair (a placed before b in ``order``), the judgment "b precedes a"
    is violated; we accumulate its positive log-odds weight. This is the weighted-FAS /
    Kemeny objective the robust orderers minimise.
    """
    W = _logodds(P)
    pos = {item: r for r, item in enumerate(order)}
    total = 0.0
    n = P.shape[0]
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if pos[a] < pos[b] and W[b, a] > 0:  # we say a<b but evidence says b<a
                total += W[b, a]
    return float(total)


# ---------------------------------------------------------------------------
# 1. baseline: naive tournament / topological sort (the failure mode)
# ---------------------------------------------------------------------------
def topo_sort_baseline(items: Sequence[str], score_matrix: np.ndarray) -> List[str]:
    """Naive tournament + topological sort -- reproduces the brittle 0/30 failure mode.

    Build a hard directed graph: edge i->j whenever P[i,j] > 0.5 (i thought to precede j).
    Attempt a topological sort (Kahn's algorithm). On a true DAG this gives the right
    order. On ANY cycle Kahn's algorithm stalls; we then fall back to the typical
    production hack -- emit remaining nodes by descending raw out-degree (net wins),
    exactly the kind of arbitrary tie-break that scrambles the order under noise.
    """
    P = _validate(items, score_matrix)
    n = len(items)
    adj = [[False] * n for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and P[i, j] > 0.5:
                adj[i][j] = True
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                indeg[j] += 1

    order: List[int] = []
    remaining = set(range(n))
    # Kahn's algorithm
    queue = sorted([i for i in remaining if indeg[i] == 0])
    while queue:
        node = queue.pop(0)
        remaining.discard(node)
        order.append(node)
        for j in range(n):
            if adj[node][j]:
                indeg[j] -= 1
                if indeg[j] == 0 and j in remaining:
                    queue.append(j)
        queue.sort()

    if remaining:  # CYCLE DETECTED -> brittle fallback: append stuck nodes in input order
        # This is the genuine failure mode (research 07 sec.2.4): a single back-edge in a
        # cycle stalls Kahn's algorithm and the remaining nodes get an essentially arbitrary
        # order. A real topo-sort cannot produce a valid order on a cyclic graph at all.
        for node in range(n):
            if node in remaining:
                order.append(node)

    return [items[i] for i in order]


# ---------------------------------------------------------------------------
# 2. weighted Minimum-Feedback-Arc-Set (Kemeny / minimum-violations)
# ---------------------------------------------------------------------------
def _eades_numpy(P: np.ndarray) -> List[int]:
    """Pure-numpy Eades-Lin-Smyth greedy heuristic on the net log-odds tournament.

    Repeatedly peel sinks to the tail and sources to the head; otherwise remove the
    vertex maximising (out-weight - in-weight). Returns a vertex ordering (head..tail).
    """
    W = _logodds(P)
    net = np.maximum(W, 0.0)  # directed positive-evidence weights
    n = P.shape[0]
    remaining = list(range(n))
    head: List[int] = []
    tail: List[int] = []

    def outw(v, rem):
        return sum(net[v, u] for u in rem if u != v)

    def inw(v, rem):
        return sum(net[u, v] for u in rem if u != v)

    while remaining:
        changed = True
        while changed and remaining:
            changed = False
            # sinks (no outgoing evidence) -> tail
            for v in list(remaining):
                if outw(v, remaining) <= _EPS:
                    tail.insert(0, v)
                    remaining.remove(v)
                    changed = True
            # sources (no incoming evidence) -> head
            for v in list(remaining):
                if inw(v, remaining) <= _EPS:
                    head.append(v)
                    remaining.remove(v)
                    changed = True
        if remaining:
            # pick vertex with max (out - in); place at head
            v = max(remaining, key=lambda x: outw(x, remaining) - inw(x, remaining))
            head.append(v)
            remaining.remove(v)
    return head + tail


def _local_search(order: List[int], P: np.ndarray, max_iter: int = 1000) -> List[int]:
    """Adjacent-swap + single-item-reinsertion local search to reduce backward weight."""
    best = list(order)
    best_cost = count_backward_weight(best, P)
    improved = True
    it = 0
    n = len(order)
    while improved and it < max_iter:
        improved = False
        it += 1
        # adjacent swaps
        for i in range(n - 1):
            cand = best[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = count_backward_weight(cand, P)
            if c < best_cost - _EPS:
                best, best_cost, improved = cand, c, True
        # single-item reinsertion
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = best[:]
                x = cand.pop(i)
                cand.insert(j, x)
                c = count_backward_weight(cand, P)
                if c < best_cost - _EPS:
                    best, best_cost, improved = cand, c, True
    return best


def mfas_order(items: Sequence[str], score_matrix: np.ndarray) -> List[str]:
    """Weighted Minimum-Feedback-Arc-Set ordering (Kemeny / minimum-violations).

    Edges are weighted by the log-odds of the pairwise precedence probability. We solve
    the minimum-weight FAS with igraph.feedback_arc_set (Eades-Lin-Smyth heuristic, or
    exact integer program for tiny graphs); falling back to a pure-numpy Eades heuristic
    if igraph is unavailable. A cheap local search then polishes the result. The returned
    order disagrees with the fewest confidence-weighted pairwise judgments and reproduces
    the topological order exactly on a true DAG.
    """
    P = _validate(items, score_matrix)
    n = len(items)
    if n <= 1:
        return list(items)

    W = _logodds(P)
    if _HAS_IGRAPH:
        # Build a weighted directed graph keeping only the *net* dominant direction
        # of each pair (a proper tournament), weight = net positive log-odds.
        edges, weights = [], []
        for i in range(n):
            for j in range(i + 1, n):
                net = W[i, j] - W[j, i]
                if net > 0:
                    edges.append((i, j)); weights.append(net)
                elif net < 0:
                    edges.append((j, i)); weights.append(-net)
                else:
                    edges.append((i, j)); weights.append(_EPS)
        g = igraph.Graph(n=n, edges=edges, directed=True)
        algo = "exact_ip" if n <= 12 else "approx_eades"
        try:
            fas = set(g.feedback_arc_set(weights=weights, method=algo))
        except Exception:
            fas = set(g.feedback_arc_set(weights=weights, method="approx_eades"))
        # remove FAS edges -> DAG -> topological order
        kept = [e for idx, e in enumerate(edges) if idx not in fas]
        dag = igraph.Graph(n=n, edges=kept, directed=True)
        try:
            order = dag.topological_sorting(mode="out")
        except Exception:
            order = _eades_numpy(P)
    else:
        order = _eades_numpy(P)

    order = _local_search(order, P)
    return [items[i] for i in order]


# ---------------------------------------------------------------------------
# 3. Bradley-Terry / Plackett-Luce strengths
# ---------------------------------------------------------------------------
def _bt_mm_numpy(P: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """Bradley-Terry strengths via Zermelo/Ford minorisation-maximisation (MM).

    Uses P[i,j] (prob i precedes j) as a soft win-count of i over j. Returns log-strengths.
    """
    n = P.shape[0]
    # soft win matrix: w[i,j] = "fractional times i beat j"
    w = P.copy()
    np.fill_diagonal(w, 0.0)
    pi = np.ones(n)
    wins = w.sum(axis=1)  # total wins of each item
    for _ in range(n_iter):
        new = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                games = w[i, j] + w[j, i]
                if games > 0:
                    denom += games / (pi[i] + pi[j])
            new[i] = wins[i] / denom if denom > 0 else pi[i]
        new = np.clip(new, _EPS, None)
        new = new / new.sum() * n  # normalise scale
        if np.allclose(new, pi, atol=1e-9):
            pi = new
            break
        pi = new
    return np.log(np.clip(pi, _EPS, None))


def bradley_terry_order(items: Sequence[str], score_matrix: np.ndarray) -> List[str]:
    """Fit Bradley-Terry / Plackett-Luce latent strengths and sort by strength (desc).

    Strength beta_i is the latent "earliness"; higher strength => earlier in the order.
    Uses choix.ilsr_pairwise (I-LSR, converges to the MLE) when choix is available,
    feeding soft pairwise win-counts derived from the precedence probabilities; falls
    back to a pure-numpy MM (Zermelo) fit otherwise. BT absorbs cycles into the best
    transitive fit, so it never fails outright the way topo-sort does.
    """
    P = _validate(items, score_matrix)
    n = len(items)
    if n <= 1:
        return list(items)

    if _HAS_CHOIX:
        # choix wants integer-ish comparison data: (winner, loser). We expand each pair
        # into weighted comparisons. To stay light, replicate by rounding prob*scale.
        scale = 20
        data = []
        for i in range(n):
            for j in range(i + 1, n):
                wij = int(round(P[i, j] * scale))
                wji = int(round(P[j, i] * scale))
                data += [(i, j)] * wij   # i beats j  (i precedes j)
                data += [(j, i)] * wji
        try:
            params = choix.ilsr_pairwise(n, data, alpha=0.01)
        except Exception:
            params = _bt_mm_numpy(P)
    else:
        params = _bt_mm_numpy(P)

    # higher strength => precedes => earlier
    order = sorted(range(n), key=lambda i: -params[i])
    return [items[i] for i in order]


# ---------------------------------------------------------------------------
# 4. exact Kemeny (minimum-violations) for small n
# ---------------------------------------------------------------------------
def kemeny_order(items: Sequence[str], score_matrix: np.ndarray, max_exact: int = 8) -> List[str]:
    """Exact minimum-violations (Kemeny) ordering by permutation search for small n.

    For n <= ``max_exact`` enumerate all permutations and return the one minimising the
    log-odds-weighted backward-arc weight (the exact Kemeny optimum). For larger n, fall
    back to the MFAS heuristic + local search.
    """
    P = _validate(items, score_matrix)
    n = len(items)
    if n <= 1:
        return list(items)
    if n > max_exact:
        return mfas_order(items, score_matrix)

    best_perm, best_cost = None, float("inf")
    for perm in permutations(range(n)):
        c = count_backward_weight(perm, P)
        if c < best_cost:
            best_cost, best_perm = c, perm
    return [items[i] for i in best_perm]


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def _demo_matrix_with_cycle():
    """A 3-item tournament with a noisy cycle: t1<t2, t2<t3, but a wrong t3<t1 edge.

    Gold order is t1, t2, t3. The spurious high-ish t3->t1 edge creates a cycle that
    breaks naive topo-sort but should be tolerated by the robust aggregators.
    """
    items = ["t1", "t2", "t3"]
    P = np.array([
        [0.5, 0.92, 0.55],   # t1 precedes t2 (strong), t1 vs t3 weak/right
        [0.08, 0.5, 0.90],   # t2 precedes t3 (strong)
        [0.45, 0.10, 0.5],   # t3 vs t1 noisy (says t1 precedes); cycle pressure
    ])
    return items, P, ["t1", "t2", "t3"]


if __name__ == "__main__":
    print("ordering.py demo")
    print(f"igraph available: {_HAS_IGRAPH}   choix available: {_HAS_CHOIX}\n")

    items, P, gold = _demo_matrix_with_cycle()
    print("Pairwise P(i precedes j):")
    print(P, "\n")
    print(f"gold order            : {gold}")
    print(f"topo_sort_baseline    : {topo_sort_baseline(items, P)}")
    print(f"mfas_order            : {mfas_order(items, P)}")
    print(f"bradley_terry_order   : {bradley_terry_order(items, P)}")
    print(f"kemeny_order (exact)  : {kemeny_order(items, P)}")

    # an explicitly cyclic tournament that DEFEATS topo-sort
    print("\n--- hard cyclic tournament (A>B>C>A all strong) ---")
    it2 = ["A", "B", "C"]
    Pc = np.array([
        [0.5, 0.9, 0.1],
        [0.1, 0.5, 0.9],
        [0.9, 0.1, 0.5],
    ])
    print(f"topo_sort_baseline    : {topo_sort_baseline(it2, Pc)}")
    print(f"mfas_order            : {mfas_order(it2, Pc)}  (graceful: 1 violated edge)")
    print(f"kemeny_order          : {kemeny_order(it2, Pc)}")
