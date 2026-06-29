import networkx as nx
from typing import Dict, Tuple


def naive_spacelike_distance(
    G: nx.DiGraph,
    longest_path: Dict[Tuple, int],
    max_distance: int,
) -> Dict[Tuple, int]:
    """
    Clough & Evans naive spacelike distance for pairs with no directed path either way.
    For each unordered spacelike pair (i, j): find the minimum longest-path between
    common past (l) and common future (k) — that defines the spacelike distance.
    """
    nodes = list(G.nodes())
    futures = {n: set(nx.descendants(G, n)) for n in nodes}
    pasts = {n: set(nx.ancestors(G, n)) for n in nodes}

    out = {}
    for ii, i in enumerate(nodes):
        for j in nodes[ii+1:]:
            if (i, j) in longest_path or (j, i) in longest_path:
                continue
            common_future = futures[i] & futures[j]
            common_past = pasts[i] & pasts[j]
            best = None
            for k in common_future:
                for l in common_past:
                    d = longest_path.get((l, k))
                    if d is None:
                        continue
                    if best is None or d < best:
                        best = d
            if best is None:
                best = max_distance
            out[(i, j)] = best
            out[(j, i)] = best
    return out
