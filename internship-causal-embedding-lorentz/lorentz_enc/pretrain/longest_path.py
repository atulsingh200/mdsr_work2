import networkx as nx
import numpy as np
from typing import Dict, Tuple


def all_pairs_longest_path(G: nx.DiGraph) -> Dict[Tuple, int]:
    """
    Compute longest directed path length between every ordered timelike pair.
    Returns dict[(u, v)] -> length (edges). Only populated for timelike pairs (u ancestor of v).
    O(V * (V+E)) using per-node DP over topological order.
    """
    topo = list(nx.topological_sort(G))
    idx = {n: i for i, n in enumerate(topo)}
    N = len(topo)

    NEG = -10**9
    # Use sparse representation for large graphs: dist[i] = dict {j: longest path from i to j}
    # For V < 5000 use dense matrix; otherwise use sparse DP
    if N <= 4000:
        dp = np.full((N, N), NEG, dtype=np.int32)
        for i in range(N):
            dp[i, i] = 0
        for u_i, u in enumerate(topo):
            for v in G.successors(u):
                v_i = idx[v]
                if dp[u_i, v_i] < 1:
                    dp[u_i, v_i] = 1
        # Propagate: iterate in topo order
        for k_i in range(N):
            for i in range(N):
                if dp[i, k_i] == NEG:
                    continue
                for j in range(N):
                    if dp[k_i, j] == NEG:
                        continue
                    cand = dp[i, k_i] + dp[k_i, j]
                    if cand > dp[i, j]:
                        dp[i, j] = cand
        out = {}
        for i, u in enumerate(topo):
            for j, v in enumerate(topo):
                if i != j and dp[i, j] > NEG and dp[i, j] >= 1:
                    out[(u, v)] = int(dp[i, j])
        return out
    else:
        # Sparse DP: for each node in topo order, propagate max distances forward
        dist = {n: {} for n in topo}
        for u in topo:
            dist[u][u] = 0
            for v in G.successors(u):
                if v not in dist[u] or dist[u][v] < 1:
                    dist[u][v] = 1
            # Propagate u's distances through u's successors
            for v in G.successors(u):
                for w, d_vw in dist.get(v, {}).items():
                    new_d = 1 + d_vw
                    if w not in dist[u] or dist[u][w] < new_d:
                        dist[u][w] = new_d
        out = {}
        for u in topo:
            for v, d in dist[u].items():
                if u != v and d >= 1:
                    out[(u, v)] = d
        return out
