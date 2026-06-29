import numpy as np
import networkx as nx
from typing import Dict, Tuple
from .longest_path import all_pairs_longest_path
from .spacelike import naive_spacelike_distance


def lorentzian_mds(
    G: nx.DiGraph,
    space_dim: int = 1004,
    time_dim: int = 20,
    max_spacelike_distance=None,
    jitter: float = 1e-6,
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Generalised MDS for Lorentzian signature.
    Returns:
        space_coords: dict[node_id] -> np.ndarray(space_dim,)
        time_coords:  dict[node_id] -> np.ndarray(time_dim,)
        eigvals: full eigenvalue spectrum (sorted by magnitude descending)
    """
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    print(f"[MDS] computing longest paths for {N} nodes…")
    lp = all_pairs_longest_path(G)

    if max_spacelike_distance is None:
        max_d = max(lp.values()) if lp else 1
    else:
        max_d = max_spacelike_distance

    print(f"[MDS] computing naive spacelike distances…")
    sl = naive_spacelike_distance(G, lp, max_distance=max_d)

    print(f"[MDS] building separation matrix ({N}x{N})…")
    M = np.zeros((N, N), dtype=np.float64)
    for (u, v), d in lp.items():
        i, j = idx[u], idx[v]
        M[i, j] = -float(d) ** 2
        M[j, i] = -float(d) ** 2
    for (u, v), d in sl.items():
        i, j = idx[u], idx[v]
        M[i, j] = +float(d) ** 2

    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ M @ J
    B = 0.5 * (B + B.T) + jitter * np.eye(N)

    print(f"[MDS] eigendecomposition…")
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    neg_mask = eigvals < 0
    pos_mask = eigvals > 0
    neg_vals = eigvals[neg_mask]
    neg_vecs = eigvecs[:, neg_mask]
    pos_vals = eigvals[pos_mask]
    pos_vecs = eigvecs[:, pos_mask]

    n_time = min(time_dim, neg_vals.shape[0])
    time_block = neg_vecs[:, :n_time] * np.sqrt(np.abs(neg_vals[:n_time]))
    if n_time < time_dim:
        pad = np.zeros((N, time_dim - n_time))
        time_block = np.concatenate([time_block, pad], axis=1)

    n_space = min(space_dim, pos_vals.shape[0])
    space_block = pos_vecs[:, :n_space] * np.sqrt(pos_vals[:n_space])
    if n_space < space_dim:
        pad = np.zeros((N, space_dim - n_space))
        space_block = np.concatenate([space_block, pad], axis=1)

    norms = np.linalg.norm(space_block, axis=1, keepdims=True).clip(min=1e-9)
    space_block = space_block / norms

    space_coords = {n: space_block[idx[n]] for n in nodes}
    time_coords = {n: time_block[idx[n]] for n in nodes}
    return space_coords, time_coords, eigvals
