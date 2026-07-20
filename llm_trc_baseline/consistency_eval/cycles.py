"""Temporal-graph construction, cycle detection, metrics, and cycle-breaking (paper 2, §5).

A workflow's predicted graph is a tournament: for each unordered step pair {i,j} the CE predicts a
direction ("i before j" or "j before i") with a confidence. We build a directed graph of the
predicted BEFORE edges, detect simple cycles (inconsistencies), and break them one edge at a time.

Gold: canonical order is i<j (step i before step j). So an edge is CORRECT iff it points i->j (i<j).
Pairwise ordering accuracy == micro-F1 here (every gold pair's label is "before" in canonical
orientation), and equals the fraction of pairs the model ordered correctly.
"""

from __future__ import annotations

import networkx as nx


def build_digraph(edges: list[dict]) -> nx.DiGraph:
    """edges: [{i,j,p_ij,pred_before,confidence}] with i<j. Directed edge in predicted order."""
    G = nx.DiGraph()
    for e in edges:
        i, j = e["i"], e["j"]
        G.add_node(i)
        G.add_node(j)
        u, v = (i, j) if e["pred_before"] else (j, i)     # predicted "before" direction
        G.add_edge(u, v, confidence=e["confidence"], correct=(u < v))
    return G


def count_simple_cycles(G: nx.DiGraph, cap: int = 100000) -> int:
    n = 0
    for _ in nx.simple_cycles(G):
        n += 1
        if n >= cap:
            break
    return n


def ordering_accuracy(G: nx.DiGraph, total_pairs: int) -> float:
    """Fraction of ALL gold pairs whose (remaining) edge points the correct way (i<j)."""
    correct = sum(1 for u, v in G.edges() if u < v)
    return correct / total_pairs if total_pairs else 0.0


def _break(G: nx.DiGraph, chooser, max_iters: int = 100000):
    """Remove one edge per detected cycle (chosen by `chooser`) until the graph is acyclic.

    chooser(cycle_edges, G) -> (u, v) edge to remove.  cycle_edges is a list of (u,v).
    Returns (removed_edges, n_llm_or_iter_calls).
    """
    removed, calls = [], 0
    G = G.copy()
    for _ in range(max_iters):
        try:
            cyc = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        cycle_edges = [(u, v) for u, v, _ in cyc]
        calls += 1
        e = chooser(cycle_edges, G)
        if e is None or not G.has_edge(*e):
            e = min(cycle_edges, key=lambda uv: G[uv[0]][uv[1]]["confidence"])  # safe fallback
        G.remove_edge(*e)
        removed.append(e)
    return G, removed, calls


def break_confidence(G: nx.DiGraph):
    """Confidence-based (paper's best): drop the lowest-confidence edge in each cycle."""
    return _break(G, lambda edges, g: min(edges, key=lambda uv: g[uv[0]][uv[1]]["confidence"]))


def break_llm(G: nx.DiGraph, steps: list[str], llm_choose):
    """LLM-assisted: `llm_choose(cycle_edges, steps) -> (u,v)` picks the edge to drop."""
    return _break(G, lambda edges, g: llm_choose(edges, steps))


def workflow_metrics(workflows: list[dict]) -> dict:
    """Pre-break metrics over the whole set."""
    n_cyclic, total_cycles, correct, total = 0, 0, 0, 0
    per = []
    for wf in workflows:
        G = build_digraph(wf["edges"])
        tp = len(wf["edges"])
        nc = count_simple_cycles(G)
        acc = ordering_accuracy(G, tp)
        per.append({"id": wf["id"], "n_steps": wf["n_steps"], "n_pairs": tp,
                    "n_cycles": nc, "acc": acc, "has_cycle": nc > 0})
        n_cyclic += int(nc > 0)
        total_cycles += nc
        correct += sum(1 for e in wf["edges"] if e["pred_before"])
        total += tp
    n = len(workflows)
    return {"n_workflows": n, "cycle_rate": n_cyclic / n if n else 0.0,
            "n_cyclic_workflows": n_cyclic, "mean_cycles_per_wf": total_cycles / n if n else 0.0,
            "total_simple_cycles": total_cycles,
            "ordering_accuracy": correct / total if total else 0.0,
            "n_pairs": total, "per_workflow": per}
