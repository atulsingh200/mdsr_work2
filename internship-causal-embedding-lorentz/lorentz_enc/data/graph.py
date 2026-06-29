import networkx as nx
from typing import Dict, List, Tuple
import json


def build_dag_from_pairs(jsonl_path: str):
    G = nx.DiGraph()
    text_of = {}

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            a_id = ex.get("anchor_id", ex["anchor"])
            p_id = ex.get("positive_id", ex["positive"])
            G.add_node(a_id)
            G.add_node(p_id)
            G.add_edge(a_id, p_id, weight=1.0)
            text_of[a_id] = ex["anchor"]
            text_of[p_id] = ex["positive"]

    # Break cycles if any
    if not nx.is_directed_acyclic_graph(G):
        while not nx.is_directed_acyclic_graph(G):
            try:
                cyc = nx.find_cycle(G, orientation="original")
                G.remove_edge(cyc[0][0], cyc[0][1])
            except nx.NetworkXNoCycle:
                break
    return G, text_of
