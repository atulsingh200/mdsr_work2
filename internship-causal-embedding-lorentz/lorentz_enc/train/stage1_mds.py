"""Stage 1: Build DAG from training pairs, run Lorentzian MDS, save coordinates."""

import torch
import yaml
import os
from ..data.graph import build_dag_from_pairs
from ..pretrain.lorentzian_mds import lorentzian_mds


def run_stage1(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    G, text_of = build_dag_from_pairs(cfg["paths"]["train_jsonl"])

    space_coords, time_coords, eigvals = lorentzian_mds(
        G,
        space_dim=cfg["heads"]["space_dim"],
        time_dim=cfg["heads"]["time_dim"],
        max_spacelike_distance=cfg["mds"].get("max_spacelike_distance"),
        jitter=cfg["mds"]["jitter"],
    )

    os.makedirs(os.path.dirname(cfg["paths"]["mds_coords"]), exist_ok=True)
    torch.save({
        "space": {k: torch.tensor(v, dtype=torch.float32) for k, v in space_coords.items()},
        "time": {k: torch.tensor(v, dtype=torch.float32) for k, v in time_coords.items()},
        "eigvals": torch.tensor(eigvals, dtype=torch.float32),
        "text_of": text_of,
    }, cfg["paths"]["mds_coords"])
    print(f"[Stage 1] saved MDS coordinates to {cfg['paths']['mds_coords']}")

    n_neg = (eigvals < 0).sum()
    n_pos = (eigvals > 0).sum()
    print(f"[Stage 1] eigenvalue spectrum: {n_neg} negative, {n_pos} positive")
    sorted_neg = sorted(eigvals[eigvals < 0])
    sorted_pos = sorted(eigvals[eigvals > 0], reverse=True)
    print(f"[Stage 1] top-5 negative: {sorted_neg[:5]}")
    print(f"[Stage 1] top-5 positive: {sorted_pos[:5]}")
