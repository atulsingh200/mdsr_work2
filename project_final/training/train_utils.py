"""Minimal training utilities used by train_encoder.py and train_encoder_reasoning.py.

Only the three functions actually imported by the CrossEncoder2x trainers are included:
  - device_auto()
  - set_seed()
  - make_logger()
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log
