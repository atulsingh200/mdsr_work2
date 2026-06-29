"""Per-dataset configuration for the finetune pipeline.

Each entry maps logical split names ("train"/"val"/"test") to the actual
JSONL file basename inside `data_6/<dataset>/`. Datasets without a val
split (qrecc) hold out a fraction of train as validation in train_finetune.py.
"""

from __future__ import annotations

from pathlib import Path

# Project root: finetune_eval/datasets.py → parent → parent
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_6"


SPLITS: dict[str, dict[str, str | None]] = {
    "aep_causal":   {"train": "train.jsonl",       "val": "val.jsonl",       "test": "test.jsonl"},
    "followupqg":   {"train": "train.jsonl",       "val": "valid.jsonl",     "test": "test.jsonl"},
    "multiwoz_v24": {"train": "train.jsonl",       "val": "val.jsonl",       "test": "test.jsonl"},
    "qrecc":        {"train": "train.jsonl",       "val": None,              "test": "test.jsonl"},
    "workflow":     {"train": "train_pairs.jsonl", "val": "dev_pairs.jsonl", "test": "test_pairs.jsonl"},
}


# Per-dataset training caps to keep total wall-clock tractable on one A100.
# Hard-negative mining is O(N²/chunk) and BERT fine-tuning is O(N · epochs); for the
# very large datasets we sample a fixed subset rather than train on all rows.
TRAIN_CAPS: dict[str, int | None] = {
    "aep_causal":   None,      # 5,487 — use all
    "followupqg":   None,      # 2,790 — use all
    "multiwoz_v24": 15_000,    # 56,719 → cap (for runtime budget on A100)
    "qrecc":        15_000,    # 52,481 → cap
    "workflow":     20_000,    # 2,116,157 → cap
}


def split_path(dataset: str, split: str) -> Path | None:
    """Resolve the JSONL path for (dataset, split). Returns None if not configured."""
    cfg = SPLITS[dataset]
    fname = cfg[split]
    if fname is None:
        return None
    return DATA_DIR / dataset / fname


def list_datasets() -> list[str]:
    return list(SPLITS.keys())
