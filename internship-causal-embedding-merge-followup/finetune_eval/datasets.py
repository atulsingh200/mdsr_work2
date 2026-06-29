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
    # Directional AEP causal classification. Train and val are the all-label-1
    # converted files (every row a forward anchor->positive pair) used by the
    # reverse-direction DPO trainer + its in-training MRR validation. Test keeps
    # the raw directional file with text_1/text_2 + 0/1 labels for thresholded
    # accuracy evaluation (see evaluate_finetune_reverse_cls.py); the raw val
    # for threshold tuning is configured in RAW_LABELED_VAL below.
    "aep_causal_cls": {
        "train": "directional_train_all_label1.jsonl",
        "val":   "directional_val_all_label1.jsonl",
        "test":  "directional_test.jsonl",
    },
}


# For threshold-based classification eval: the raw labeled val file (with the
# original text_1/text_2 + 0/1 labels) used to tune the decision threshold.
# Resolved through the same DATA_SUBDIR mapping as split_path.
RAW_LABELED_VAL: dict[str, str] = {
    "aep_causal_cls": "directional_val.jsonl",
}


# Datasets whose files live in a directory that differs from the logical name.
DATA_SUBDIR: dict[str, str] = {
    "aep_causal_cls": "aep_causal_classification",
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
    "aep_causal_cls": None,    # 110,775 — use all (train on the entire dataset)
}


def split_path(dataset: str, split: str) -> Path | None:
    """Resolve the JSONL path for (dataset, split). Returns None if not configured."""
    cfg = SPLITS[dataset]
    fname = cfg[split]
    if fname is None:
        return None
    subdir = DATA_SUBDIR.get(dataset, dataset)
    return DATA_DIR / subdir / fname


def raw_labeled_val_path(dataset: str) -> Path | None:
    """Path to the raw labeled val file (text_1/text_2 + 0/1 label) used for
    threshold tuning in the classification eval. None if not configured."""
    fname = RAW_LABELED_VAL.get(dataset)
    if fname is None:
        return None
    subdir = DATA_SUBDIR.get(dataset, dataset)
    return DATA_DIR / subdir / fname


def list_datasets() -> list[str]:
    return list(SPLITS.keys())
