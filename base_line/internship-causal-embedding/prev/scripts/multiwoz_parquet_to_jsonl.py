"""Convert multiwoz_v24 parquet files to PairExample-style JSONL.

Each row becomes one JSON line:
  {"anchor", "positive", "dataset", "context", "metadata"}

Anchor = history turns + current user utterance, joined with newlines
  (history is folded INTO the anchor, not kept in context).
Positive = system_response (the next part).
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data_6" / "multiwoz_v24"
DATASET = "multiwoz_v24"
SPLIT_OUT = {
    "train-00000-of-00001-d27f1cce8ef2d445.parquet": "train.jsonl",
    "validation-00000-of-00001-e2d540d091777ea4.parquet": "val.jsonl",
    "test-00000-of-00001-98ae416deade52a1.parquet": "test.jsonl",
}


def build_anchor(history, user: str) -> str:
    turns = []
    if history is not None and len(history) > 0:
        turns.extend(str(h).strip() for h in history if str(h).strip())
    user = (user or "").strip()
    if user:
        turns.append(user)
    return "\n".join(turns)


def convert(parquet_path: Path, out_path: Path) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    written = skipped = 0
    with out_path.open("w") as f:
        for _, row in df.iterrows():
            user = (row.get("user") or "").strip()
            positive = (row.get("system_response") or "").strip()
            if not user or not positive:
                skipped += 1
                continue
            anchor = build_anchor(row.get("history"), user)
            ex = {
                "anchor": anchor,
                "positive": positive,
                "dataset": DATASET,
                "context": [],
                "metadata": {
                    "dialogue_id": row.get("dialogue_id"),
                    "turn_id": int(row["turn_id"]) if "turn_id" in row else None,
                    "degenerate_user": bool(row.get("degenerate_user", False)),
                },
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def main() -> None:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No .parquet files under {DATA_DIR}")
    for p in parquet_files:
        out = DATA_DIR / SPLIT_OUT.get(p.name, p.stem + ".jsonl")
        written, skipped = convert(p, out)
        in_mb = p.stat().st_size / 1e6
        out_mb = out.stat().st_size / 1e6
        print(
            f"{p.name}  ({in_mb:.1f} MB)  ->  {out.name}  "
            f"({out_mb:.1f} MB, {written} pairs, {skipped} skipped)"
        )


if __name__ == "__main__":
    main()
