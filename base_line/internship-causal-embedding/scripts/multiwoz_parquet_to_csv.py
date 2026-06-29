"""Convert multiwoz_v24 parquet files to CSV in place.

Nested columns (history, system_acts, belief_state, prev_belief_state,
belief_state_delta) are JSON-encoded so the CSV round-trips back to the
same structure with json.loads().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data_6" / "multiwoz_v24"
NESTED_COLS = (
    "history",
    "system_acts",
    "belief_state",
    "prev_belief_state",
    "belief_state_delta",
)


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def convert(parquet_path: Path) -> Path:
    df = pd.read_parquet(parquet_path)
    for col in NESTED_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda x: json.dumps(to_jsonable(x), ensure_ascii=False))
    csv_path = parquet_path.with_suffix(".csv.gz")
    df.to_csv(csv_path, index=False, compression="gzip")
    return csv_path


def main() -> None:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No .parquet files under {DATA_DIR}")
    for p in parquet_files:
        csv = convert(p)
        in_mb = p.stat().st_size / 1e6
        out_mb = csv.stat().st_size / 1e6
        print(f"{p.name}  ({in_mb:.1f} MB)  ->  {csv.name}  ({out_mb:.1f} MB)")


if __name__ == "__main__":
    main()
