"""Convert a directional_*.jsonl file so every row has label=1.

For label=0 rows: swap text_1/text_2, url_1/url_2, tier_1/tier_2, sub_1/sub_2
and set label=1.  Also renames text_1 -> anchor and text_2 -> positive so the
output is a forward (anchor, positive) pair file ready for the reverse-direction
finetune pipeline.

Usage:
  python make_all_label1.py                      # train -> train_all_label1
  python make_all_label1.py --in directional_val.jsonl --out directional_val_all_label1.jsonl
"""
import argparse
import json
import pathlib

DATA_DIR = pathlib.Path(__file__).parent

SWAP_PAIRS = [
    ("text_1", "text_2"),
    ("url_1", "url_2"),
    ("tier_1", "tier_2"),
    ("sub_1", "sub_2"),
]


def convert(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    n_total = n_swapped = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1
            if row.get("label") == 0:
                for a, b in SWAP_PAIRS:
                    if a in row and b in row:
                        row[a], row[b] = row[b], row[a]
                row["label"] = 1
                n_swapped += 1
            row["anchor"] = row.pop("text_1")
            row["positive"] = row.pop("text_2")
            fout.write(json.dumps(row) + "\n")
    print(f"Done. {n_total} rows total, {n_swapped} swapped -> {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_file", default="directional_train.jsonl")
    ap.add_argument("--out", dest="out_file", default="directional_train_all_label1.jsonl")
    args = ap.parse_args()
    in_path = (DATA_DIR / args.in_file) if not pathlib.Path(args.in_file).is_absolute() else pathlib.Path(args.in_file)
    out_path = (DATA_DIR / args.out_file) if not pathlib.Path(args.out_file).is_absolute() else pathlib.Path(args.out_file)
    convert(in_path, out_path)


if __name__ == "__main__":
    main()
