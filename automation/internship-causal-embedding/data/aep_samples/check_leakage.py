"""
Check data leakage: for each text in test, check if it appears in train or val.
Checks both text_1 and text_2 from every row.
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent

def load_texts(split_file):
    """Load all (text_1, text_2) pairs from a JSONL file, return set of texts and list of rows."""
    texts = set()
    rows = []
    with open(split_file) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            t1 = row.get("text_1", "").strip()
            t2 = row.get("text_2", "").strip()
            texts.add(t1)
            texts.add(t2)
            rows.append((i, t1, t2, row))
    return texts, rows


def main():
    train_texts, train_rows = load_texts(DATA_DIR / "directional_train.jsonl")
    val_texts, val_rows = load_texts(DATA_DIR / "directional_val.jsonl")
    _, test_rows = load_texts(DATA_DIR / "directional_test.jsonl")

    train_and_val = train_texts | val_texts

    print(f"Train unique texts : {len(train_texts)}")
    print(f"Val unique texts   : {len(val_texts)}")
    print(f"Train∪Val texts    : {len(train_and_val)}")
    print(f"Test rows          : {len(test_rows)}")
    print()

    leaked_rows = []          # test rows where either text leaks
    leaked_text1_only = []
    leaked_text2_only = []
    leaked_both = []

    # per-source breakdown
    source_counts = defaultdict(int)  # "train" | "val" | "both"

    for idx, t1, t2, row in test_rows:
        t1_in_train = t1 in train_texts
        t1_in_val   = t1 in val_texts
        t2_in_train = t2 in train_texts
        t2_in_val   = t2 in val_texts

        t1_leaked = t1_in_train or t1_in_val
        t2_leaked = t2_in_train or t2_in_val

        if not (t1_leaked or t2_leaked):
            continue

        entry = {
            "test_row_idx": idx,
            "text_1_leaked": t1_leaked,
            "text_1_in_train": t1_in_train,
            "text_1_in_val": t1_in_val,
            "text_2_leaked": t2_leaked,
            "text_2_in_train": t2_in_train,
            "text_2_in_val": t2_in_val,
            "row": row,
        }
        leaked_rows.append(entry)

        if t1_leaked and t2_leaked:
            leaked_both.append(entry)
        elif t1_leaked:
            leaked_text1_only.append(entry)
        else:
            leaked_text2_only.append(entry)

        # source attribution
        sources = set()
        if t1_in_train or t2_in_train:
            sources.add("train")
        if t1_in_val or t2_in_val:
            sources.add("val")
        source_counts["+".join(sorted(sources))] += 1

    # --- Summary ---
    print("=" * 60)
    print("LEAKAGE SUMMARY")
    print("=" * 60)
    print(f"Test rows with ANY leakage    : {len(leaked_rows)} / {len(test_rows)}"
          f"  ({100*len(leaked_rows)/len(test_rows):.1f}%)")
    print(f"  Both text_1 and text_2 leak : {len(leaked_both)}")
    print(f"  Only text_1 leaks           : {len(leaked_text1_only)}")
    print(f"  Only text_2 leaks           : {len(leaked_text2_only)}")
    print()
    print("Source of leakage (per test row):")
    for src, cnt in sorted(source_counts.items()):
        print(f"  from {src:10s} : {cnt}")
    print()

    # Count unique leaked texts
    leaked_t1s = {e["row"]["text_1"] for e in leaked_rows if e["text_1_leaked"]}
    leaked_t2s = {e["row"]["text_2"] for e in leaked_rows if e["text_2_leaked"]}
    all_leaked_texts = leaked_t1s | leaked_t2s
    print(f"Unique leaked text_1 strings  : {len(leaked_t1s)}")
    print(f"Unique leaked text_2 strings  : {len(leaked_t2s)}")
    print(f"Total unique leaked texts     : {len(all_leaked_texts)}")
    print()

    # --- Unique leaked texts in full ---
    # Build a map: text -> {sources, affected_rows, slot(text_1/text_2)}
    unique_leaked = {}  # text -> dict
    for e in leaked_rows:
        for slot in ("text_1", "text_2"):
            if not e[f"{slot}_leaked"]:
                continue
            txt = e["row"][slot]
            if txt not in unique_leaked:
                src = []
                if e[f"{slot}_in_train"]: src.append("TRAIN")
                if e[f"{slot}_in_val"]:   src.append("VAL")
                unique_leaked[txt] = {"sources": src, "rows": [], "slots": set()}
            unique_leaked[txt]["rows"].append(e["test_row_idx"])
            unique_leaked[txt]["slots"].add(slot)

    print("=" * 60)
    print(f"UNIQUE LEAKED TEXTS ({len(unique_leaked)} total)")
    print("=" * 60)
    for n, (txt, info) in enumerate(unique_leaked.items(), 1):
        print(f"\n[Leaked text #{n}]")
        print(f"  Found in       : {', '.join(info['sources'])}")
        print(f"  Appears as     : {', '.join(sorted(info['slots']))}")
        print(f"  Affects rows   : {len(info['rows'])} test rows  (indices: {info['rows'][:10]}{'...' if len(info['rows'])>10 else ''})")
        print(f"  Full text:")
        print(f"  {'-'*56}")
        # wrap at 80 chars for readability
        for line in txt.split('\n'):
            while len(line) > 80:
                print(f"    {line[:80]}")
                line = line[80:]
            print(f"    {line}")
        print(f"  {'-'*56}")

    if not leaked_rows:
        print("No leakage detected. Test set is clean.")

    # Save full details to JSON
    out_file = DATA_DIR / "leakage_report.json"
    report = {
        "summary": {
            "train_unique_texts": len(train_texts),
            "val_unique_texts": len(val_texts),
            "train_val_union": len(train_and_val),
            "test_total_rows": len(test_rows),
            "test_rows_with_leakage": len(leaked_rows),
            "leaked_both_texts": len(leaked_both),
            "leaked_text1_only": len(leaked_text1_only),
            "leaked_text2_only": len(leaked_text2_only),
            "unique_leaked_text1": len(leaked_t1s),
            "unique_leaked_text2": len(leaked_t2s),
            "total_unique_leaked_texts": len(all_leaked_texts),
            "source_counts": dict(source_counts),
        },
        "leaked_rows": [
            {
                "test_row_idx": e["test_row_idx"],
                "text_1_leaked": e["text_1_leaked"],
                "text_1_in_train": e["text_1_in_train"],
                "text_1_in_val": e["text_1_in_val"],
                "text_2_leaked": e["text_2_leaked"],
                "text_2_in_train": e["text_2_in_train"],
                "text_2_in_val": e["text_2_in_val"],
                "label": e["row"].get("label"),
                "url_1": e["row"].get("url_1"),
                "url_2": e["row"].get("url_2"),
            }
            for e in leaked_rows
        ]
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {out_file}")


if __name__ == "__main__":
    main()
