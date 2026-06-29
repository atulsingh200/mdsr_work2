"""
Check for data leakage: any text_1 or text_2 from the test set
that also appears in train or val.
"""

import json
from pathlib import Path

DATA_DIR = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification")


def iter_json_lines(filepath):
    """Yield parsed JSON objects, handling embedded newlines inside strings."""
    with open(filepath) as f:
        buf = ""
        line_num = 0
        for raw in f:
            buf += raw
            try:
                obj = json.loads(buf)
                line_num += 1
                yield line_num, obj
                buf = ""
            except json.JSONDecodeError:
                # incomplete record — keep accumulating
                continue


def load_texts(filepath):
    texts = set()
    for _, row in iter_json_lines(filepath):
        texts.add(row["text_1"])
        texts.add(row["text_2"])
    return texts


def load_rows(filepath):
    rows = []
    for line_num, row in iter_json_lines(filepath):
        rows.append((line_num, row))
    return rows


print("Loading splits...")
train_texts = load_texts(DATA_DIR / "directional_train.jsonl")
val_texts   = load_texts(DATA_DIR / "directional_val.jsonl")
test_rows   = load_rows(DATA_DIR / "directional_test.jsonl")

train_val_texts = train_texts | val_texts

print(f"  Train unique texts : {len(train_texts)}")
print(f"  Val   unique texts : {len(val_texts)}")
print(f"  Train ∩ Val        : {len(train_texts & val_texts)}")
print(f"  Train ∪ Val        : {len(train_val_texts)}")
print(f"  Test rows          : {len(test_rows)}")
print()

leaked_rows = []
for line_num, row in test_rows:
    t1_leaked = row["text_1"] in train_val_texts
    t2_leaked = row["text_2"] in train_val_texts
    if t1_leaked or t2_leaked:
        leaked_rows.append({
            "line": line_num,
            "text_1_leaked": t1_leaked,
            "text_2_leaked": t2_leaked,
            "label": row.get("label"),
            "text_1_snippet": row["text_1"][:80],
            "text_2_snippet": row["text_2"][:80],
        })

if not leaked_rows:
    print("NO LEAKAGE DETECTED — test set is clean.")
else:
    print(f"LEAKAGE DETECTED — {len(leaked_rows)} test row(s) have texts that appear in train/val:\n")
    for r in leaked_rows:
        print(f"  Line {r['line']:>5} | label={r['label']} | text_1_leaked={r['text_1_leaked']} | text_2_leaked={r['text_2_leaked']}")
        if r["text_1_leaked"]:
            print(f"            text_1: {r['text_1_snippet']}...")
        if r["text_2_leaked"]:
            print(f"            text_2: {r['text_2_snippet']}...")

    # also break down by source
    t1_only = sum(1 for r in leaked_rows if r["text_1_leaked"] and not r["text_2_leaked"])
    t2_only = sum(1 for r in leaked_rows if r["text_2_leaked"] and not r["text_1_leaked"])
    both    = sum(1 for r in leaked_rows if r["text_1_leaked"] and r["text_2_leaked"])
    print()
    print(f"  text_1 leaked only : {t1_only}")
    print(f"  text_2 leaked only : {t2_only}")
    print(f"  both leaked        : {both}")

    # check if leaks come from train, val, or both
    print()
    print("Leak source breakdown:")
    train_leaked = sum(
        1 for _, row in test_rows
        if row["text_1"] in train_texts or row["text_2"] in train_texts
    )
    val_leaked = sum(
        1 for _, row in test_rows
        if row["text_1"] in val_texts or row["text_2"] in val_texts
    )
    print(f"  Rows leaking into train : {train_leaked}")
    print(f"  Rows leaking into val   : {val_leaked}")
