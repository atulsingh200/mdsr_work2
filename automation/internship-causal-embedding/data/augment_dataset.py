import json
import random
import shutil
import pathlib

SRC = pathlib.Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34")
DST = pathlib.Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34_augmented")
DST.mkdir(exist_ok=True)

SEED = 42


def swap(row):
    return {
        "text_1": row["text_2"],
        "text_2": row["text_1"],
        "label": 1 - row["label"],
        "tier_1": row["tier_2"],
        "tier_2": row["tier_1"],
        "sub_1": row["sub_2"],
        "sub_2": row["sub_1"],
        "url_1": row["url_2"],
        "url_2": row["url_1"],
    }


def augment(split, shuffle=True):
    src_file = SRC / f"directional_{split}.jsonl"
    rows = [json.loads(line) for line in src_file.read_text().splitlines() if line.strip()]
    augmented = rows + [swap(r) for r in rows]
    if shuffle:
        rng = random.Random(SEED)
        rng.shuffle(augmented)
    out = DST / f"directional_{split}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in augmented))
    pos = sum(r["label"] for r in augmented)
    neg = len(augmented) - pos
    print(f"{split}: {len(rows)} original → {len(augmented)} augmented  (label=1: {pos}, label=0: {neg})")


augment("train")
augment("val")

shutil.copy(SRC / "directional_test.jsonl", DST / "directional_test.jsonl")
test_rows = sum(1 for l in (SRC / "directional_test.jsonl").read_text().splitlines() if l.strip())
print(f"test: copied unchanged ({test_rows} rows)")

# Update manifest
manifest = json.loads((SRC / "directional_manifest.json").read_text())
manifest["augmented"] = True
manifest["augmentation"] = "swapped_direction"
manifest["split_counts"] = {
    "train": sum(1 for l in (DST / "directional_train.jsonl").read_text().splitlines() if l.strip()),
    "val":   sum(1 for l in (DST / "directional_val.jsonl").read_text().splitlines() if l.strip()),
    "test":  test_rows,
}
(DST / "directional_manifest.json").write_text(json.dumps(manifest, indent=2))
print("Manifest written.")
print(f"\nOutput: {DST}")
