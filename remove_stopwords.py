"""
Remove stop words from text_1 and text_2 fields in aep_causal_classification_34 dataset.
Produces a new dataset in aep_causal_classification_34_nostop/.

Stop words removed:
  - NLTK English stop words
  - Domain-specific additions (contraction artifacts, single-char tokens, corpus-specific noise)
"""

import json
import shutil
import sys
from pathlib import Path

import nltk

# Download required NLTK data silently if not present
for resource in ("punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Stop word set ──────────────────────────────────────────────────────────────

NLTK_STOPS = set(stopwords.words("english"))

# Tokens that are high-frequency noise specific to this corpus:
#   's', 're', 'll', 've', 'd'  — contraction fragments after word_tokenize
#   'e', 'f', 'i', 'n'          — single-char artifacts from hyphenated terms / abbreviations
#   'also', 'one'               — additive fillers with no causal signal in this domain
DOMAIN_STOPS = {
    "s", "re", "ll", "ve", "d", "m", "t",  # contraction fragments
    "e", "f", "i", "n",                      # single-char artifacts
    "also", "one",                            # domain noise
}

ALL_STOPS = NLTK_STOPS | DOMAIN_STOPS


def clean_text(text: str) -> str:
    tokens = word_tokenize(text)
    kept = [
        tok for tok in tokens
        if len(tok) > 1 and tok.lower() not in ALL_STOPS
    ]
    return " ".join(kept)


# ── Paths ──────────────────────────────────────────────────────────────────────

SRC_DIR = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34")
DST_DIR = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34_nostop")
DST_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["directional_train.jsonl", "directional_val.jsonl", "directional_test.jsonl"]

# ── Process each split ─────────────────────────────────────────────────────────

print(f"Stop word set size: {len(ALL_STOPS)} tokens")
print(f"  NLTK: {len(NLTK_STOPS)}  |  domain additions: {len(DOMAIN_STOPS)}")
print()

for split_name in SPLITS:
    src_path = SRC_DIR / split_name
    dst_path = DST_DIR / split_name

    if not src_path.exists():
        print(f"[SKIP] {split_name} not found at {src_path}")
        continue

    total_orig_1, total_clean_1 = 0, 0
    total_orig_2, total_clean_2 = 0, 0
    n_rows = 0

    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            orig_1 = obj["text_1"]
            orig_2 = obj["text_2"]
            clean_1 = clean_text(orig_1)
            clean_2 = clean_text(orig_2)

            obj["text_1"] = clean_1
            obj["text_2"] = clean_2

            fout.write(json.dumps(obj) + "\n")

            total_orig_1 += len(orig_1.split())
            total_clean_1 += len(clean_1.split())
            total_orig_2 += len(orig_2.split())
            total_clean_2 += len(clean_2.split())
            n_rows += 1

    avg_orig = (total_orig_1 + total_orig_2) / (2 * n_rows)
    avg_clean = (total_clean_1 + total_clean_2) / (2 * n_rows)
    reduction = 100 * (1 - avg_clean / avg_orig)

    print(f"{split_name}")
    print(f"  rows: {n_rows:,}")
    print(f"  avg tokens/text  before: {avg_orig:.1f}  after: {avg_clean:.1f}  reduction: {reduction:.1f}%")
    print()

# Copy manifest unchanged
manifest_src = SRC_DIR / "directional_manifest.json"
if manifest_src.exists():
    shutil.copy2(manifest_src, DST_DIR / "directional_manifest.json")
    print("Copied directional_manifest.json")

print(f"\nDone. Output → {DST_DIR}")

# ── Spot-check ─────────────────────────────────────────────────────────────────

print("\n── Spot-check (3 examples from train) ──")
src_path = SRC_DIR / "directional_train.jsonl"
dst_path = DST_DIR / "directional_train.jsonl"
with open(src_path) as fs, open(dst_path) as fd:
    for i in range(3):
        orig = json.loads(fs.readline())
        clean = json.loads(fd.readline())
        print(f"\nExample {i+1}:")
        print(f"  ORIGINAL  text_1: {orig['text_1'][:120]!r}")
        print(f"  CLEANED   text_1: {clean['text_1'][:120]!r}")
        assert orig["label"] == clean["label"], "label mismatch!"
        assert orig["url_1"] == clean["url_1"], "url_1 mismatch!"
print("\nAll checks passed.")
