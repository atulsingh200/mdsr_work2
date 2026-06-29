"""
WorkflowLLM → anchor/positive pair extractor  (NL comments only)

Sliding window of 2 over the ordered comment sequence of each workflow.
A workflow with N comment steps produces N-1 pairs.

Output format per pair:
  {
    "query":    "NL description of the overall workflow goal",
    "anchor":   "NL comment explaining step t",
    "positive": "NL comment explaining step t+1"
  }

Input:  data.zip  (openbmb/WorkflowLLM, downloaded from HuggingFace)
Output:
  anchor_positive_pairs.jsonl       — full dataset, one pair per line
  splits/train_pairs.jsonl
  splits/dev_pairs.jsonl
  splits/test_pairs.jsonl
  splits/reward_pairs.jsonl
"""

import json
import zipfile
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ZIP_PATH   = SCRIPT_DIR / "data.zip"
SPLITS_DIR = SCRIPT_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)


# ── extract ordered NL comments from workflow_code ────────────────────────────
def extract_comments(workflow_code: str) -> list[str]:
    """
    Return only the NL comment lines from a workflow_code string,
    in order of appearance, with the leading '# ' stripped.

    Each comment describes one sequential step in natural language.
    Code lines (assignments, function calls) are discarded entirely.
    """
    comments = []
    for line in workflow_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # strip leading '#' and whitespace, keep the sentence
            text = stripped.lstrip("#").strip()
            if text:                        # skip empty comment lines
                comments.append(text)
    return comments


# ── sliding window of 2 → (anchor, positive) pairs ───────────────────────────
def emit_pairs(record: dict, source: str) -> list[dict]:
    """
    For a workflow with N ordered NL comments, emit N-1 pairs:
        (comment_0, comment_1), (comment_1, comment_2), ..., (comment_{N-2}, comment_{N-1})

    Each pair:  {"query": ..., "anchor": ..., "positive": ...}
    """
    comments = extract_comments(record.get("workflow_code", ""))
    if len(comments) < 2:
        return []

    query = (record.get("query") or "").strip()
    pairs = []
    for t in range(len(comments) - 1):
        pairs.append({
            "query":    query,
            "anchor":   comments[t],
            "positive": comments[t + 1],
        })
    return pairs


# ── load zip, extract all pairs, write output ──────────────────────────────────
def main():
    print(f"Opening {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open("seed_data.json") as f:
            seed_data = json.load(f)
        with z.open("synthesized_data.json") as f:
            synth_data = json.load(f)
        with z.open("dataset_split_keys.json") as f:
            split_keys = json.load(f)   # {"train": [...keys...], "dev": [...], "test": [...], "reward": [...]}

    print(f"  seed_data:        {len(seed_data):>7,} workflows")
    print(f"  synthesized_data: {len(synth_data):>7,} workflows")

    # build lookup: workflow key → split name
    key_to_split = {}
    for split_name, keys in split_keys.items():
        for k in keys:
            key_to_split[k] = split_name

    # ── extract pairs ─────────────────────────────────────────────────────────
    print("\nExtracting NL-comment anchor→positive pairs ...")

    all_pairs   = []
    split_pairs = {"train": [], "dev": [], "test": [], "reward": []}

    for rec in seed_data:
        pairs = emit_pairs(rec, source="seed")
        all_pairs.extend(pairs)
        split_name = key_to_split.get(rec.get("key", ""), "train")
        split_pairs[split_name].extend(pairs)

    for rec in synth_data:
        pairs = emit_pairs(rec, source="synthesized")
        all_pairs.extend(pairs)
        split_pairs["train"].extend(pairs)   # synthesized → train

    # ── write full JSONL ──────────────────────────────────────────────────────
    out_all = SCRIPT_DIR / "anchor_positive_pairs.jsonl"
    with open(out_all, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Saved anchor_positive_pairs.jsonl  ({len(all_pairs):,} pairs)")

    # ── write split JSONL files ───────────────────────────────────────────────
    for split_name, pairs in split_pairs.items():
        if not pairs:
            continue
        out = SPLITS_DIR / f"{split_name}_pairs.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"  Saved splits/{split_name}_pairs.jsonl  ({len(pairs):,} pairs)")

    # ── print samples ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SAMPLE PAIRS  (first 6, from first workflow)")
    print("="*70)
    for i, p in enumerate(all_pairs[:6]):
        print(f"\n[Pair {i+1}]")
        print(f"  QUERY   : {p['query'][:100]}...")
        print(f"  ANCHOR  : {p['anchor']}")
        print(f"  POSITIVE: {p['positive']}")

    # ── summary ───────────────────────────────────────────────────────────────
    total_workflows = len(seed_data) + len(synth_data)
    avg_pairs = len(all_pairs) / total_workflows if total_workflows else 0
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"  Total workflows  : {total_workflows:>8,}")
    print(f"  Avg pairs/wf     : {avg_pairs:>8.1f}")
    print(f"  Total pairs      : {len(all_pairs):>8,}")
    print(f"  Train pairs      : {len(split_pairs['train']):>8,}")
    print(f"  Dev   pairs      : {len(split_pairs['dev']):>8,}")
    print(f"  Test  pairs      : {len(split_pairs['test']):>8,}")
    print(f"  Reward pairs     : {len(split_pairs['reward']):>8,}")


if __name__ == "__main__":
    main()
