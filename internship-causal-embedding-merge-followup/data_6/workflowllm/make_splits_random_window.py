"""
WorkflowLLM — random-window anchor/positive pair extractor
===========================================================
Source: raw_workflows.json  (merged seed + synthesized workflows)

For each workflow, for each valid position t (where step t+1 exists), we
randomly pick a concatenation window size k from {1, 2, 3, 4} (constrained
to what is available at position t):

    k=1  anchor = comment[t]
    k=2  anchor = comment[t-1] -> comment[t]
    k=3  anchor = comment[t-2] -> comment[t-1] -> comment[t]
    k=4  anchor = comment[t-3] -> comment[t-2] -> comment[t-1] -> comment[t]
    positive = comment[t+1]   (always the next step)

Steps are joined with  " -> "  to mark temporal ordering.

Splits are made at the WORKFLOW level (not pair level) so no workflow
leaks across splits:
    train : 80 %
    dev   : 10 %
    test  : 10 %

Output (written to splits_random_window/):
    train_pairs.jsonl
    dev_pairs.jsonl
    test_pairs.jsonl

Each line:
    {
      "query":    "<overall workflow goal>",
      "anchor":   "<k concatenated NL comments>",
      "positive": "<next NL comment>",
      "variant":  <k>,          # window size actually used
      "source":   "seed" | "synthesized",
      "workflow_key": "..."
    }
"""

import json
import random
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RAW_PATH    = SCRIPT_DIR / "raw_workflows.json"
OUT_DIR     = SCRIPT_DIR / "splits_random_window"
STEP_SEP    = " "      # separator between concatenated comments
TRAIN_FRAC  = 0.80
DEV_FRAC    = 0.10
# test gets the remainder (≈10 %)
SEED        = 42


# ── helpers ───────────────────────────────────────────────────────────────────
def clean_comment(raw: str) -> str:
    """Strip leading '# ' from a comment line."""
    s = raw.strip()
    if s.startswith("#"):
        s = s.lstrip("#").strip()
    return s


def extract_comments(steps: list[dict]) -> list[str]:
    """Return ordered list of cleaned NL comments from a workflow's steps."""
    out = []
    for step in sorted(steps, key=lambda s: s.get("step_index", 0)):
        c = clean_comment(step.get("comment", ""))
        if c:
            out.append(c)
    return out


def emit_pairs(workflow: dict, rng: random.Random) -> list[dict]:
    """
    For a workflow, emit one pair per valid position t (where comment[t+1] exists).
    Window size k is chosen randomly from the variants available at t.
    """
    comments = extract_comments(workflow.get("steps", []))
    if len(comments) < 2:
        return []

    query        = (workflow.get("workflow_query") or "").strip()
    source       = workflow.get("source", "")
    workflow_key = workflow.get("workflow_key", "")
    pairs        = []

    for t in range(len(comments) - 1):
        # available variants at position t: k=1 always; k=2 needs t>=1, etc.
        max_k     = min(4, t + 1)          # t=0 → max_k=1, t=1 → 2, …
        k         = rng.randint(1, max_k)  # pick randomly from available
        start     = t - k + 1             # inclusive start index
        anchor    = STEP_SEP.join(comments[start : t + 1])
        positive  = comments[t + 1]

        pairs.append({
            "query":        query,
            "anchor":       anchor,
            "positive":     positive,
            "variant":      k,
            "source":       source,
            "workflow_key": workflow_key,
        })

    return pairs


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(SEED)

    print(f"Loading {RAW_PATH} ...")
    with open(RAW_PATH, encoding="utf-8") as f:
        workflows = json.load(f)
    print(f"  {len(workflows):,} workflows loaded")

    # ── shuffle and split at workflow level ───────────────────────────────────
    indices = list(range(len(workflows)))
    rng.shuffle(indices)

    n_train = int(len(indices) * TRAIN_FRAC)
    n_dev   = int(len(indices) * DEV_FRAC)
    train_idx = set(indices[:n_train])
    dev_idx   = set(indices[n_train : n_train + n_dev])
    test_idx  = set(indices[n_train + n_dev :])

    print(f"  Workflow split → train={len(train_idx):,}  dev={len(dev_idx):,}  test={len(test_idx):,}")

    # ── extract pairs per split ───────────────────────────────────────────────
    split_pairs: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for i, wf in enumerate(workflows):
        if i in train_idx:
            split = "train"
        elif i in dev_idx:
            split = "dev"
        else:
            split = "test"
        split_pairs[split].extend(emit_pairs(wf, rng))

    # ── write output ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)

    for split_name, pairs in split_pairs.items():
        out_path = OUT_DIR / f"{split_name}_pairs.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"  Saved {out_path.name}  ({len(pairs):,} pairs)")

    # ── variant distribution stats ────────────────────────────────────────────
    all_pairs = split_pairs["train"] + split_pairs["dev"] + split_pairs["test"]
    variant_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in all_pairs:
        variant_counts[p["variant"]] += 1
    total = len(all_pairs)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"  Total pairs   : {total:>10,}")
    print(f"  Train pairs   : {len(split_pairs['train']):>10,}")
    print(f"  Dev   pairs   : {len(split_pairs['dev']):>10,}")
    print(f"  Test  pairs   : {len(split_pairs['test']):>10,}")
    print(f"\nVariant distribution (anchor window size):")
    for k, cnt in sorted(variant_counts.items()):
        print(f"  k={k}  {cnt:>10,}  ({cnt/total*100:.1f} %)")

    # ── show a sample from each variant ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAMPLES (one per variant from train)")
    print("="*60)
    shown = {}
    for p in split_pairs["train"]:
        k = p["variant"]
        if k not in shown:
            shown[k] = p
        if len(shown) == 4:
            break
    for k in sorted(shown):
        p = shown[k]
        print(f"\n[variant k={k}]")
        print(f"  QUERY   : {p['query'][:90]}")
        print(f"  ANCHOR  : {p['anchor'][:200]}")
        print(f"  POSITIVE: {p['positive']}")


if __name__ == "__main__":
    main()
