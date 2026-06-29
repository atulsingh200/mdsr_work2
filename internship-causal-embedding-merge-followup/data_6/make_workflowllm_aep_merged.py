#!/usr/bin/env python3
"""
Create merged classification dataset: WorkflowLLM nC2 pairs + AEP Causal.

Produces:
  workflowllm_aep_merged/directional_train.jsonl   (110775 aep + 100000 wf)
  workflowllm_aep_merged/directional_val.jsonl     (6004 aep only)
  workflowllm_aep_merged/directional_test.jsonl    (6004 aep only)
  workflowllm_aep_merged/manifest.json
"""

from __future__ import annotations
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
WF_JSONL = Path("/mnt/localssd/internship-causal-embedding-merge-followup/data_6/workflowllm/anchor_positive_pairs.jsonl")
AEP_DIR  = Path("/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34")
OUT_DIR  = Path("/mnt/localssd/internship-causal-embedding-merge-followup/data_6/workflowllm_aep_merged")

TOTAL_WF_SAMPLES = 100_000
WF_TRAIN_N = 100_000   # all WF samples go to train
WF_VAL_N   = 0
WF_TEST_N  = 0
SEED = 42


# ── PHASE 1: stream blocks and reconstruct step sequences ─────────────────────

def iter_blocks(path: Path):
    """
    Stream anchor_positive_pairs.jsonl and yield one ordered step list per
    contiguous block (same query = same logical workflow).

    Reconstruction: build fwd dict {anchor -> positive}, find unique start
    (anchor not appearing as any positive), chain forward. Skip broken blocks.
    """
    active_query = None
    block_pairs: list[tuple[str, str]] = []
    skipped = 0
    emitted = 0

    def emit(pairs):
        fwd = {a: p for a, p in pairs}
        all_pos = {p for _, p in pairs}
        starts = [a for a, _ in pairs if a not in all_pos]
        if len(starts) != 1:
            return None
        steps = [starts[0]]
        cur = starts[0]
        for _ in range(len(pairs)):
            if cur not in fwd:
                break
            cur = fwd[cur]
            steps.append(cur)
        return steps if len(steps) == len(pairs) + 1 else None

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["query"]
            if q != active_query:
                if block_pairs:
                    steps = emit(block_pairs)
                    if steps:
                        emitted += 1
                        yield steps
                    else:
                        skipped += 1
                block_pairs = []
                active_query = q
            block_pairs.append((r["anchor"], r["positive"]))

    if block_pairs:
        steps = emit(block_pairs)
        if steps:
            emitted += 1
            yield steps
        else:
            skipped += 1

    print(f"[iter_blocks] emitted={emitted:,}  skipped={skipped:,}", file=sys.stderr)


# ── PHASE 2: reservoir sampling of nC2 pairs (Algorithm R) ───────────────────

def reservoir_sample_nc2(path: Path, k: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    n_seen = 0
    t0 = time.time()
    blocks_done = 0

    for steps in iter_blocks(path):
        n = len(steps)
        blocks_done += 1
        for i, j in combinations(range(n), 2):
            n_seen += 1
            if len(reservoir) < k:
                reservoir.append((steps[i], steps[j]))
            else:
                r = rng.randint(0, n_seen - 1)
                if r < k:
                    reservoir[r] = (steps[i], steps[j])

        if blocks_done % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"[reservoir] blocks={blocks_done:,}  pairs_seen={n_seen:,}  "
                f"reservoir={len(reservoir):,}  elapsed={elapsed:.0f}s",
                file=sys.stderr,
            )

    elapsed = time.time() - t0
    print(
        f"[reservoir] DONE  blocks={blocks_done:,}  total_nc2_pairs={n_seen:,}  "
        f"sampled={len(reservoir):,}  elapsed={elapsed:.0f}s",
        file=sys.stderr,
    )
    return reservoir


# ── PHASE 3: assign direction (coin flip) and split ──────────────────────────

def assign_and_split(
    pairs: list[tuple[str, str]],
    train_n: int, val_n: int, test_n: int,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)
    records: list[dict] = []
    for step_i, step_j in pairs:
        if rng.random() < 0.5:
            records.append({"text_1": step_i, "text_2": step_j, "label": 1})
        else:
            records.append({"text_1": step_j, "text_2": step_i, "label": 0})

    rng.shuffle(records)
    return {
        "train": records[:train_n],
        "val":   records[train_n : train_n + val_n],
        "test":  records[train_n + val_n :],
    }


# ── PHASE 4: merge with AEP and write ────────────────────────────────────────

def merge_and_write(
    wf_by_split: dict[str, list[dict]],
    aep_dir: Path,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + 2)
    counts = {}

    for split in ("train", "val", "test"):
        aep_path = aep_dir / f"directional_{split}.jsonl"
        out_path = out_dir / f"directional_{split}.jsonl"

        aep_records = []
        with open(aep_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                aep_records.append({"text_1": r["text_1"], "text_2": r["text_2"], "label": r["label"]})

        combined = aep_records + wf_by_split[split]
        rng.shuffle(combined)

        with open(out_path, "w", encoding="utf-8") as out:
            for rec in combined:
                out.write(json.dumps(rec) + "\n")

        label_1 = sum(1 for r in combined if r["label"] == 1)
        label_0 = len(combined) - label_1
        counts[split] = {"total": len(combined), "label_1": label_1, "label_0": label_0}
        print(
            f"[write] {split}: total={len(combined):,}  label=1:{label_1:,}  label=0:{label_0:,}",
            file=sys.stderr,
        )

    return counts


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("[phase 1+2] reservoir sampling nC2 pairs from WorkflowLLM ...", file=sys.stderr)
    pairs = reservoir_sample_nc2(WF_JSONL, TOTAL_WF_SAMPLES, SEED)

    print("[phase 3] assigning directions and splits ...", file=sys.stderr)
    wf_by_split = assign_and_split(pairs, WF_TRAIN_N, WF_VAL_N, WF_TEST_N, SEED)

    wf_label_counts = {}
    for split, recs in wf_by_split.items():
        l1 = sum(1 for r in recs if r["label"] == 1)
        wf_label_counts[split] = {"label_1": l1, "label_0": len(recs) - l1}

    print("[phase 4] merging with AEP and writing output ...", file=sys.stderr)
    merged_counts = merge_and_write(wf_by_split, AEP_DIR, OUT_DIR)

    manifest = {
        "source_wf": str(WF_JSONL),
        "source_aep": str(AEP_DIR),
        "seed": SEED,
        "total_wf_sampled": len(pairs),
        "wf_split_target": {"train": WF_TRAIN_N, "val": WF_VAL_N, "test": WF_TEST_N},
        "wf_label_counts": wf_label_counts,
        "aep_counts": {"train": 110775, "val": 6004, "test": 6004},
        "merged_counts": merged_counts,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] output written to {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
