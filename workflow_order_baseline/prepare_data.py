"""Prepare the workflow step-ordering baseline data.

Task (following arXiv:2511.04688v2): given a shuffled list of procedural workflow steps,
recover the correct order. We shuffle each eval workflow deterministically (seed keyed by id)
and store the gold ordering, then pick one fixed set of few-shot demos from the train pool.

Conventions (1-indexed, matching how steps are numbered in the prompt):
  - shuffled_steps: the step texts in shuffled order (what the model sees, numbered 1..n).
  - gold_order:     list s.t. reading shuffled_steps at these 1-indexed positions yields the
                    correct original order. This is what the model must predict as "order".
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CER = Path("/mnt/localssd/causal-embedding-research")
EVAL_PATH = CER / "final_data/workflows_curated_eval_final.json"
TRAIN_PATH = CER / "automation/internship-causal-embedding/data/new_aep_workflow_scrap/all_procedural_workflows.json"
OUT_DIR = Path(__file__).parent / "data"

SHUFFLE_SEED_BASE = 20251121  # combined with per-record id for reproducibility


def shuffled_with_gold(steps: list[str], seed: int) -> tuple[list[str], list[int]]:
    """Return (shuffled_steps, gold_order) where gold_order is 1-indexed positions into
    shuffled_steps that recover the original order. Re-shuffles until non-identity when n>1."""
    n = len(steps)
    rng = random.Random(seed)
    idx = list(range(n))  # idx[j] = original position of the j-th shuffled step
    for _ in range(20):
        rng.shuffle(idx)
        if n == 1 or idx != list(range(n)):
            break
    shuffled = [steps[j] for j in idx]
    # gold_order[k] (0-indexed k) = shuffled position (1-indexed) holding original step k
    pos_of_original = {orig: j for j, orig in enumerate(idx)}
    gold_order = [pos_of_original[k] + 1 for k in range(n)]
    return shuffled, gold_order


def load_eval() -> list[dict]:
    raw = json.load(open(EVAL_PATH))
    rows = []
    for r in raw:
        n = r["num_steps"]
        steps = [r[f"t{i}"] for i in range(1, n + 1)]
        assert len(steps) == n and all(isinstance(s, str) for s in steps), r["id"]
        shuffled, gold = shuffled_with_gold(steps, SHUFFLE_SEED_BASE + r["id"])
        rows.append({
            "id": r["id"], "title": r["title"], "source": r.get("source"),
            "num_steps": n, "correct_steps": steps,
            "shuffled_steps": shuffled, "gold_order": gold,
        })
    return rows


def pick_fewshot(eval_titles: set[str]) -> dict:
    train = json.load(open(TRAIN_PATH))
    by_n: dict[int, list[dict]] = {2: [], 3: [], 4: []}
    for r in train:
        n = r["num_steps"]
        if n in by_n and r["title"] not in eval_titles and len(r["steps"]) == n:
            by_n[n].append(r)
    for n in (2, 3, 4):
        print(f"  train demo candidates with {n} steps: {len(by_n[n])}")

    rng = random.Random(SHUFFLE_SEED_BASE)
    for n in by_n:
        rng.shuffle(by_n[n])

    # NOTE: the train pool has no 2-step workflows (min is 3 steps), so we cannot mirror the
    # paper's exact per-length composition. We fall back to the available step counts {3,4}:
    #   5-shot superset: 3x(3-step), 2x(4-step); 3-shot = first 2x(3-step) + 1x(4-step).
    plan_5 = [3, 3, 3, 4, 4]
    used = {3: 0, 4: 0}
    demos = []
    for n in plan_5:
        cand = by_n[n][used[n]]
        used[n] += 1
        shuffled, gold = shuffled_with_gold(cand["steps"], SHUFFLE_SEED_BASE * 7 + cand["id"])
        demos.append({"id": cand["id"], "title": cand["title"], "num_steps": n,
                      "shuffled_steps": shuffled, "gold_order": gold})
    fewshot = {"shots_3": [demos[0], demos[1], demos[3]], "shots_5": demos}
    return fewshot


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_rows = load_eval()
    eval_titles = {r["title"] for r in eval_rows}
    print(f"eval rows: {len(eval_rows)}")
    dist = {}
    for r in eval_rows:
        dist[r["num_steps"]] = dist.get(r["num_steps"], 0) + 1
    print(f"  step-count distribution: {dict(sorted(dist.items()))}")

    fewshot = pick_fewshot(eval_titles)

    with open(OUT_DIR / "eval_prepared.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    json.dump(fewshot, open(OUT_DIR / "fewshot_examples.json", "w"), indent=2)
    print(f"wrote {OUT_DIR}/eval_prepared.jsonl and fewshot_examples.json")

    # spot-check: reconstruct correct order from shuffled + gold and verify
    bad = 0
    for r in eval_rows:
        recon = [r["shuffled_steps"][k - 1] for k in r["gold_order"]]
        if recon != r["correct_steps"]:
            bad += 1
    print(f"reconstruction check: {bad} mismatches out of {len(eval_rows)}")


if __name__ == "__main__":
    main()
