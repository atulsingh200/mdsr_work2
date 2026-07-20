"""Assemble the eval workflow set for the consistency / cycle-breaking study (paper 2).

Each workflow becomes a "document": an ordered list of step texts whose gold total order is the
list index (step 0 before step 1 before ... before step N-1). We combine:
  - ALL curated workflows from final_data/workflows_curated_eval_final.json (t1..tN, 2-4 steps)
  - a seeded sample of 5-8-step workflows from all_procedural_workflows.json (where cycles matter)

Output: workflows_eval.json  ->  [{id, source, origin, n_steps, steps:[...]}]

Run:
  .venv/bin/python consistency_eval/build_workflows.py --sample-5to8 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CER = Path("/mnt/localssd/causal-embedding-research")
CURATED = CER / "final_data/workflows_curated_eval_final.json"
PROCEDURAL = (CER / "automation/internship-causal-embedding/data"
              / "new_aep_workflow_scrap/all_procedural_workflows.json")
HERE = Path(__file__).parent


def load_curated() -> list[dict]:
    out = []
    for w in json.loads(CURATED.read_text()):
        n = int(w.get("num_steps", 0))
        steps = [w[f"t{i}"] for i in range(1, n + 1) if w.get(f"t{i}")]
        if len(steps) < 2:
            continue
        out.append({"id": f"curated_{w['id']}", "source": w.get("source", "?"),
                    "origin": "curated", "n_steps": len(steps), "steps": steps})
    return out


def load_procedural_sample(lo: int, hi: int, k: int, seed: int) -> list[dict]:
    pool = []
    for w in json.loads(PROCEDURAL.read_text()):
        steps = w.get("steps") or []
        if lo <= len(steps) <= hi and all(isinstance(s, str) and s.strip() for s in steps):
            pool.append(w)
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:k] if k and k < len(pool) else pool
    out = []
    for w in chosen:
        steps = [s.strip() for s in w["steps"]]
        out.append({"id": f"proc_{w.get('id')}", "source": w.get("title", "?"),
                    "origin": "procedural", "n_steps": len(steps), "steps": steps})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-5to8", type=int, default=200,
                    help="how many 5-8 step procedural workflows to sample (0 = all)")
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "workflows_eval.json"))
    args = ap.parse_args()

    curated = load_curated()
    proc = load_procedural_sample(args.min_steps, args.max_steps, args.sample_5to8, args.seed)
    allwf = curated + proc

    import collections
    by_origin = collections.Counter(w["origin"] for w in allwf)
    by_nsteps = collections.Counter(w["n_steps"] for w in allwf)
    total_pairs = sum(w["n_steps"] * (w["n_steps"] - 1) // 2 for w in allwf)
    print(f"curated: {len(curated)}  procedural(5-8): {len(proc)}  total: {len(allwf)}")
    print(f"by origin: {dict(by_origin)}")
    print(f"n_steps dist: {dict(sorted(by_nsteps.items()))}")
    print(f"total unordered pairs to score per model: {total_pairs}")

    Path(args.out).write_text(json.dumps(allwf, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
