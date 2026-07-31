#!/usr/bin/env python3
"""
Directional (step-order) pair builder for all_procedural_workflows.json.

For every workflow's `steps` list, forms pairs (step_i, step_j), i<j, where
gap = j-i-1 (steps skipped in between) is in {0,1,2,3}. For each pair, keeps
only ONE direction, chosen by a seeded coin flip:
  - forward (earlier -> later)  -> label 1
  - reverse (later -> earlier)  -> label 0
(never both for the same pair, matching an unordered-pair-at-inference setup).

Splits are document-level (by url, 80/10/10) since many docs contain multiple
extracted workflows -- splitting by individual workflow id would leak
near-duplicate content across train/val/test.

Output (new folder, same field schema as directional_test.jsonl):
  directional_procedural/directional_{train,val,test}.jsonl
    {text_1, text_2, label, step_1, step_2, url, title}
"""
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

GAPS = {0, 1, 2, 3}   # steps skipped between the pair (0 = adjacent)
SEED = 42


def main():
    here = Path(__file__).parent
    src = here / "all_procedural_workflows.json"
    out_dir = here / "directional_procedural"
    out_dir.mkdir(exist_ok=True)

    with open(src) as f:
        workflows = json.load(f)
    print(f"Loaded {len(workflows)} workflows", file=sys.stderr)

    rng = random.Random(SEED)

    # ---- document-level split (by url) 80/10/10, no leakage ----
    urls = sorted({wf["url"] for wf in workflows})
    rng.shuffle(urls)
    n = len(urls)
    ntest, nval = int(0.1 * n), int(0.1 * n)
    url_split = {}
    for rank, u in enumerate(urls):
        url_split[u] = "test" if rank < ntest else "val" if rank < ntest + nval else "train"

    # ---- build pairs: gap 0..3, single direction per pair ----
    rows = {"train": [], "val": [], "test": []}
    for wf in workflows:
        sp = url_split[wf["url"]]
        steps = wf["steps"]
        base = {"url": wf["url"], "title": wf["title"]}
        n_steps = len(steps)
        for i in range(n_steps):
            for j in range(i + 1, n_steps):
                gap = j - i - 1
                if gap not in GAPS:
                    continue
                if rng.random() < 0.5:
                    rows[sp].append({
                        "text_1": steps[i], "text_2": steps[j], "label": 1,
                        "step_1": f"t{i+1}", "step_2": f"t{j+1}", **base,
                    })
                else:
                    rows[sp].append({
                        "text_1": steps[j], "text_2": steps[i], "label": 0,
                        "step_1": f"t{j+1}", "step_2": f"t{i+1}", **base,
                    })

    # ---- exact-dup removal across splits; train wins ties (no eval leakage) ----
    seen = set()
    for sp in ("train", "val", "test"):
        kept = []
        for r in rows[sp]:
            key = (r["text_1"], r["text_2"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            kept.append(r)
        rows[sp] = kept

    for sp in ("train", "val", "test"):
        rng.shuffle(rows[sp])
        with open(out_dir / f"directional_{sp}.jsonl", "w") as f:
            for r in rows[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in rows.values())
    lbl = defaultdict(int)
    for r in rows["train"]:
        lbl[r["label"]] += 1
    print(f"docs: {n} (train={n-ntest-nval}, val={nval}, test={ntest})", file=sys.stderr)
    print(f"pairs: total={total} " + ", ".join(f"{sp}={len(rows[sp])}" for sp in rows), file=sys.stderr)
    print(f"train label balance: {dict(lbl)}", file=sys.stderr)
    print(f"wrote to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
