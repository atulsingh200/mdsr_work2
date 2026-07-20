"""Head-to-head consistency analysis: baseline CE vs our reasoning CE, per workflow.

Reports (honestly, both directions) the cyclic/acyclic confusion matrix, and the key conditional
stat: of the workflows where the BASELINE is inconsistent (has a cycle) — the exact cases where the
paper's hybrid would fire an inference-time LLM — how many our reasoning model resolves NATIVELY
(no LLM). Also writes a curated `showcase_winning_workflows.json` (baseline-cyclic AND ours-clean)
for qualitative case-study examples, and the reverse set for full transparency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cycles import build_digraph, count_simple_cycles  # noqa: E402

OUT = HERE.parent / "outputs/consistency"
SLICES = {
    "all": lambda w: True,
    "procedural": lambda w: w["origin"] == "procedural",
    "proc_n>=6": lambda w: w["origin"] == "procedural" and w["n_steps"] >= 6,
}


def load(tag):
    d = json.loads((OUT / f"pairs_{tag}.json").read_text())
    out = {}
    for wf in d["workflows"]:
        G = build_digraph(wf["edges"])
        out[wf["id"]] = {"wf": wf, "n_cycles": count_simple_cycles(G), "G": G}
    return out


def wf_detail(rec_b, rec_r):
    wf = rec_b["wf"]
    def edges(rec):
        es = []
        for e in wf["edges"]:
            u, v = (e["i"], e["j"]) if e["pred_before"] else (e["j"], e["i"])
            es.append({"edge": [u, v], "confidence": e["confidence"], "correct": u < v})
        return es
    return {
        "id": wf["id"], "origin": wf["origin"], "n_steps": wf["n_steps"],
        "steps": wf["steps"],
        "baseline_n_cycles": rec_b["n_cycles"], "reasoning_n_cycles": rec_r["n_cycles"],
    }


def main():
    B, R = load("baseline"), load("reasoning")
    report = {}
    win_ids, lose_ids = [], []
    for name, pred in SLICES.items():
        both = onlyB = onlyR = neither = 0
        for i, rb in B.items():
            wf = rb["wf"]
            if not pred(wf):
                continue
            bc, rc = rb["n_cycles"] > 0, R[i]["n_cycles"] > 0
            if bc and rc:
                both += 1
            elif bc and not rc:
                onlyB += 1
                if name == "all":
                    win_ids.append(i)
            elif rc and not bc:
                onlyR += 1
                if name == "all":
                    lose_ids.append(i)
            else:
                neither += 1
        base_cyc = both + onlyB
        report[name] = {
            "both_cyclic": both, "baseline_only_cyclic_OURS_WINS": onlyB,
            "reasoning_only_cyclic_ours_worse": onlyR, "neither_cyclic": neither,
            "baseline_cyclic_total": base_cyc,
            "ours_resolves_natively_pct": round(onlyB / base_cyc, 4) if base_cyc else 0.0,
            "net_cyclic_baseline": base_cyc, "net_cyclic_reasoning": both + onlyR,
        }

    (OUT / "disagreement.json").write_text(json.dumps(report, indent=2))

    # curated showcase files (for qualitative examples), sorted by baseline cycle count
    wins = sorted((wf_detail(B[i], R[i]) for i in win_ids),
                  key=lambda d: -d["baseline_n_cycles"])
    loses = sorted((wf_detail(B[i], R[i]) for i in lose_ids),
                   key=lambda d: -d["reasoning_n_cycles"])
    (OUT / "showcase_winning_workflows.json").write_text(json.dumps(wins, indent=2))
    (OUT / "showcase_reverse_workflows.json").write_text(json.dumps(loses, indent=2))

    print("=== consistency head-to-head (baseline CE vs OUR reasoning CE) ===\n")
    for name, r in report.items():
        print(f"[{name}]  baseline-cyclic={r['baseline_cyclic_total']}  "
              f"-> OURS resolves natively (no LLM): {r['baseline_only_cyclic_OURS_WINS']} "
              f"({r['ours_resolves_natively_pct']:.0%})   "
              f"| ours-worse cases: {r['reasoning_only_cyclic_ours_worse']}   "
              f"| net cyclic: base {r['net_cyclic_baseline']} vs ours {r['net_cyclic_reasoning']}")
    print(f"\nwrote {OUT/'disagreement.json'}")
    print(f"wrote {OUT/'showcase_winning_workflows.json'}  ({len(wins)} workflows)")
    print(f"wrote {OUT/'showcase_reverse_workflows.json'}  ({len(loses)} workflows, for transparency)")


if __name__ == "__main__":
    main()
