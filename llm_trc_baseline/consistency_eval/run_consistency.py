"""Run the full consistency comparison and write per-system metrics.

Systems (all read the same predicted-pairs files from predict_pairs.py):
  baseline_plain        - baseline CE, no cycle breaking            (LLM@inference: no)
  baseline_confidence   - baseline CE + confidence-based breaking   (no)   [paper's best]
  baseline_llm          - baseline CE + LLM-assisted breaking       (YES)  [paper's hybrid]
  reasoning_plain       - OUR reasoning-trace CE, no breaking       (no)   [ours]
  reasoning_confidence  - OUR reasoning-trace CE + confidence break (no)

Metrics per slice (all / curated / procedural / proc_n>=6): cycle_rate, mean_cycles,
ordering_accuracy (post-break where applicable), edges_removed, llm_calls.

Run (no LLM):     .venv/bin/python consistency_eval/run_consistency.py
Run (with LLM):   CUDA_VISIBLE_DEVICES=3 .venv/bin/python consistency_eval/run_consistency.py --with-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cycles import (build_digraph, count_simple_cycles, break_confidence, break_llm)  # noqa: E402

SLICES = {
    "all": lambda w: True,
    "curated": lambda w: w["origin"] == "curated",
    "procedural": lambda w: w["origin"] == "procedural",
    "proc_n>=6": lambda w: w["origin"] == "procedural" and w["n_steps"] >= 6,
}


def eval_system(workflows, breaker, llm=None):
    """breaker in {None,'confidence','llm'}. Returns per-slice metrics dict."""
    rows = []
    for wf in workflows:
        G = build_digraph(wf["edges"])
        total = len(wf["edges"])
        removed, calls = [], 0
        if breaker == "confidence":
            G, removed, calls = break_confidence(G)
        elif breaker == "llm":
            G, removed, calls = break_llm(G, wf["steps"], llm.choose)
        n_cyc = count_simple_cycles(G)
        correct = sum(1 for u, v in G.edges() if u < v)
        rows.append({"origin": wf["origin"], "n_steps": wf["n_steps"], "n_pairs": total,
                     "n_cycles": n_cyc, "correct": correct, "removed": len(removed),
                     "llm_calls": calls})
    out = {}
    for name, pred in SLICES.items():
        sub = [r for r, w in zip(rows, workflows) if pred(w)]
        n = len(sub)
        tot_pairs = sum(r["n_pairs"] for r in sub)
        cyc_wfs = sum(1 for r in sub if r["n_cycles"] > 0)
        out[name] = {
            "n_workflows": n,
            "cycle_rate": cyc_wfs / n if n else 0.0,
            "n_cyclic_workflows": cyc_wfs,
            "mean_cycles_per_wf": sum(r["n_cycles"] for r in sub) / n if n else 0.0,
            "ordering_accuracy": sum(r["correct"] for r in sub) / tot_pairs if tot_pairs else 0.0,
            "edges_removed": sum(r["removed"] for r in sub),
            "llm_calls": sum(r["llm_calls"] for r in sub),
            "n_pairs": tot_pairs,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(HERE.parent / "outputs/consistency"))
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--gpu-mem-util", type=float, default=0.70)
    ap.add_argument("--out", default=str(HERE.parent / "outputs/consistency/consistency_results.json"))
    args = ap.parse_args()

    pdir = Path(args.pairs_dir)
    baseline = json.loads((pdir / "pairs_baseline.json").read_text())["workflows"]
    reasoning = json.loads((pdir / "pairs_reasoning.json").read_text())["workflows"]

    results = {
        "baseline_plain": {"llm_at_inference": False, "note": "baseline CE, no breaking",
                           "metrics": eval_system(baseline, None)},
        "baseline_confidence": {"llm_at_inference": False, "note": "baseline + confidence break (paper best)",
                                "metrics": eval_system(baseline, "confidence")},
        "reasoning_plain": {"llm_at_inference": False, "note": "OURS: reasoning-trace CE, no breaking",
                            "metrics": eval_system(reasoning, None)},
        "reasoning_confidence": {"llm_at_inference": False, "note": "OURS + confidence break",
                                 "metrics": eval_system(reasoning, "confidence")},
    }

    if args.with_llm:
        from llm_breaker import LLMBreaker
        print(f"loading LLM breaker: {args.llm_model} ...", flush=True)
        llm = LLMBreaker(model=args.llm_model, gpu_mem_util=args.gpu_mem_util)
        results["baseline_llm"] = {"llm_at_inference": True,
                                   "note": f"baseline + LLM-assisted break ({args.llm_model})",
                                   "metrics": eval_system(baseline, "llm", llm=llm)}

    Path(args.out).write_text(json.dumps(results, indent=2))
    # console summary on the procedural slice
    print(f"\n{'system':24s} {'cyc_rate':>9s} {'mean_cyc':>9s} {'acc':>8s} {'removed':>8s} {'llm':>6s}  (procedural)")
    for name, r in results.items():
        m = r["metrics"]["procedural"]
        print(f"{name:24s} {m['cycle_rate']:9.3f} {m['mean_cycles_per_wf']:9.3f} "
              f"{m['ordering_accuracy']:8.4f} {m['edges_removed']:8d} {m['llm_calls']:6d}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
