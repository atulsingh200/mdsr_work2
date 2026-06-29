"""Unified evaluation CLI — one command to evaluate any model on any dataset.

Examples
--------
# Evaluate CrossEncoder-2x on aep_causal with BM25 negatives
uv run python scripts/run_eval.py \\
    --model cross_encoder_2x:my_work/kl/cross_encoder_2x/results/aep_causal/checkpoint_best.pt \\
    --dataset aep_causal \\
    --negatives bm25 \\
    --split test

# Compare BiEncoder vs pretrained on two datasets with two negative types
uv run python scripts/run_eval.py \\
    --model biencoder:finetune_eval/results/aep_causal/checkpoint_best.pt \\
    --model pretrained:BAAI/bge-small-en-v1.5 \\
    --dataset aep_causal --dataset followupqg \\
    --negatives random --negatives bm25 \\
    --split test

# Run from a frozen YAML config (fully reproducible)
uv run python scripts/run_eval.py --config configs/aep_causal_bm25.yaml

Model spec format
-----------------
  biencoder:<checkpoint_path>
  cde:<checkpoint_path>
  cross_encoder_2x:<checkpoint_path>
  lorentz:<checkpoint_path>
  pretrained:<hf_model_id>      (e.g. pretrained:BAAI/bge-small-en-v1.5)
  tfidf                         (no path needed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parse_model_spec(spec: str):
    """Parse 'type:path_or_id' or 'tfidf' into (model_type, kwargs)."""
    if ":" not in spec:
        return spec, {}
    model_type, rest = spec.split(":", 1)
    model_type = model_type.strip()
    rest = rest.strip()

    if model_type == "pretrained":
        return model_type, {"hf_id": rest}
    else:
        return model_type, {"checkpoint": rest}


def load_config(path: Path) -> dict:
    """Load a YAML experiment config. Requires PyYAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for --config: pip install pyyaml")
    with path.open() as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", action="append", dest="models", metavar="SPEC",
        help="Model spec (repeatable). Format: type:path or pretrained:hf_id or tfidf.",
    )
    ap.add_argument(
        "--dataset", action="append", dest="datasets", metavar="NAME",
        help="Dataset name (repeatable). Any registered followup_data dataset.",
    )
    ap.add_argument(
        "--negatives", action="append", dest="negatives",
        choices=["random", "shuffle", "bm25"],
        help="Negative strategy (repeatable): random, shuffle, bm25.",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pool-size", type=int, default=1000,
                    help="Candidate pool size for MRR/Recall@K.")
    ap.add_argument("--k-negatives", type=int, default=4,
                    help="Negatives per anchor for AUC/P@1.")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                    help="K values for Recall@K and Hit@K.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=0,
                    help="Cap queries per dataset (0=all). For smoke tests.")
    ap.add_argument("--out-dir", default="results",
                    help="Output directory. Results saved under out-dir/<timestamp>/")
    ap.add_argument("--config", default=None,
                    help="YAML config file. CLI flags override config values.")
    ap.add_argument("--device", default=None,
                    help="Force device: cuda, cpu, mps. Default: auto-detect.")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve config: YAML first, then CLI overrides
    # ------------------------------------------------------------------ #
    cfg: dict = {}
    if args.config:
        cfg = load_config(Path(args.config))

    model_specs  = args.models    or [m["type"] + ":" + m.get("checkpoint", m.get("hf_id", ""))
                                       for m in cfg.get("models", [])]
    datasets     = args.datasets  or cfg.get("datasets", [])
    neg_strats   = args.negatives or cfg.get("negatives", ["random"])
    split        = args.split     or cfg.get("splits", ["test"])[0]
    pool_size    = args.pool_size or cfg.get("eval", {}).get("pool_size", 1000)
    k_negatives  = args.k_negatives or cfg.get("eval", {}).get("k_negatives", 4)
    k_values     = sorted(set(args.k or cfg.get("eval", {}).get("k_values", [1, 3, 5, 10])))
    seed         = args.seed      or cfg.get("eval", {}).get("seed", 42)
    max_queries  = args.max_queries or cfg.get("eval", {}).get("max_queries", 0)
    out_dir      = Path(args.out_dir or cfg.get("output", {}).get("dir", "results"))

    if not model_specs:
        ap.error("At least one --model is required.")
    if not datasets:
        ap.error("At least one --dataset is required.")

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] output → {run_dir}")
    print(f"      models:   {model_specs}")
    print(f"      datasets: {datasets}")
    print(f"      negatives:{neg_strats}")
    print(f"      split:    {split}  pool={pool_size}  k_neg={k_negatives}  seed={seed}")

    # ------------------------------------------------------------------ #
    # Run all (model × dataset × negatives) combos
    # ------------------------------------------------------------------ #
    from models.registry import load_retriever
    from evaluations.runner import run_eval
    from evaluations.report import print_table, save_csv, save_json

    all_results: list[dict] = []
    errors: list[dict] = []

    for model_spec in model_specs:
        model_type, model_kwargs = parse_model_spec(model_spec)
        if args.device:
            model_kwargs["device"] = args.device

        print(f"\n{'='*60}")
        print(f"Loading model: {model_spec}")
        try:
            retriever = load_retriever(model_type, **model_kwargs)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            errors.append({"model_spec": model_spec, "error": str(e)})
            continue

        for dataset_name in datasets:
            for neg_strategy in neg_strats:
                print(f"\n  >> {retriever.name} | {dataset_name} | {neg_strategy}")
                t0 = time.time()
                try:
                    result = run_eval(
                        retriever=retriever,
                        dataset_name=dataset_name,
                        split=split,
                        negative_strategy=neg_strategy,
                        k_negatives=k_negatives,
                        pool_size=pool_size,
                        seed=seed,
                        k_values=k_values,
                        max_queries=max_queries,
                    )
                    result["wall_seconds"] = time.time() - t0
                    all_results.append(result)

                    # Save individual run immediately (safe against crashes)
                    fname = f"{retriever.name}__{dataset_name}__{neg_strategy}__{split}.json"
                    (run_dir / "runs").mkdir(exist_ok=True)
                    (run_dir / "runs" / fname).write_text(json.dumps(result, indent=2))

                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    errors.append({
                        "model": model_spec,
                        "dataset": dataset_name,
                        "negatives": neg_strategy,
                        "error": str(e),
                    })

    # ------------------------------------------------------------------ #
    # Save summary + print table
    # ------------------------------------------------------------------ #
    save_json(all_results, run_dir / "summary.json")
    save_csv(all_results, run_dir / "summary.csv")
    if errors:
        save_json(errors, run_dir / "errors.json")

    print_table(all_results)
    print(f"\n[done] {len(all_results)} runs completed, {len(errors)} errors.")
    print(f"       results → {run_dir}")


if __name__ == "__main__":
    main()
