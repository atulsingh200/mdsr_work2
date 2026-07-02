"""
Sweep EVERY causal-LM checkpoint under a models root (default: pythia_train),
rerank match_results.jsonl by full-sequence perplexity of
"question + ' ' + follow_up", and report hit@k / recall@k (k=1,3,5) for each
model. Prints a leaderboard and the best score achievable per metric.

Reuses the scoring + evaluation logic from rerank_gpt2_ppl.py so every model
is measured identically to the single-model runs.

Usage:
    python sweep_ppl_models.py \\
        --input match_results.jsonl \\
        --models-root /mnt/localssd/automation/internship-causal-embedding/pythia_train \\
        --out-dir sweep_ppl \\
        [--include-checkpoints] [--batch-size 32] [--device cuda]
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# reuse the exact scoring + metric logic from the single-model script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_gpt2_ppl import (  # noqa: E402
    PairDataset, make_collate, sequence_perplexity, compute_metrics,
)

KS = (1, 3, 5)
METRIC_KEYS = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS]


def has_weights(d: Path) -> bool:
    return (
        (d / "model.safetensors").exists()
        or (d / "pytorch_model.bin").exists()
        or any(d.glob("model-*.safetensors"))
        or any(d.glob("pytorch_model-*.bin"))
    )


def discover_models(root: Path, include_checkpoints: bool) -> list[Path]:
    """Every directory with a config.json + weights. Top-level dirs first;
    optionally their checkpoint-* subdirs."""
    found: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "config.json").exists() and has_weights(d):
            found.append(d)
        if include_checkpoints:
            for sub in sorted(d.glob("checkpoint-*"), key=lambda p: p.name):
                if (sub / "config.json").exists() and has_weights(sub):
                    found.append(sub)
    return found


def eval_one_model(model_dir: Path, records: list, args, device: torch.device,
                   root: Path) -> dict:
    label = str(model_dir.relative_to(root))
    use_bf16 = device.type == "cuda"

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # correct absolute/rotary positions for perplexity

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16 if use_bf16 else torch.float32,
    ).to(device).eval()

    n_pos = getattr(model.config, "n_positions", None) \
        or getattr(model.config, "max_position_embeddings", 1024)
    max_len = min(args.max_len, n_pos)

    dataset = PairDataset(records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate(tok, max_len), num_workers=0)

    all_scores: list[dict] = [{} for _ in records]
    for enc, rec_idxs, fup_idxs in tqdm(loader, desc=label, leave=False):
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        ppl = sequence_perplexity(model, input_ids, attn, use_bf16).float().cpu().tolist()
        for s, r, fidx in zip(ppl, rec_idxs, fup_idxs):
            all_scores[r][fidx] = s

    metrics = compute_metrics(records, all_scores, ks=KS)

    if args.write_reranked:
        out = Path(args.out_dir) / f"reranked_ppl_{label.replace('/', '__')}.jsonl"
        with open(out, "w") as f:
            for ri, rec in enumerate(records):
                n = len(rec["generated_follow_ups"])
                order = sorted(range(n), key=lambda i: all_scores[ri].get(i, float("inf")))
                reranked = [rec["generated_follow_ups"][i] for i in order]
                f.write(json.dumps({**rec, "generated_follow_ups": reranked},
                                   ensure_ascii=False) + "\n")

    n_params = sum(p.numel() for p in model.parameters())
    model_type = getattr(model.config, "model_type", "?")

    # free GPU before the next model
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": label,
        "model_type": model_type,
        "n_params": n_params,
        "max_len": max_len,
        **{k: metrics[k] for k in METRIC_KEYS},
        "n_eval": metrics["n_eval"],
    }


def print_leaderboard(rows: list[dict], sort_metric: str):
    rows = sorted(rows, key=lambda r: r[sort_metric], reverse=True)
    name_w = max(len(r["model"]) for r in rows) + 2

    header = f"{'model':<{name_w}}{'type':<10}" + "".join(f"{k:>11}" for k in METRIC_KEYS)
    print("\n" + "=" * len(header))
    print(f"LEADERBOARD  (sorted by {sort_metric} desc, n_eval={rows[0]['n_eval']})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['model']:<{name_w}}{r['model_type']:<10}"
        line += "".join(f"{r[k]:>11.4f}" for k in METRIC_KEYS)
        print(line)

    # best per metric
    print("\n" + "-" * len(header))
    print("BEST PER METRIC (best possible across all models):")
    print("-" * len(header))
    for k in METRIC_KEYS:
        best = max(rows, key=lambda r: r[k])
        print(f"  {k:<10}  {best[k]:.4f}   <-  {best['model']}")

    # overall best by average of the six metrics
    def avg(r):
        return sum(r[k] for k in METRIC_KEYS) / len(METRIC_KEYS)
    overall = max(rows, key=avg)
    print("\n" + "-" * len(header))
    print(f"OVERALL BEST MODEL (mean of 6 metrics): {overall['model']}  "
          f"(avg={avg(overall):.4f})")
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/localssd/z_followupq/match_results.jsonl")
    ap.add_argument("--models-root",
                    default="/mnt/localssd/automation/internship-causal-embedding/pythia_train")
    ap.add_argument("--out-dir", default="/mnt/localssd/z_followupq/sweep_ppl")
    ap.add_argument("--include-checkpoints", action="store_true",
                    help="Also evaluate intermediate checkpoint-* subdirs.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Explicit list of model dirs to evaluate (overrides discovery).")
    ap.add_argument("--sort-metric", default="hit@1", choices=METRIC_KEYS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write-reranked", action="store_true", default=True)
    ap.add_argument("--no-write-reranked", dest="write_reranked", action="store_false")
    args = ap.parse_args()

    device = torch.device(args.device)
    root = Path(args.models_root)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.models:
        models = [Path(m) for m in args.models]
    else:
        models = discover_models(root, args.include_checkpoints)

    print(f"Device: {device}")
    print(f"Models root: {root}")
    print(f"Discovered {len(models)} model(s):")
    for m in models:
        print(f"  - {m.relative_to(root) if str(m).startswith(str(root)) else m}")

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    n_total = len(records)
    n_eval = sum(1 for r in records
                 if any(fp.get("clicked", 0) == 1 for fp in r["generated_follow_ups"]))
    print(f"\nRecords: {n_total} total, {n_eval} with ≥1 clicked follow-up (eval set)\n")

    rows = []
    for m in models:
        print(f"[eval] {m}")
        try:
            row = eval_one_model(m, records, args, device, root)
            rows.append(row)
            print(f"       hit@1={row['hit@1']:.4f}  hit@3={row['hit@3']:.4f}  "
                  f"hit@5={row['hit@5']:.4f}  recall@1={row['recall@1']:.4f}  "
                  f"recall@3={row['recall@3']:.4f}  recall@5={row['recall@5']:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"       !! FAILED: {type(e).__name__}: {e}")

    if not rows:
        print("No models evaluated successfully.")
        return

    print_leaderboard(rows, args.sort_metric)

    # persist summary
    summary_json = Path(args.out_dir) / "sweep_summary.json"
    summary_csv = Path(args.out_dir) / "sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2))
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary written to:\n  {summary_json}\n  {summary_csv}")


if __name__ == "__main__":
    main()
