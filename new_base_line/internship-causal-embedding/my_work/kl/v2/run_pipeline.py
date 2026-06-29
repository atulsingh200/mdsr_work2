"""End-to-end driver: run the SOTA `CDEv2 + CE-rerank` pipeline on ANY dataset.

This is the exact recipe that beat the BiEncoder by +12.9% MRR on aep_causal
(see v2/RESULTS.md), generalised to any dataset that has a trained BiEncoder
baseline under finetune_eval/results/<dataset>/.

Steps (all self-contained, single command):
  0. Stage data into results/<dataset>/ (train/val/test pairs + hard negs).
     If the dataset has no val split, hold out 10% of train as val.
  1. Mine K=8 MiniLM hard negatives (train + test).
  2. Train the warm-started CDEv2 retriever (run5c config).
  3. Train one BERT cross-encoder reranker on the mined negatives.
  4. rerank_eval on VAL to pick (pool, blend); then report on TEST.

Usage:
  python v2/run_pipeline.py --dataset followupqg
  python v2/run_pipeline.py --dataset multiwoz_v24 --gpu 0

Each step shells out to the existing (now --dataset-aware) scripts so behaviour
is identical to the hand-run aep_causal pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
PROJECT_ROOT = KL_DIR.parent.parent
PY = sys.executable


def run(cmd, env=None):
    print(f"\n\033[1;36m$ {' '.join(str(c) for c in cmd)}\033[0m", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(KL_DIR), env=env)


def stage_data(dataset: str, seed: int):
    """Copy pairs + hard negatives into results/<dataset>/, creating a val split
    (10% of train) when the dataset ships none. Returns the results dir."""
    src = PROJECT_ROOT / "finetune_eval" / "results" / dataset
    dst = KL_DIR / "results" / dataset
    dst.mkdir(parents=True, exist_ok=True)
    if not (src / "train_pairs.jsonl").exists():
        raise SystemExit(f"[fatal] no baseline data at {src} — train a BiEncoder first.")

    for fn in ("train_pairs.jsonl", "val_pairs.jsonl", "test_pairs.jsonl",
               "hard_negatives.npy", "val_hard_negatives.npy", "test_hard_negatives.npy"):
        s = src / fn
        if s.exists():
            shutil.copy2(s, dst / fn)

    # If no val split, hold out 10% of train (lines) into val_pairs.jsonl.
    if not (dst / "val_pairs.jsonl").exists():
        lines = (dst / "train_pairs.jsonl").read_text().splitlines()
        rng = random.Random(seed)
        idx = list(range(len(lines)))
        rng.shuffle(idx)
        n_val = max(1, int(len(lines) * 0.10))
        val_idx = set(idx[:n_val])
        val = [lines[i] for i in sorted(val_idx)]
        tr = [lines[i] for i in range(len(lines)) if i not in val_idx]
        (dst / "val_pairs.jsonl").write_text("\n".join(val) + "\n")
        (dst / "train_pairs.jsonl").write_text("\n".join(tr) + "\n")
        print(f"[stage] {dataset}: no val split shipped — held out {len(val)} "
              f"of {len(lines)} train pairs as val.")
    print(f"[stage] data ready at {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="Dataset under finetune_eval/results/ (followupqg, multiwoz_v24, qrecc, workflow, aep_causal).")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES to pin (e.g. 0 or 0,1).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=3000)
    ap.add_argument("--ce-epochs", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="hard negatives K for CDEv2.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-mine", action="store_true",
                    help="Reuse shipped K=4 hard_negatives.npy instead of mining K=8.")
    args = ap.parse_args()

    ds = args.dataset
    suffix = "_run5c"
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"\n{'='*70}\n  CDEv2 + CE-rerank pipeline  —  dataset={ds}\n{'='*70}")

    # 0. stage data
    stage_data(ds, args.seed)

    biencoder_ckpt = PROJECT_ROOT / "finetune_eval" / "results" / ds / "checkpoint_best.pt"
    if not biencoder_ckpt.exists():
        raise SystemExit(f"[fatal] missing BiEncoder warm-start ckpt {biencoder_ckpt}")

    # 1. mine K=8 MiniLM hard negatives (train + test)
    if not args.skip_mine:
        run([PY, "v2/mine_hard_neg_v2.py", "--dataset", ds, "--split", "both", "--k", args.k], env)
    hard_neg_file = (KL_DIR / "results" / ds / f"hard_negatives_k{args.k}.npy")
    hn_arg = ["--hard-neg-file", str(hard_neg_file)] if hard_neg_file.exists() else []

    # 2. train warm-started CDEv2 retriever (run5c config: pure InfoNCE, CLS, mu-identity)
    run([PY, "v2/train_v2.py", "--dataset", ds,
         "--init-from-biencoder", str(biencoder_ckpt),
         "--proj-dim", 768, "--pooling", "cls", "--mu-identity",
         "--log-sigma-init", 3.0, "--kl-scale", "dim", "--init-cos-weight", 1.0,
         "--backbone-lr", 2e-5, "--batch-size", args.batch_size,
         "--max-seq-length", args.max_seq_length,
         "--phase-a-steps", args.total_steps, "--phase-b-steps", 0,
         "--total-steps", args.total_steps,
         "--lambda-entropy", 0, "--lambda-antisym", 0, "--lambda-bpr", 0,
         "--out-suffix", suffix, *hn_arg], env)

    # 3. train the cross-encoder reranker (MiniLM hard negatives shipped with baseline)
    ce_dir = KL_DIR / "cross_encoder" / "results" / f"{ds}_rerank"
    run([PY, "cross_encoder/train_ce.py", "--dataset", ds,
         "--data-dir", str(PROJECT_ROOT / "finetune_eval" / "results"),
         "--epochs", args.ce_epochs, "--out-dir", str(ce_dir)], env)

    # 4a. tune (pool, blend) on VAL
    run([PY, "v2/rerank_eval.py", "--dataset", ds, "--split", "val",
         "--cde-suffix", suffix, "--ce-dir", str(ce_dir),
         "--pool", 20, 50, "--blend", 0.3, 0.5, 0.7,
         "--batch-size", args.batch_size, "--max-length", args.max_seq_length], env)

    # 4b. report on TEST with the val-selected config (full sweep; best printed)
    run([PY, "v2/rerank_eval.py", "--dataset", ds, "--split", "test",
         "--cde-suffix", suffix, "--ce-dir", str(ce_dir),
         "--pool", 20, 50, "--blend", 0.3, 0.5, 0.7,
         "--batch-size", args.batch_size, "--max-length", args.max_seq_length], env)

    res = json.loads((KL_DIR / "results" / ds / "rerank_eval.json").read_text())
    print(f"\n{'='*70}\n  DONE  {ds}\n{'='*70}")
    print(f"  stage-1 retriever MRR : {res['stage1']['mrr']:.4f}")
    print(f"  best fused            : MRR={res['best']['mrr']:.4f}  "
          f"({res['best'].get('tag','')})")
    if "auc_hardneg" in res["best"]:
        print(f"  fused AUC (hard-neg)  : {res['best']['auc_hardneg']:.4f}")
    lift = (res["best"]["mrr"] - res["stage1"]["mrr"]) / res["stage1"]["mrr"] * 100
    print(f"  rerank lift over stage-1: {lift:+.1f}%")


if __name__ == "__main__":
    main()
