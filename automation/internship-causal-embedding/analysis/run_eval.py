#!/usr/bin/env python3
"""
Run inference on the test set, compute semantic similarity (all-MiniLM-L6-v2)
for every pair, and write analysis/predictions.jsonl.

Each output row:
  idx, text_1, text_2, label, pred, prob, correct,
  tier_1, tier_2, sub_1, sub_2, url_1, url_2, sem_sim

Usage (from repo root):
  python analysis/run_eval.py \
      --ckpt runs/classifier/best_mlp_bge_small/best.pt \
      --data-dir data/aep_causal_classification_34 \
      --out analysis/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifier.model import DirectionalClassifier
from src.classifier.training.data import Collate, PairDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pt / final.pt")
    ap.add_argument("--data-dir", required=True, help="Dir with directional_test.jsonl")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="analysis/predictions.jsonl")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--sim-batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--sim-model", default="all-MiniLM-L6-v2",
                    help="sentence-transformers model for semantic similarity")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load classifier checkpoint
    # ------------------------------------------------------------------ #
    print(f"[eval] loading checkpoint {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    base_model = cfg["base_model"]
    head_cfg = cfg["head_cfg"]
    n_tiers = cfg.get("n_tiers", 0) or (15 if cfg.get("tier_aux_weight", 0.0) > 0 else 0)
    max_seq_len = cfg["max_seq_len"]
    data_dir = Path(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    ds = PairDataset(data_dir / f"directional_{args.split}.jsonl")
    print(f"[eval] {args.split} examples: {len(ds)}", flush=True)

    collate = Collate(tokenizer, max_seq_len)
    dl = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    model = DirectionalClassifier(base_model, head_cfg=head_cfg, n_tiers=n_tiers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16
        else nullcontext
    )

    # ------------------------------------------------------------------ #
    # 2. Run classifier inference — collect probs + row metadata
    # ------------------------------------------------------------------ #
    print("[eval] running classifier inference ...", flush=True)
    loss_fn = nn.BCEWithLogitsLoss()
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_meta: list[dict] = []

    with torch.no_grad():
        for batch in dl:
            enc1 = {k: v.to(device, non_blocking=True) for k, v in batch["enc1"].items()}
            enc2 = {k: v.to(device, non_blocking=True) for k, v in batch["enc2"].items()}
            labels = batch["labels"].to(device, non_blocking=True)
            with ctx_factory():
                dir_logits, _, _ = model(enc1, enc2)
            probs = torch.sigmoid(dir_logits).float().cpu().numpy()
            lbls = labels.cpu().numpy()
            for j, i_ds in enumerate(batch["idx"]):
                r = ds.rows[i_ds]
                all_probs.append(float(probs[j]))
                all_labels.append(int(lbls[j]))
                all_meta.append({
                    "idx": int(i_ds),
                    "text_1": r["text_1"],
                    "text_2": r["text_2"],
                    "tier_1": r["tier_1"],
                    "tier_2": r["tier_2"],
                    "sub_1": r.get("sub_1", ""),
                    "sub_2": r.get("sub_2", ""),
                    "url_1": r.get("url_1", ""),
                    "url_2": r.get("url_2", ""),
                })

    print(f"[eval] inference done — {len(all_probs)} examples", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Compute semantic similarity with all-MiniLM-L6-v2
    # ------------------------------------------------------------------ #
    print(f"[eval] loading sentence-transformer: {args.sim_model}", flush=True)
    from sentence_transformers import SentenceTransformer
    sim_model = SentenceTransformer(args.sim_model, device=str(device))

    texts_1 = [m["text_1"] for m in all_meta]
    texts_2 = [m["text_2"] for m in all_meta]
    n = len(texts_1)

    print(f"[eval] encoding {n} text_1 ...", flush=True)
    emb1 = sim_model.encode(
        texts_1,
        batch_size=args.sim_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[eval] encoding {n} text_2 ...", flush=True)
    emb2 = sim_model.encode(
        texts_2,
        batch_size=args.sim_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # cosine similarity = dot product of L2-normalised embeddings (row-wise)
    sem_sims = (emb1 * emb2).sum(axis=1).tolist()
    print(f"[eval] semantic similarity: mean={np.mean(sem_sims):.4f}  "
          f"min={np.min(sem_sims):.4f}  max={np.max(sem_sims):.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Write predictions.jsonl
    # ------------------------------------------------------------------ #
    with open(out_path, "w") as f:
        for i, (meta, prob, label, sim) in enumerate(
            zip(all_meta, all_probs, all_labels, sem_sims)
        ):
            pred = int(prob > 0.5)
            row = {
                **meta,
                "label": label,
                "pred": pred,
                "prob": round(prob, 6),
                "correct": bool(pred == label),
                "sem_sim": round(float(sim), 6),
            }
            f.write(json.dumps(row) + "\n")

    n_correct = sum(p == l for p, l in zip([int(p > 0.5) for p in all_probs], all_labels))
    print(f"[eval] wrote {out_path}  accuracy={n_correct/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
