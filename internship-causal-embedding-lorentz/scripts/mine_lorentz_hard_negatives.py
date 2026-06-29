"""Mine semantic hard negatives for Lorentz encoder training.

Same approach as finetune_eval/mine_hard_negatives.py, plus anchor-similarity filter.
Now supports any HF model as the mining encoder (e5-base-v2, BGE-M3, etc.)
and text prefix convention for query/passage models.

Usage:
  # Default: mine with all-MiniLM (fast)
  python scripts/mine_lorentz_hard_negatives.py --dataset aep_causal

  # Mine with e5-base-v2 (better quality negatives, matches e5 training)
  python scripts/mine_lorentz_hard_negatives.py --dataset all \
      --miner-model intfloat/e5-base-v2 \
      --miner-anchor-prefix "query: " --miner-passage-prefix "passage: " \
      --miner-pooling mean --out-dir lorentz_enc_results_e5

  # Full qrecc (no cap)
  python scripts/mine_lorentz_hard_negatives.py --dataset qrecc --no-cap
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finetune_eval.datasets import SPLITS, TRAIN_CAPS, list_datasets, split_path  # noqa
from finetune_eval.data import load_pairs                                            # noqa


def mine_for_dataset(
    pairs, out_dir, split, model_name, anchor_prefix, passage_prefix,
    pooling, batch_size, max_seq_length, k, anchor_sim_threshold, chunk_size,
):
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    print(f"  Loading miner: {model_name}  pooling={pooling}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc_model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_model = enc_model.to(device).eval()

    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    N = len(pairs)

    def encode(texts, prefix):
        if prefix:
            texts = [prefix + t for t in texts]
        all_emb = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                enc = tokenizer(chunk, padding=True, truncation=True,
                                max_length=max_seq_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = enc_model(**enc)
                if pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    emb = out.last_hidden_state[:, 0]
                emb = F.normalize(emb, dim=-1)
                all_emb.append(emb.float().cpu().numpy())
        return np.concatenate(all_emb, axis=0)

    print(f"  Encoding {N} positives (passage: {passage_prefix!r})…")
    t0 = time.time()
    pos_emb = encode(positives, passage_prefix)
    print(f"  Encoded in {time.time()-t0:.1f}s  dim={pos_emb.shape[1]}")

    print(f"  Encoding {N} anchors (anchor: {anchor_prefix!r})…")
    t1 = time.time()
    anc_emb = encode(anchors, anchor_prefix)
    print(f"  Encoded in {time.time()-t1:.1f}s")

    pos_t = torch.from_numpy(pos_emb).to(device)
    anc_t = torch.from_numpy(anc_emb).to(device)

    top_k_extra = min(k * 10 + 50, N - 1)
    hard_idx = np.full((N, k), -1, dtype=np.int32)

    print(f"  kNN + anchor-sim filter (threshold={anchor_sim_threshold})…")
    t2 = time.time()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_sz = end - start

        pos_sims = pos_t[start:end] @ pos_t.T
        rows = torch.arange(start, end, device=device)
        pos_sims[torch.arange(chunk_sz, device=device), rows] = float("-inf")

        anc_sims = anc_t[start:end] @ anc_t.T

        _, top_candidates = pos_sims.topk(top_k_extra, dim=1, largest=True, sorted=True)

        for bi in range(chunk_sz):
            gi = start + bi
            kept = []
            for cand_j in top_candidates[bi].cpu().numpy():
                if cand_j == gi:
                    continue
                asim = float(anc_sims[bi, cand_j].item())
                if asim > anchor_sim_threshold:
                    continue
                kept.append(cand_j)
                if len(kept) == k:
                    break
            for ki, idx in enumerate(kept):
                hard_idx[gi, ki] = idx

    print(f"  kNN done in {time.time()-t2:.1f}s  "
          f"full-coverage={100*(hard_idx!=-1).all(1).mean():.1f}%")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "train" else f"{split}_"
    np.save(out_dir / f"{suffix}hard_negatives.npy", hard_idx)
    with open(out_dir / f"{suffix}pairs.jsonl", "w") as fh:
        for a, p in pairs:
            fh.write(json.dumps({"anchor": a, "positive": p}) + "\n")
    print(f"  saved {out_dir / f'{suffix}hard_negatives.npy'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal", help="Dataset name or 'all'.")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--anchor-sim-threshold", type=float, default=0.85)
    ap.add_argument("--miner-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--miner-anchor-prefix", default="")
    ap.add_argument("--miner-passage-prefix", default="")
    ap.add_argument("--miner-pooling", default="mean")
    ap.add_argument("--miner-max-seq-length", type=int, default=256)
    ap.add_argument("--miner-batch-size", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "lorentz_enc_results"))
    args = ap.parse_args()

    targets = list_datasets() if args.dataset == "all" else [args.dataset]
    out_root = Path(args.out_dir)

    for ds in targets:
        if ds not in SPLITS:
            print(f"skip unknown: {ds}")
            continue
        src_path = split_path(ds, args.split)
        if src_path is None or not src_path.exists():
            # Try alternate location (merge-followup project)
            alt_root = Path("/mnt/localssd/internship-causal-embedding-merge-followup")
            alt_path = alt_root / "data_6" / ds / SPLITS[ds][args.split]
            if alt_path and alt_path.exists():
                src_path = alt_path
            else:
                print(f"[{ds}] skip: no {args.split} file")
                continue

        cap = None if (args.no_cap or args.split != "train") else TRAIN_CAPS.get(ds)
        print(f"\n[{ds} · {args.split}] loading  (cap={cap})")
        pairs = load_pairs(src_path, cap=cap, seed=args.seed)
        print(f"  {len(pairs)} pairs")

        ds_dir = out_root / ds
        try:
            mine_for_dataset(
                pairs=pairs,
                out_dir=ds_dir,
                split=args.split,
                model_name=args.miner_model,
                anchor_prefix=args.miner_anchor_prefix,
                passage_prefix=args.miner_passage_prefix,
                pooling=args.miner_pooling,
                batch_size=args.miner_batch_size,
                max_seq_length=args.miner_max_seq_length,
                k=args.k,
                anchor_sim_threshold=args.anchor_sim_threshold,
                chunk_size=args.chunk_size,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n[done]")


if __name__ == "__main__":
    main()
