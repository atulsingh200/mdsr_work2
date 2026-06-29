"""Mine in-domain hard negatives for CE v2 training using the trained CDEv2 retriever.

For each training anchor, runs the CDEv2 retriever over the full positive corpus
and takes the top-K retrieved positives (excluding the true positive) as hard
negatives.  These are harder than MiniLM-kNN negatives because they are the
exact candidates that fool the trained retriever.

Output: cross_encoder/data_indomain/<dataset>/
    hard_negatives.npy         (N_train, K) — indices into train positives
    val_hard_negatives.npy     (N_val,   K) — indices into val   positives
    train_pairs.jsonl          copy of train pairs (for train_ce.py)
    val_pairs.jsonl            copy of val   pairs (for train_ce.py)

Usage (from project root):
  .venv/bin/python my_work/kl/v2/mine_indomain_negs.py --dataset aep_causal
  .venv/bin/python my_work/kl/v2/mine_indomain_negs.py --dataset followupqg
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

V2_DIR   = Path(__file__).resolve().parent
KL_DIR   = V2_DIR.parent
PROJECT_ROOT = KL_DIR.parent.parent
_SIBLING = PROJECT_ROOT.parent / "internship-causal-embedding"
BIENCODER_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "src" / "evaluation").exists() else _SIBLING
sys.path.insert(0, str(KL_DIR))
sys.path.insert(0, str(BIENCODER_ROOT))
sys.path.insert(0, str(BIENCODER_ROOT / "src"))

from dataset import get_tokenizer, load_pairs, tokenize_batch   # noqa: E402
from v2.model_v2 import CDEv2                                   # noqa: E402


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_causes(
    model: CDEv2,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode anchors (cause side) and apply transport. Returns mu (N, D)."""
    model.eval()
    mus = []
    for i in range(0, len(texts), batch_size):
        ids, mask = tokenize_batch(tokenizer, texts[i: i + batch_size], max_length)
        ids = ids.to(device); mask = mask.to(device)
        mu, log_sigma, delta = model.encode_cause(ids, mask)
        mu_t, _ = model.transport(mu, log_sigma, delta)
        mus.append(mu_t.cpu())
    return torch.cat(mus, dim=0)  # (N, D)


@torch.no_grad()
def encode_effects(
    model: CDEv2,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode positives (effect side). Returns mu (N, D)."""
    model.eval()
    mus = []
    for i in range(0, len(texts), batch_size):
        ids, mask = tokenize_batch(tokenizer, texts[i: i + batch_size], max_length)
        ids = ids.to(device); mask = mask.to(device)
        mu, _ = model.encode_effect(ids, mask)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0)  # (N, D)


# ---------------------------------------------------------------------------
# Core mining: for each anchor retrieve top-K positives != true positive
# ---------------------------------------------------------------------------

def mine_split(
    split_name: str,
    anchors: list[str],
    positives: list[str],
    model: CDEv2,
    tokenizer,
    max_length: int,
    batch_size: int,
    k: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return (N, K) int32 array of hard-negative indices into `positives`."""
    N = len(anchors)
    print(f"\n[{split_name}] encoding {N} anchors …")
    t0 = time.time()
    a_mu = encode_causes(model, tokenizer, anchors, max_length, batch_size, device)
    print(f"  anchors encoded in {time.time()-t0:.1f}s")

    print(f"[{split_name}] encoding {N} positives …")
    t1 = time.time()
    p_mu = encode_effects(model, tokenizer, positives, max_length, batch_size, device)
    print(f"  positives encoded in {time.time()-t1:.1f}s")

    # Normalize for cosine-like similarity (model scores also include KL but
    # mu cosine similarity is a good proxy for retrieval ranking order)
    a_mu = torch.nn.functional.normalize(a_mu, dim=-1)
    p_mu = torch.nn.functional.normalize(p_mu, dim=-1)

    a_t = a_mu.to(device)
    p_t = p_mu.to(device)

    hard_idx = np.empty((N, k), dtype=np.int32)
    print(f"[{split_name}] retrieving top-{k} hard negatives (chunk={chunk_size}) …")
    t2 = time.time()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # sim: (chunk, N) — anchor-query vs all positives
        sim = a_t[start:end] @ p_t.T                          # (chunk, N)
        # mask out the true positive (diagonal for same-indexed pairs)
        rows = torch.arange(end - start, device=device)
        sim[rows, torch.arange(start, end, device=device)] = float("-inf")
        _, top_idx = sim.topk(k, dim=1, largest=True, sorted=True)
        hard_idx[start:end] = top_idx.cpu().numpy()
    print(f"  retrieval done in {time.time()-t2:.1f}s")
    return hard_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",    required=True,
                    help="Dataset name, e.g. aep_causal, followupqg, multiwoz_v24, qrecc, workflow.")
    ap.add_argument("--cde-suffix", default="_run5c",
                    help="Suffix of the CDEv2 checkpoint to use (default: _run5c).")
    ap.add_argument("--k",          type=int, default=4,
                    help="Number of in-domain hard negatives per anchor (default: 4).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    ds      = args.dataset
    src_dir = KL_DIR / "results" / ds
    out_dir = KL_DIR / "cross_encoder" / "data_indomain" / ds
    ckpt_path = src_dir / f"cde_v2{args.cde_suffix}_best.pt"

    if not ckpt_path.exists():
        raise SystemExit(
            f"[fatal] CDEv2 checkpoint not found: {ckpt_path}\n"
            f"  Run train_v2.py --dataset {ds} --out-suffix {args.cde_suffix} first."
        )

    # ---- Load model ----
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(cfg["backbone"])
    model = CDEv2(
        cfg["backbone"],
        cfg["proj_dim"],
        shared_encoder=cfg.get("shared_encoder", False),
        kl_scale=cfg.get("kl_scale", "dim"),
        init_cos_weight=cfg.get("init_cos_weight", 1.0),
        pooling=cfg.get("pooling", "mean"),
        mu_identity=cfg.get("mu_identity", False),
        log_sigma_init=cfg.get("log_sigma_init", 0.0),
        score_type=cfg.get("score_type", "kl"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  backbone={cfg['backbone']}  proj_dim={cfg['proj_dim']}  device={device}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mine train split ----
    train_pairs_path = src_dir / "train_pairs.jsonl"
    if not train_pairs_path.exists():
        raise SystemExit(f"[fatal] train pairs not found: {train_pairs_path}")

    train_pairs = load_pairs(train_pairs_path, cap=None, seed=args.seed)
    train_anchors   = [a for a, _ in train_pairs]
    train_positives = [p for _, p in train_pairs]

    train_hn = mine_split(
        "train", train_anchors, train_positives,
        model, tokenizer, args.max_length, args.batch_size,
        args.k, args.chunk_size, device,
    )
    np.save(out_dir / "hard_negatives.npy", train_hn)
    shutil.copy2(train_pairs_path, out_dir / "train_pairs.jsonl")
    print(f"  saved {out_dir}/hard_negatives.npy  shape={train_hn.shape}")

    # ---- Mine val split ----
    val_pairs_path = src_dir / "val_pairs.jsonl"
    if val_pairs_path.exists():
        val_pairs = load_pairs(val_pairs_path, cap=None, seed=args.seed)
        val_anchors   = [a for a, _ in val_pairs]
        val_positives = [p for _, p in val_pairs]

        val_hn = mine_split(
            "val", val_anchors, val_positives,
            model, tokenizer, args.max_length, args.batch_size,
            args.k, args.chunk_size, device,
        )
        np.save(out_dir / "val_hard_negatives.npy", val_hn)
        shutil.copy2(val_pairs_path, out_dir / "val_pairs.jsonl")
        print(f"  saved {out_dir}/val_hard_negatives.npy  shape={val_hn.shape}")
    else:
        print(f"[val] no val_pairs.jsonl at {val_pairs_path} — skipping val mining")

    # ---- Write info ----
    info = {
        "dataset":     ds,
        "checkpoint":  str(ckpt_path),
        "backbone":    cfg["backbone"],
        "k":           args.k,
        "n_train":     len(train_pairs),
        "out_dir":     str(out_dir),
    }
    (out_dir / "mining_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n[done] in-domain negatives written to {out_dir}")


if __name__ == "__main__":
    main()
