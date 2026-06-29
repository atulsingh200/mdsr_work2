"""Pure cross-encoder eval on aep_causal — AUC + P@1, no retriever.

NO CDEv2, NO BiEncoder. We take the two pipeline cross-encoders, pass each
(anchor, candidate) pair through them directly, and measure how well they
separate the true positive from negatives. Reports THREE systems — CE #1 alone,
CE #2 alone, and their ensemble — so each model's contribution is visible.

  CE #1  cross_encoder/results/aep_causal_rerank      (trained on MiniLM semantic negs)
  CE #2  cross_encoder/results/aep_causal_rerank_v2   (trained on in-domain CDEv2 negs)
  ENS    per-anchor z-score each CE's logits, then average the two.

Two metric groups (identical setup to cross_encoder/evaluate_ce.py):
  1. vs 4 RANDOM negatives per anchor          (rng seed 0)
  2. vs 4 SEMANTIC HARD negatives per anchor   (pre-mined all-MiniLM-L6-v2 kNN,
                                                test_hard_negatives.npy)
For each: AUC (positive vs negatives) and P@1 (positive logit > all 4 neg logits).

Usage (from my_work/kl):
  python v2/eval_ce_ensemble.py                  # test split
  python v2/eval_ce_ensemble.py --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
sys.path.insert(0, str(KL_DIR / "cross_encoder"))

from model_ce import CrossEncoder, get_tokenizer as ce_tok_fn   # noqa: E402
from dataset_ce import load_pairs                               # noqa: E402

# ── The two pipeline cross-encoders ─────────────────────────────────────────
DEFAULT_CE_DIRS = [
    KL_DIR / "cross_encoder" / "results" / "aep_causal_rerank",
    KL_DIR / "cross_encoder" / "results" / "aep_causal_rerank_v2",
]
DEFAULT_CE_LABELS = ["CE#1 (semantic-negs)", "CE#2 (in-domain-negs)"]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


@torch.no_grad()
def ce_score_pairs(model, tok, text_a, text_b, max_length, batch_size, device):
    out = []
    for s in range(0, len(text_a), batch_size):
        enc = tok(text_a[s:s + batch_size], text_b[s:s + batch_size],
                  padding=True, truncation="longest_first", max_length=max_length,
                  return_tensors="pt", return_token_type_ids=True)
        ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask, tti)
        out.extend(logits.float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def load_ce(ce_dir, device):
    ckpt = torch.load(ce_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config", {"backbone": "google-bert/bert-base-uncased"}))
    backbone = cfg.get("backbone", "google-bert/bert-base-uncased")
    tok = ce_tok_fn(backbone)
    model = CrossEncoder(backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tok, backbone


def auc_p1(pos: np.ndarray, neg: np.ndarray) -> dict:
    """pos: (N,) positive logits. neg: (N, K) negative logits."""
    p_at_1 = float(np.mean(pos > neg.max(axis=1)))
    scores = np.concatenate([pos, neg.ravel()])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(neg.size)])
    return {"auc": float(roc_auc_score(labels, scores)), "p_at_1": p_at_1,
            "n_anchors": int(len(pos)), "n_neg_per_anchor": int(neg.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aep_causal")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--ce-dir", nargs="+", default=[str(d) for d in DEFAULT_CE_DIRS],
                    help="Cross-encoder dirs. Each evaluated alone + ensembled.")
    ap.add_argument("--n-negatives", type=int, default=4,
                    help="Random negatives per anchor (hard negs use the mined K).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for random negatives.")
    args = ap.parse_args()

    out_dir = KL_DIR / "results" / args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_pairs(out_dir / f"{args.split}_pairs.jsonl")
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    N = len(pairs)

    hn_path = out_dir / ("test_hard_negatives.npy" if args.split == "test"
                         else "val_hard_negatives.npy")
    test_hn = np.load(hn_path)        # (N, K) MiniLM-L6-v2 semantic hard negatives
    K_hn = test_hn.shape[1]

    ce_dirs = [Path(d) if Path(d).is_absolute() else (KL_DIR / d) for d in args.ce_dir]
    labels = (DEFAULT_CE_LABELS if [str(d) for d in ce_dirs] == [str(d) for d in DEFAULT_CE_DIRS]
              else [f"CE#{i+1} ({d.name})" for i, d in enumerate(ce_dirs)])
    print(f"[eval] dataset={args.dataset} split={args.split} N={N} device={device}")
    print(f"[eval] semantic hard negs: {hn_path.name}  shape={test_hn.shape}  "
          f"(miner=all-MiniLM-L6-v2)")
    for lab, d in zip(labels, ce_dirs):
        print(f"[eval] {lab}  ←  {d.relative_to(KL_DIR) if d.is_relative_to(KL_DIR) else d}")

    # ── Build the candidate index sets (shared across all CEs) ──
    # Random negatives: K per anchor, seeded, excluding self.
    rng = np.random.default_rng(args.seed)
    rand_neg_idx = np.empty((N, args.n_negatives), dtype=np.int64)
    for i in range(N):
        picks = rng.choice(N - 1, size=args.n_negatives, replace=False)
        picks[picks >= i] += 1
        rand_neg_idx[i] = picks

    # ── Score every (anchor, candidate) pair once per CE ──
    # We need: positive logits (N,), random-neg logits (N, n_neg), hard-neg logits (N, K_hn).
    def flatten(neg_idx):
        fa, fb = [], []
        for i in range(N):
            for j in neg_idx[i]:
                fa.append(anchors[i]); fb.append(positives[int(j)])
        return fa, fb

    rand_a, rand_b = flatten(rand_neg_idx)
    hn_a, hn_b = flatten(test_hn)

    pos_logits_all, rand_logits_all, hn_logits_all = [], [], []
    for lab, d in zip(labels, ce_dirs):
        model, tok, backbone = load_ce(d, device)
        print(f"\n[score] {lab} ({backbone}) …")
        pos = ce_score_pairs(model, tok, anchors, positives,
                             args.max_length, args.batch_size, device)            # (N,)
        rnd = ce_score_pairs(model, tok, rand_a, rand_b,
                             args.max_length, args.batch_size, device).reshape(N, args.n_negatives)
        hn = ce_score_pairs(model, tok, hn_a, hn_b,
                            args.max_length, args.batch_size, device).reshape(N, K_hn)
        pos_logits_all.append(pos)
        rand_logits_all.append(rnd)
        hn_logits_all.append(hn)
        del model; torch.cuda.empty_cache()

    # ── Ensemble: per-anchor z-score over each anchor's own candidate set, then mean ──
    # For a fair z-score, normalise each CE's logits over [positive | its negatives]
    # per anchor, average across CEs, then split back out.
    def ensemble(pos_list, neg_list):
        """pos_list: list of (N,), neg_list: list of (N, K). Returns (pos_ens, neg_ens)."""
        Nce = len(pos_list)
        Kc = neg_list[0].shape[1]
        pos_e = np.zeros(N); neg_e = np.zeros((N, Kc))
        for i in range(N):
            zsum_pos = 0.0; zsum_neg = np.zeros(Kc)
            for c in range(Nce):
                row = np.concatenate([[pos_list[c][i]], neg_list[c][i]])   # (1+K,)
                z = _zscore(row)
                zsum_pos += z[0]; zsum_neg += z[1:]
            pos_e[i] = zsum_pos / Nce; neg_e[i] = zsum_neg / Nce
        return pos_e, neg_e

    systems = []
    for c, lab in enumerate(labels):
        systems.append((lab, pos_logits_all[c], rand_logits_all[c], hn_logits_all[c]))
    pos_ens_r, rand_ens = ensemble(pos_logits_all, rand_logits_all)
    pos_ens_h, hn_ens = ensemble(pos_logits_all, hn_logits_all)
    # (positive z differs between the two neg sets since z is over the candidate set;
    #  that's expected — AUC/P@1 only compare within the same set.)
    systems.append(("ENSEMBLE (mean z)", (pos_ens_r, pos_ens_h), rand_ens, hn_ens))

    # ── Compute metrics ──
    rows = []
    for name, pos, rnd, hn in systems:
        pos_r, pos_h = pos if isinstance(pos, tuple) else (pos, pos)
        m_rand = auc_p1(pos_r, rnd)
        m_hard = auc_p1(pos_h, hn)
        rows.append((name, m_rand, m_hard))

    # ── Print summary ──
    print("\n" + "=" * 74)
    print(f"Cross-encoder direct eval — {args.dataset} {args.split} (N={N})")
    print("=" * 74)
    hdr = (f"{'System':<24}{'AUC(rand)':>11}{'P@1(rand)':>11}"
           f"{'AUC(hard)':>11}{'P@1(hard)':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, m_rand, m_hard in rows:
        print(f"{name:<24}{m_rand['auc']:>11.4f}{m_rand['p_at_1']:>11.4f}"
              f"{m_hard['auc']:>11.4f}{m_hard['p_at_1']:>11.4f}")

    # ── Save ──
    payload = {
        "dataset": args.dataset, "split": args.split, "n_pairs": N,
        "n_random_negatives": args.n_negatives, "n_hard_negatives": K_hn,
        "hard_neg_miner": "sentence-transformers/all-MiniLM-L6-v2",
        "hard_neg_file": str(hn_path), "seed": args.seed,
        "ce_dirs": [str(d) for d in ce_dirs], "ce_labels": labels,
        "systems": {
            name: {
                "random_negatives": m_rand,
                "semantic_hard_negatives": m_hard,
            } for name, m_rand, m_hard in rows
        },
    }
    out_path = out_dir / f"eval_ce_ensemble_aucp1_{args.split}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
