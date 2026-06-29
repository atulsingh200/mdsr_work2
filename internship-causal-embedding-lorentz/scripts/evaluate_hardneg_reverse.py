"""AUC / P@1 evaluation with SEMANTIC HARD NEGATIVES + the REVERSE PAIR.

For every test pair (a_i, b_i) the candidate set scored against the gold is:

    gold (label 1):     a_i -> b_i        forward causal pair
    hard negs (label 0): a_i -> b_j        K semantic-nearest other positives
    reverse  (label 0):  b_i -> a_i        the SAME pair, direction flipped

The reverse pair is the key addition: a model that only does topical similarity
will score b_i -> a_i almost identically to a_i -> b_i (cosine is direction-blind
for tied reps; only mildly asymmetric for untied towers), so the reverse pair is a
very hard negative.  A model with a working causal/time signal should rank the
forward pair above the reverse.

Scoring uses the model's own scoring function:
    s(x -> y) = cos(space_anchor(x), space_pos(y))
                + beta * mean_k sigma(tau * (t_pos(y) - t_anchor(x)))     [if use_time]

Embeddings and mined hard-negative indices are cached to disk (per checkpoint)
so reruns are cheap and CPU memory stays bounded (we never hold an N x N matrix;
kNN is chunked on GPU).

Usage:
  python scripts/evaluate_hardneg_reverse.py \
      --results-dir lorentz_enc_results_e5_vanilla --tag vanilla --dataset all
  python scripts/evaluate_hardneg_reverse.py \
      --results-dir lorentz_enc_results_e5 --tag lorentz --dataset all
  python scripts/evaluate_hardneg_reverse.py \
      --results-dir lorentz_enc_results_e5_time --tag lorentz_time --dataset aep_causal followupqg
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer
from finetune_eval.data import load_pairs
from finetune_eval.datasets import split_path
from lorentz_enc.model.encoder import LorentzEncoder

WORKFLOW_TEST = Path("/mnt/localssd/internship-causal-embedding-merge-followup") / \
                "data_6" / "workflow" / "test_pairs.jsonl"


def check_mem(label=""):
    import psutil
    avail = psutil.virtual_memory().available / 1024**3
    print(f"  [mem {label}] avail={avail:.1f}GB")
    if avail < 4.0:
        raise MemoryError(f"OOM risk: {avail:.1f}GB available")


@torch.no_grad()
def encode_tower(model, tokenizer, texts, which, prefix, max_length, device, batch_size=128):
    """Encode texts through anchor or positive tower → (space[N,Ds], time[N,Dt]) float32."""
    model.eval()
    S, T = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        if prefix:
            chunk = [prefix + t for t in chunk]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        if which == "anchor":
            s, t = model.encode_anchor(ids, mask)
        else:
            s, t = model.encode_positive(ids, mask)
        S.append(s.float().cpu().numpy())
        T.append(t.float().cpu().numpy())
    return np.concatenate(S), np.concatenate(T)


@torch.no_grad()
def mine_hard_negs(pos_space, k, device, chunk=512):
    """kNN nearest OTHER positives by cosine over the model's own positive space.

    Returns (N, k) int32 indices. Chunked on GPU — never materialises N×N on CPU.
    """
    N = pos_space.shape[0]
    P = torch.from_numpy(pos_space).to(device)
    out = np.empty((N, k), dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sims = P[start:end] @ P.T                 # (b, N)
        rows = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), rows] = float("-inf")  # mask self
        _, top = sims.topk(k, dim=1, largest=True)
        out[start:end] = top.cpu().numpy()
    del P
    torch.cuda.empty_cache()
    return out


def time_term(t_query, t_cand, tau):
    """mean_k sigmoid(tau * (t_cand - t_query)); supports broadcasting."""
    return (1.0 / (1.0 + np.exp(-tau * (t_cand - t_query)))).mean(-1)


def evaluate(results_dir, dataset, tag, k_hard, batch_size, seed, cache_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(results_dir) / dataset / "lorentz_best.pt"
    if not ckpt_path.exists():
        return {"dataset": dataset, "error": "no checkpoint"}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    ap, pp = cfg.get("anchor_prefix", ""), cfg.get("positive_prefix", "")
    use_time = cfg.get("use_time", False)
    beta = cfg.get("beta", 0.0) if use_time else 0.0
    tau = cfg.get("tau_score", 10.0)
    max_length = cfg.get("max_length", 256)

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    model = LorentzEncoder(
        backbone_name=cfg["backbone"], space_dim=cfg["space_dim"], time_dim=cfg["time_dim"],
        space_hidden=cfg["space_hidden"], time_hidden=cfg["time_hidden"],
        time_layers=cfg.get("time_layers", 2), pooling=cfg["pooling"], tied=False,
    ).to(device)
    # strict=False: older space-only checkpoints predate the deeper TimeHead module;
    # their (unused, beta=0) time head simply stays at init. Space heads always match.
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        # Only tolerate time-head mismatches; a space-head mismatch is a real error.
        bad = [k for k in (list(missing) + list(unexpected)) if "time_head" not in k]
        if bad:
            raise RuntimeError(f"non-time-head state_dict mismatch: {bad[:5]}")
        if beta > 0:
            raise RuntimeError("time head failed to load but beta>0 — would score with random time head")
    model.eval()

    test_path = WORKFLOW_TEST if dataset == "workflow" else split_path(dataset, "test")
    pairs = load_pairs(test_path, cap=None)
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    N = len(pairs)

    # ---- Encode all four tower×text combinations (cache to disk) ----
    cache = Path(cache_dir) / f"{tag}_{dataset}"
    cache.mkdir(parents=True, exist_ok=True)
    arrs = {}
    specs = [("A_anc", anchors, "anchor", ap), ("A_pos", anchors, "positive", pp),
             ("B_anc", positives, "anchor", ap), ("B_pos", positives, "positive", pp)]
    for name, texts, tower, prefix in specs:
        sp = cache / f"{name}_space.npy"; tp = cache / f"{name}_time.npy"
        if sp.exists() and tp.exists():
            arrs[name + "_s"] = np.load(sp); arrs[name + "_t"] = np.load(tp)
        else:
            s, t = encode_tower(model, tokenizer, texts, tower, prefix, max_length, device, batch_size)
            np.save(sp, s); np.save(tp, t)
            arrs[name + "_s"] = s; arrs[name + "_t"] = t
    check_mem(f"{tag}/{dataset} encoded")

    # ---- Mine K hard negatives among test positives (model's own space) ----
    hn_path = cache / f"hardneg_k{k_hard}.npy"
    if hn_path.exists():
        hard_idx = np.load(hn_path)
    else:
        hard_idx = mine_hard_negs(arrs["B_pos_s"], k_hard, device)
        np.save(hn_path, hard_idx)

    # free GPU model — rest is numpy
    del model
    gc.collect(); torch.cuda.empty_cache()

    A_anc_s, A_anc_t = arrs["A_anc_s"], arrs["A_anc_t"]
    A_pos_s, A_pos_t = arrs["A_pos_s"], arrs["A_pos_t"]
    B_anc_s, B_anc_t = arrs["B_anc_s"], arrs["B_anc_t"]
    B_pos_s, B_pos_t = arrs["B_pos_s"], arrs["B_pos_t"]

    # Pre-compute space-cosine and time terms separately so we can sweep beta cheaply.
    def cos(q_s, c_s):
        return (q_s * c_s).sum(-1)

    cos_gold = cos(A_anc_s, B_pos_s)
    cos_rev = cos(B_anc_s, A_pos_s)
    cos_hard = np.stack([cos(A_anc_s, B_pos_s[hard_idx[:, k]]) for k in range(k_hard)], axis=1)

    if use_time:
        tt_gold = time_term(A_anc_t, B_pos_t, tau)
        tt_rev = time_term(B_anc_t, A_pos_t, tau)
        tt_hard = np.stack([time_term(A_anc_t, B_pos_t[hard_idx[:, k]], tau)
                            for k in range(k_hard)], axis=1)
    else:
        tt_gold = tt_rev = tt_hard = None

    # beta sweep: 0 = space only; cfg beta; plus a couple higher to test time signal
    betas = [0.0]
    if use_time:
        betas = sorted(set([0.0, beta, 0.5, 1.0, 2.0]))

    def metrics_at(b):
        gold = cos_gold.copy()
        rev = cos_rev.copy()
        hard = cos_hard.copy()
        if b > 0 and tt_gold is not None:
            gold = gold + b * tt_gold
            rev = rev + b * tt_rev
            hard = hard + b * tt_hard
        neg_max = np.maximum(hard.max(axis=1), rev)
        scores = np.concatenate([gold, hard.ravel(), rev])
        labels = np.concatenate([np.ones(N), np.zeros(N * k_hard), np.zeros(N)])
        return {
            "beta": b,
            "auc_hard_plus_reverse": float(roc_auc_score(labels, scores)),
            "p_at_1_hard_plus_reverse": float((gold > neg_max).mean()),
            "p_at_1_hard_only": float((gold > hard.max(axis=1)).mean()),
            "directional_acc_fwd_gt_rev": float((gold > rev).mean()),
        }

    sweep = [metrics_at(b) for b in betas]
    main = next((m for m in sweep if m["beta"] == beta), sweep[0])

    return {
        "dataset": dataset, "tag": tag, "n_test_pairs": N,
        "k_hard": k_hard, "use_time": use_time, "beta": beta,
        "auc_hard_plus_reverse": main["auc_hard_plus_reverse"],
        "p_at_1_hard_plus_reverse": main["p_at_1_hard_plus_reverse"],
        "p_at_1_hard_only": main["p_at_1_hard_only"],
        "directional_acc_fwd_gt_rev": main["directional_acc_fwd_gt_rev"],
        "beta_sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tag", required=True, help="Label for this model (vanilla/lorentz/lorentz_time).")
    ap.add_argument("--dataset", nargs="+", default=["all"])
    ap.add_argument("--k-hard", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(ROOT / "hardneg_reverse_cache"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from finetune_eval.datasets import list_datasets
    targets = list_datasets() if args.dataset == ["all"] else args.dataset

    rows = []
    for ds in targets:
        print(f"\n[{args.tag} · {ds}] evaluating…")
        try:
            r = evaluate(args.results_dir, ds, args.tag, args.k_hard,
                         args.batch_size, args.seed, args.cache_dir)
            rows.append(r)
            if "error" not in r:
                print(f"  AUC(hard+rev)={r['auc_hard_plus_reverse']:.4f}  "
                      f"P@1(hard+rev)={r['p_at_1_hard_plus_reverse']:.4f}  "
                      f"P@1(hard-only)={r['p_at_1_hard_only']:.4f}  "
                      f"dir_acc={r['directional_acc_fwd_gt_rev']:.4f}  "
                      f"[use_time={r['use_time']} beta={r['beta']}]")
            else:
                print(f"  SKIP: {r['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            rows.append({"dataset": ds, "tag": args.tag, "error": str(e)})

    out = Path(args.out) if args.out else Path(args.results_dir) / "hardneg_reverse_eval.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
