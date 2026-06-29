#!/usr/bin/env python3
"""
Perplexity-based causal-ordering eval for the Pythia / GPT-2 causal LMs trained under
/mnt/localssd/automation/internship-causal-embedding/pythia_train.

Idea (per request): for a pair (A, B) we concatenate "A B" and compute the FULL-SEQUENCE
perplexity. The ordering with the LOWER perplexity wins. We turn the two opposing
orderings into a directed score P(i precedes j) = softmax over negative NLL, matching the
bi-encoder's P(text1 precedes text2) semantics so the existing tournament + net-precedence
ranking works unchanged.

Two eval sets:
  1. 30-sample 3-step set : /mnt/localssd/ajo_orchestrated_workflows_flat.json  (id,t1,t2,t3 ; gold t1->t2->t3)
  2. 6-step reorder set   : /mnt/localssd/test_samples_milan.json               (30 directed pairs over t1..t6 ; gold t1->..->t6)

Output: /mnt/localssd/pythia_perplexity_eval_results.json
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------- config
PYTHIA_ROOT = Path("/mnt/localssd/automation/internship-causal-embedding/pythia_train")
MODELS = {
    "pythia-410m-20ep": PYTHIA_ROOT / "output",
    "pythia-410m-40ep": PYTHIA_ROOT / "output_40ep",
    "gpt2-medium":      PYTHIA_ROOT / "output_gpt2",
    "gpt2-stable":      PYTHIA_ROOT / "output_gpt2_stable",
    "gpt2-smooth":      PYTHIA_ROOT / "output_gpt2_smooth",
}

THIRTY_PATH = Path("/mnt/localssd/ajo_orchestrated_workflows_flat.json")
SIXSTEP_PATH = Path("/mnt/localssd/test_samples_milan.json")
OUT_PATH = Path("/mnt/localssd/pythia_perplexity_eval_results.json")

MAX_LEN = 1024  # GPT-2 ctx; Pythia handles 2048 but concatenated pairs are far shorter


# ----------------------------------------------------------------------------- model loading
def resolve_model_dir(d: Path) -> Path:
    """Use the top-level dir if it has weights, else the latest checkpoint-* subdir."""
    if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
        return d
    ckpts = sorted(
        [p for p in d.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No weights or checkpoint-* dirs under {d}")
    return ckpts[-1]


def load_model(d: Path, device, dtype):
    path = resolve_model_dir(d)
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device)
    model.eval()
    return model, tok, str(path)


# ----------------------------------------------------------------------------- perplexity
@torch.no_grad()
def seq_nll(model, tok, text: str, device) -> float:
    """Mean per-token negative log-likelihood (cross-entropy) of the full sequence."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
    # HF returns mean token CE (loss) with internal label shift handled.
    return float(out.loss.item())


# ----------------------------------------------------------------------------- ranking on RAW perplexity (lower PPL => precedes)
def order_events(labels, PPL):
    """Rank events from a raw-perplexity matrix. PPL[i][j] = perplexity of "text_i text_j".

    Direction: i precedes j when PPL[i][j] < PPL[j][i] (the lower-perplexity ordering wins).
    Tournament wins first, then net perplexity margin as tiebreak.
    """
    wins = {a: 0 for a in labels}
    net = {a: 0.0 for a in labels}  # higher = tends to come earlier (lower fwd perplexity)
    for i in labels:
        for j in labels:
            if i == j:
                continue
            pij, pji = PPL[i].get(j), PPL[j].get(i)
            if pij is not None and pji is not None:
                net[i] += pji - pij  # positive when "i j" is cheaper than "j i"
    for i, j in itertools.combinations(labels, 2):
        pij, pji = PPL[i].get(j), PPL[j].get(i)
        pij = pij if pij is not None else math.inf
        pji = pji if pji is not None else math.inf
        if pij <= pji:
            wins[i] += 1
        else:
            wins[j] += 1
    order = sorted(labels, key=lambda a: (-wins[a], -net[a]))
    return order, wins, net


def kendall_tau(pred, gold):
    """Kendall tau over the gold ordinal positions implied by `pred`."""
    pos = {lab: i for i, lab in enumerate(gold)}
    seq = [pos[l] for l in pred if l in pos]
    n = len(seq)
    if n < 2:
        return 1.0
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                concord += 1
            else:
                discord += 1
    return (concord - discord) / (n * (n - 1) / 2)


# ----------------------------------------------------------------------------- caching scorer
def make_scorer(model, tok, device):
    cache = {}

    def concat_ppl(text_i, text_j):
        """Raw full-sequence perplexity of the concatenation 'text_i text_j'."""
        seq = text_i + " " + text_j
        if seq not in cache:
            cache[seq] = math.exp(seq_nll(model, tok, seq, device))
        return cache[seq]

    return concat_ppl


# ----------------------------------------------------------------------------- task 1: 30-sample
def eval_thirty(concat_ppl, data):
    labels = ["t1", "t2", "t3"]
    gold = ["t1", "t2", "t3"]
    per_sample = []
    n_correct = 0
    pair_hits = pair_total = 0
    for s in data:
        texts = {l: s[l] for l in labels}
        PPL = {a: {b: None for b in labels} for a in labels}
        for i, j in itertools.permutations(labels, 2):
            PPL[i][j] = concat_ppl(texts[i], texts[j])
        pred, _, _ = order_events(labels, PPL)
        ok = pred == gold
        n_correct += int(ok)
        for i, j in itertools.combinations(labels, 2):  # gold direction i<j: "i j" should be cheaper
            pair_total += 1
            if PPL[i][j] < PPL[j][i]:
                pair_hits += 1
        per_sample.append({
            "id": s.get("id"),
            "predicted_order": pred,
            "correct_order": gold,
            "order_correct": ok,
            "perplexity_matrix": {a: {b: (round(PPL[a][b], 4) if PPL[a][b] is not None else None)
                                      for b in labels} for a in labels},
        })
    return {
        "order_accuracy": round(n_correct / len(data), 4),
        "n_correct": n_correct,
        "n_total": len(data),
        "pairwise_direction_accuracy": round(pair_hits / pair_total, 4),
        "per_sample": per_sample,
    }


# ----------------------------------------------------------------------------- task 2: 6-step
def eval_sixstep(concat_ppl, rows):
    labels = sorted(
        set(r["label1"] for r in rows) | set(r["label2"] for r in rows),
        key=lambda l: (l[0], int(l[1:])) if l[1:].isdigit() else (l, 0),
    )
    gold = labels[:]  # t1..t6 natural order
    text_of = {}
    for r in rows:
        text_of[r["label1"]] = r["text1"]
        text_of[r["label2"]] = r["text2"]

    PPL = {a: {b: None for b in labels} for a in labels}
    for i, j in itertools.permutations(labels, 2):
        PPL[i][j] = concat_ppl(text_of[i], text_of[j])

    pred, wins, net = order_events(labels, PPL)
    pair_hits = pair_total = 0
    for i, j in itertools.combinations(labels, 2):  # gold i<j: "i j" should be cheaper
        pair_total += 1
        if PPL[i][j] < PPL[j][i]:
            pair_hits += 1
    return {
        "predicted_order": pred,
        "correct_order": gold,
        "exact_match": pred == gold,
        "pairwise_direction_accuracy": round(pair_hits / pair_total, 4),
        "kendall_tau": round(kendall_tau(pred, gold), 4),
        "perplexity_matrix": {a: {b: (round(PPL[a][b], 4) if PPL[a][b] is not None else None)
                                  for b in labels} for a in labels},
    }, PPL, labels


# ----------------------------------------------------------------------------- confusion matrix png
def save_confusion(PPL, labels, out_png):
    """Heatmap of raw perplexity: cell[i,j] = PPL('text_i text_j'). Lower (greener) = better."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"  [skip confusion png: {e}]")
        return
    n = len(labels)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if PPL[a][b] is not None:
                mat[i, j] = PPL[a][b]
    vmin = np.nanmin(mat); vmax = np.nanmax(mat)
    fig, ax = plt.subplots(figsize=(1.2 * n + 1, 1.2 * n + 1))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=vmin, vmax=vmax)  # _r: low PPL -> green
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("label2 (text2)"); ax.set_ylabel("label1 (text1)")
    mid = (vmin + vmax) / 2
    span = (vmax - vmin) or 1.0
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                v = mat[i, j]
                color = "black" if abs(v - mid) < 0.35 * span else "white"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="perplexity")
    ax.set_title(Path(out_png).stem)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help="subset of model keys to evaluate")
    ap.add_argument("--no-png", action="store_true", help="skip confusion-matrix PNGs")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    thirty = json.load(open(THIRTY_PATH))
    sixrows = json.load(open(SIXSTEP_PATH))

    results = {
        "config": {"score": "raw_full_sequence_perplexity",
                   "direction_rule": "lower PPL('text_i text_j') => i precedes j",
                   "join": "space", "device": str(device), "dtype": str(dtype)},
        "models": {},
        "summary": [],
    }

    for key in args.models:
        if key not in MODELS:
            print(f"!! unknown model key '{key}', skipping")
            continue
        print(f"\n=== {key} ===")
        model, tok, path = load_model(MODELS[key], device, dtype)
        print(f"  loaded {path}")
        scorer = make_scorer(model, tok, device)

        thirty_res = eval_thirty(scorer, thirty)
        print(f"  30-sample : order_acc={thirty_res['order_accuracy']} "
              f"pairwise={thirty_res['pairwise_direction_accuracy']}")

        six_res, P, labels = eval_sixstep(scorer, sixrows)
        print(f"  6-step    : exact={six_res['exact_match']} "
              f"pairwise={six_res['pairwise_direction_accuracy']} "
              f"tau={six_res['kendall_tau']} order={'->'.join(six_res['predicted_order'])}")

        if not args.no_png:
            save_confusion(P, labels, f"/mnt/localssd/eval_ppl_confusion_{key}.png")

        results["models"][key] = {
            "model_path": path,
            "thirty_sample": thirty_res,
            "six_step": six_res,
        }
        results["summary"].append({
            "model": key,
            "thirty_order_acc": thirty_res["order_accuracy"],
            "thirty_pairwise_acc": thirty_res["pairwise_direction_accuracy"],
            "six_pairwise_acc": six_res["pairwise_direction_accuracy"],
            "six_kendall_tau": six_res["kendall_tau"],
            "six_exact": six_res["exact_match"],
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\n=== SUMMARY ===")
    hdr = f"{'model':<18} {'30_order':>9} {'30_pair':>8} {'6_pair':>7} {'6_tau':>7} {'6_exact':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results["summary"]:
        print(f"{r['model']:<18} {r['thirty_order_acc']:>9} {r['thirty_pairwise_acc']:>8} "
              f"{r['six_pairwise_acc']:>7} {r['six_kendall_tau']:>7} {str(r['six_exact']):>8}")


if __name__ == "__main__":
    main()
