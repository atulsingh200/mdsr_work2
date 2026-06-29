#!/usr/bin/env python3
"""
Permutation-style causal-ordering eval (whole-sequence, NOT pairwise).

For each sample we concatenate ALL its events in EVERY possible order, compute the
full-sequence perplexity of each concatenation, and the ordering with the LOWEST
perplexity wins. We then check how often that winning order equals the gold order.

  - 3-step set (30 samples) : 3! = 6 permutations / sample  -> count correct == (t1,t2,t3)
  - 6-step set (1 sample)   : 6! = 720 permutations         -> best order, rank of gold, tau

Two separators are tried:
  - "space" : join events with a single space  (" ")
  - "eos"   : join with the EOS token <|endoftext|> -- the SAME separator used between
              documents during training (prepare_data.py: doc["text"] + eos)

Output: /mnt/localssd/pythia_permutation_eval_results.json
"""
import itertools
import json
import math

import torch

# reuse model loading + perplexity from the pairwise script
from eval_pythia_perplexity import (
    MODELS, THIRTY_PATH, SIXSTEP_PATH, load_model, seq_nll, kendall_tau,
)

OUT_PATH = "/mnt/localssd/pythia_permutation_eval_results.json"


def make_ppl(model, tok, device):
    cache = {}

    def ppl(text):
        if text not in cache:
            cache[text] = math.exp(seq_nll(model, tok, text, device))
        return cache[text]

    return ppl


def best_permutation(ppl, parts_by_label, labels, sep):
    """Return (best_order, ranked) where ranked is list of (order, perplexity) asc by ppl."""
    scored = []
    for perm in itertools.permutations(labels):
        seq = sep.join(parts_by_label[l] for l in perm)
        scored.append((list(perm), ppl(seq)))
    scored.sort(key=lambda x: x[1])
    return scored[0][0], scored


# --------------------------------------------------------------------------- 3-step (30 samples)
def eval_three(ppl, data, sep):
    labels = ["t1", "t2", "t3"]
    gold = ["t1", "t2", "t3"]
    n_correct = 0
    per_sample = []
    for s in data:
        parts = {l: s[l] for l in labels}
        best, scored = best_permutation(ppl, parts, labels, sep)
        ok = best == gold
        n_correct += int(ok)
        per_sample.append({
            "id": s.get("id"),
            "best_order": best,
            "best_perplexity": round(scored[0][1], 4),
            "gold_perplexity": round(next(p for o, p in scored if o == gold), 4),
            "order_correct": ok,
        })
    return {
        "n_correct": n_correct,
        "n_total": len(data),
        "order_accuracy": round(n_correct / len(data), 4),
        "per_sample": per_sample,
    }


# --------------------------------------------------------------------------- 6-step (1 sample)
def eval_six(ppl, rows, sep):
    labels = sorted(
        set(r["label1"] for r in rows) | set(r["label2"] for r in rows),
        key=lambda l: (l[0], int(l[1:])) if l[1:].isdigit() else (l, 0),
    )
    gold = labels[:]
    text_of = {}
    for r in rows:
        text_of[r["label1"]] = r["text1"]
        text_of[r["label2"]] = r["text2"]

    best, scored = best_permutation(ppl, text_of, labels, sep)
    # 1-based rank of the gold ordering among all permutations (lower ppl = better)
    rank_of_gold = next(i for i, (o, _) in enumerate(scored, 1) if o == gold)
    return {
        "best_order": best,
        "correct_order": gold,
        "exact_match": best == gold,
        "best_perplexity": round(scored[0][1], 4),
        "gold_perplexity": round(next(p for o, p in scored if o == gold), 4),
        "rank_of_gold": rank_of_gold,
        "n_permutations": len(scored),
        "kendall_tau_best_vs_gold": round(kendall_tau(best, gold), 4),
        "top5_orders": [["->".join(o), round(p, 4)] for o, p in scored[:5]],
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    thirty = json.load(open(THIRTY_PATH))
    sixrows = json.load(open(SIXSTEP_PATH))

    results = {
        "config": {"score": "raw_full_sequence_perplexity",
                   "method": "all-permutations, lowest perplexity wins",
                   "separators": {"space": "single space", "eos": "training doc separator <|endoftext|>"},
                   "device": str(device), "dtype": str(dtype)},
        "models": {},
        "summary": [],
    }

    for key in MODELS:
        print(f"\n=== {key} ===")
        model, tok, path = load_model(MODELS[key], device, dtype)
        ppl = make_ppl(model, tok, device)
        seps = {"space": " ", "eos": tok.eos_token or " "}

        model_res = {"model_path": path, "separators": {}}
        for sname, sval in seps.items():
            three = eval_three(ppl, thirty, sval)
            six = eval_six(ppl, sixrows, sval)
            model_res["separators"][sname] = {"three_step": three, "six_step": six}
            print(f"  [{sname:>5}] 3-step: {three['n_correct']}/{three['n_total']} "
                  f"(acc={three['order_accuracy']}) | 6-step: exact={six['exact_match']} "
                  f"rank_of_gold={six['rank_of_gold']}/{six['n_permutations']} "
                  f"tau={six['kendall_tau_best_vs_gold']} best={'->'.join(six['best_order'])}")
            results["summary"].append({
                "model": key, "separator": sname,
                "three_acc": three["order_accuracy"],
                "three_correct": three["n_correct"],
                "six_exact": six["exact_match"],
                "six_rank_of_gold": six["rank_of_gold"],
                "six_tau": six["kendall_tau_best_vs_gold"],
            })

        results["models"][key] = model_res
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\n=== SUMMARY ===")
    hdr = f"{'model':<18} {'sep':>5} {'3_acc':>6} {'3_ok':>5} {'6_exact':>8} {'6_rank':>7} {'6_tau':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results["summary"]:
        print(f"{r['model']:<18} {r['separator']:>5} {r['three_acc']:>6} {r['three_correct']:>5} "
              f"{str(r['six_exact']):>8} {r['six_rank_of_gold']:>7} {r['six_tau']:>7}")


if __name__ == "__main__":
    main()
