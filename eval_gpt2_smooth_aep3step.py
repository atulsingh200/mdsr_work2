#!/usr/bin/env python3
"""
Permutation-style causal-ordering eval for gpt2-smooth on the new AEP 3-step samples.

For each of the 10 samples we enumerate all 3! = 6 permutations of (t1, t2, t3),
compute the full-sequence perplexity of each concatenation, and the ordering with the
LOWEST perplexity wins.  We then check whether that winning order equals the gold
order (t1 -> t2 -> t3).

Two separators are tried (mirroring eval_pythia_permutation.py):
  - "space" : events joined with a single space
  - "eos"   : events joined with the EOS token <|endoftext|>

Data  : /mnt/localssd/automation/internship-causal-embedding/data/scrap_transcript/aep_3step_workflows_simple.json
Model : /mnt/localssd/automation/internship-causal-embedding/pythia_train/output_gpt2_smooth
Output: /mnt/localssd/gpt2_smooth_aep3step_results.json

Run:
  python3 /mnt/localssd/eval_gpt2_smooth_aep3step.py
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────── #
DATA_PATH  = Path("/mnt/localssd/automation/internship-causal-embedding"
                  "/data/scrap_transcript/aep_3step_workflows_simple.json")
MODEL_DIR  = Path("/mnt/localssd/automation/internship-causal-embedding"
                  "/pythia_train/output_gpt2_smooth")
OUT_PATH   = Path("/mnt/localssd/gpt2_smooth_aep3step_results.json")
MAX_LEN    = 1024


# ── model loading ─────────────────────────────────────────────────────────── #
def load_model(d: Path, device, dtype):
    """Load from top-level dir if weights exist, else from latest checkpoint-* subdir."""
    if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
        path = d
    else:
        ckpts = sorted(
            [p for p in d.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No weights found under {d}")
        path = ckpts[-1]
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device)
    model.eval()
    print(f"  Loaded model from: {path}")
    return model, tok, str(path)


# ── perplexity ────────────────────────────────────────────────────────────── #
@torch.no_grad()
def seq_nll(model, tok, text: str, device) -> float:
    """Mean per-token negative log-likelihood of the full sequence."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    input_ids = enc["input_ids"].to(device)
    attn      = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
    return float(out.loss.item())


def make_ppl_fn(model, tok, device):
    """Return a cached perplexity function."""
    cache: dict[str, float] = {}
    def ppl(text: str) -> float:
        if text not in cache:
            cache[text] = math.exp(seq_nll(model, tok, text, device))
        return cache[text]
    return ppl


# ── evaluation ────────────────────────────────────────────────────────────── #
LABELS = ["t1", "t2", "t3"]
GOLD   = ["t1", "t2", "t3"]

def eval_samples(ppl_fn, data: list[dict], sep: str) -> dict:
    """
    For each sample enumerate all 6 permutations, pick lowest perplexity,
    compare to gold (t1 -> t2 -> t3).
    """
    n_correct  = 0
    per_sample = []

    for s in data:
        parts = {l: s[l] for l in LABELS}

        # score all 6 permutations
        scored = []
        for perm in itertools.permutations(LABELS):
            seq = sep.join(parts[l] for l in perm)
            scored.append((list(perm), ppl_fn(seq)))
        scored.sort(key=lambda x: x[1])  # ascending perplexity — best first

        best_order = scored[0][0]
        best_ppl   = scored[0][1]
        gold_ppl   = next(p for o, p in scored if o == GOLD)
        ok         = (best_order == GOLD)
        n_correct += int(ok)

        # pairwise direction accuracy: for each gold-ordered pair (i, j) with i<j,
        # check whether concatenating in forward order has lower perplexity
        pair_hits = 0
        pair_data = []
        for i, j in itertools.combinations(LABELS, 2):   # (t1,t2), (t1,t3), (t2,t3)
            fwd = sep.join([parts[i], parts[j]])
            rev = sep.join([parts[j], parts[i]])
            fwd_ppl = ppl_fn(fwd)
            rev_ppl = ppl_fn(rev)
            hit = fwd_ppl < rev_ppl
            pair_hits += int(hit)
            pair_data.append({
                "pair": f"{i}->{j}",
                "fwd_ppl": round(fwd_ppl, 4),
                "rev_ppl": round(rev_ppl, 4),
                "correct_direction": hit,
            })

        per_sample.append({
            "id":             s.get("id"),
            "title":          s.get("title", ""),
            "gold_order":     GOLD,
            "best_order":     best_order,
            "order_correct":  ok,
            "best_ppl":       round(best_ppl, 4),
            "gold_ppl":       round(gold_ppl, 4),
            "ppl_margin":     round(gold_ppl - best_ppl, 4),   # positive = gold is worse
            "all_permutations": [
                {"order": o, "ppl": round(p, 4)} for o, p in scored
            ],
            "pairwise": pair_data,
        })

    pairwise_acc = sum(
        sum(int(p["correct_direction"]) for p in s["pairwise"])
        for s in per_sample
    ) / (len(data) * 3)   # 3 pairs per sample

    return {
        "n_correct":      n_correct,
        "n_total":        len(data),
        "order_accuracy": round(n_correct / len(data), 4),
        "pairwise_direction_accuracy": round(pairwise_acc, 4),
        "per_sample":     per_sample,
    }


# ── main ──────────────────────────────────────────────────────────────────── #
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    data = json.loads(DATA_PATH.read_text())
    print(f"Loaded {len(data)} samples from {DATA_PATH.name}")

    model, tok, path = load_model(MODEL_DIR, device, dtype)
    ppl_fn = make_ppl_fn(model, tok, device)

    eos_sep = tok.eos_token or " "
    separators = {"space": " ", "eos": eos_sep}

    results = {
        "config": {
            "model":   "gpt2-smooth",
            "model_path": path,
            "data":    str(DATA_PATH),
            "n_samples": len(data),
            "method":  "all-permutations (3! = 6), lowest perplexity wins",
            "score":   "full-sequence perplexity (exp of mean per-token NLL)",
            "gold_order": GOLD,
            "device":  str(device),
            "dtype":   str(dtype),
        },
        "separators": {},
        "summary": [],
    }

    for sname, sval in separators.items():
        print(f"\n--- separator: {sname!r} ({repr(sval)}) ---")
        res = eval_samples(ppl_fn, data, sval)
        results["separators"][sname] = res
        results["summary"].append({
            "separator":             sname,
            "order_accuracy":        res["order_accuracy"],
            "n_correct":             res["n_correct"],
            "n_total":               res["n_total"],
            "pairwise_direction_acc": res["pairwise_direction_accuracy"],
        })
        print(f"  order accuracy : {res['n_correct']}/{res['n_total']} = {res['order_accuracy']}")
        print(f"  pairwise acc   : {res['pairwise_direction_accuracy']}")
        print()
        print(f"  {'ID':<4} {'Title':<35} {'Best order':<20} {'Gold PPL':>9} {'Best PPL':>9} {'OK':>4}")
        print(f"  {'-'*4} {'-'*35} {'-'*20} {'-'*9} {'-'*9} {'-'*4}")
        for s in res["per_sample"]:
            mark = "✓" if s["order_correct"] else "✗"
            print(f"  {s['id']:<4} {s['title']:<35} {'->'.join(s['best_order']):<20} "
                  f"{s['gold_ppl']:>9.4f} {s['best_ppl']:>9.4f} {mark:>4}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {OUT_PATH}")

    print("\n=== SUMMARY ===")
    print(f"  {'separator':<8} {'order_acc':>10} {'pairwise_acc':>13}")
    print(f"  {'-'*8} {'-'*10} {'-'*13}")
    for r in results["summary"]:
        print(f"  {r['separator']:<8} {r['order_accuracy']:>10} {r['pairwise_direction_acc']:>13}")


if __name__ == "__main__":
    main()
