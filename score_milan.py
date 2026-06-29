#!/usr/bin/env python3
"""Score text pairs with the trained ensemble cross-encoder (aep_dataset).

Loads the two BERT-base cross-encoders trained by run_all_aep_dataset.sh
(ce_semantic + ce_indomain), scores each pair as

    P(text_1 causally precedes text_2) = sigmoid(logit)

and blends with the best alpha chosen on val (0.5 -> simple mean):

    p_blend = alpha * p_semantic + (1 - alpha) * p_indomain

Flat input  (default): list of {label1,label2,text1,text2} objects
  Reads  /mnt/localssd/test_samples_milan.json
  Writes /mnt/localssd/test_samples_milan_scored.json

3-step input (--three-step flag): list-of-lists, each inner list is one
  3-step workflow with 6 pairwise combos (e.g. test_samples_milan_3step_new5.json)
  Output preserves the nested structure and adds per-workflow ordering accuracy.

Usage:
  cd /mnt/localssd/automation/internship-causal-embedding
  source .venv/bin/activate

  # flat pairs (original behaviour)
  python3 /mnt/localssd/score_milan.py

  # 3-step workflow file
  python3 /mnt/localssd/score_milan.py --three-step \\
      --input  /mnt/localssd/test_samples_milan_3step_new5.json \\
      --output /mnt/localssd/test_samples_milan_3step_new5_scored_ce.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/localssd/automation/internship-causal-embedding")
sys.path.insert(0, str(REPO))
from src.classifier.crossencoder.model import CrossEncoderClassifier  # noqa: E402

RUNS = REPO / "runs/crossencoder"
SEM_CKPT = RUNS / "ce_semantic" / "best.pt"
IND_CKPT = RUNS / "ce_indomain" / "best.pt"
ALPHA = 0.5  # blend_best_alpha.alpha_on_semantic from ensemble_metrics.json

SAMPLES = Path("/mnt/localssd/test_samples_milan.json")
OUT = Path("/mnt/localssd/test_samples_milan_scored.json")

BASE_MODEL = "bert-base-uncased"
MAX_LEN = 512  # matches training config
BATCH_SIZE = 16


def load_model(ckpt_path: Path, device) -> CrossEncoderClassifier:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    base = cfg.get("base_model", BASE_MODEL)
    dropout = cfg.get("dropout", 0.1)
    model = CrossEncoderClassifier(base, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def score_pairs(model, tokenizer, t1, t2, device, use_bf16) -> list[float]:
    """Return P(text_1 precedes text_2) for each pair."""
    probs: list[float] = []
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.no_grad())
    for s in range(0, len(t1), BATCH_SIZE):
        bt1, bt2 = t1[s:s + BATCH_SIZE], t2[s:s + BATCH_SIZE]
        enc = tokenizer(bt1, bt2, padding=True, truncation="longest_first",
                        max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with ctx:
            logits = model(enc)
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs


def _build_scored_pairs(pairs, p_sem, p_ind) -> list[dict]:
    results = []
    for p, ps, pi in zip(pairs, p_sem, p_ind):
        blend = ALPHA * ps + (1 - ALPHA) * pi
        results.append({
            **{k: p[k] for k in ("label1", "label2", "workflow") if k in p},
            "text1": p["text1"],
            "text2": p["text2"],
            "score_semantic": round(ps, 6),
            "score_indomain": round(pi, 6),
            "score": round(blend, 6),
            "pred": int(blend > 0.5),
        })
    return results


def _rank_workflow(scored_pairs: list[dict]) -> dict:
    """Rank t1/t2/t3 by net score sum across all 6 pairwise combos."""
    labels = ["t1", "t2", "t3"]
    net: dict[str, float] = {l: 0.0 for l in labels}
    for p in scored_pairs:
        l1, l2 = p.get("label1"), p.get("label2")
        if l1 in net:
            net[l1] += p["score"]
        if l2 in net:
            net[l2] -= p["score"]
    predicted_order = sorted(labels, key=lambda l: net[l], reverse=True)
    correct_order = ["t1", "t2", "t3"]
    return {
        "predicted_order": predicted_order,
        "correct_order": correct_order,
        "order_correct": predicted_order == correct_order,
        "net_scores": {l: round(net[l], 6) for l in labels},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pairs with the cross-encoder ensemble.")
    parser.add_argument("--three-step", action="store_true",
                        help="Input is a list-of-lists (3-step workflow format).")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to input JSON (overrides built-in default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to output JSON (overrides built-in default).")
    args = parser.parse_args()

    input_path = args.input or SAMPLES
    output_path = args.output or OUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"device={device}  bf16={use_bf16}")

    raw = json.loads(input_path.read_text())

    if args.three_step:
        workflows: list[list[dict]] = raw
        flat_pairs = [p for wf in workflows for p in wf]
        print(f"loaded {len(workflows)} workflows ({len(flat_pairs)} pairs) from {input_path}")
    else:
        flat_pairs = raw
        print(f"loaded {len(flat_pairs)} pairs from {input_path}")

    t1 = [p["text1"] for p in flat_pairs]
    t2 = [p["text2"] for p in flat_pairs]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"loading semantic CE  <- {SEM_CKPT}")
    sem = load_model(SEM_CKPT, device)
    p_sem = score_pairs(sem, tokenizer, t1, t2, device, use_bf16)
    del sem
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"loading indomain CE  <- {IND_CKPT}")
    ind = load_model(IND_CKPT, device)
    p_ind = score_pairs(ind, tokenizer, t1, t2, device, use_bf16)
    del ind
    if device.type == "cuda":
        torch.cuda.empty_cache()

    flat_scored = _build_scored_pairs(flat_pairs, p_sem, p_ind)

    if args.three_step:
        results = []
        idx = 0
        correct_count = 0
        for wf_idx, wf in enumerate(workflows):
            wf_scored = flat_scored[idx: idx + len(wf)]
            idx += len(wf)
            ranking = _rank_workflow(wf_scored)
            results.append({"ranking": ranking, "pairs": wf_scored})
            status = "CORRECT" if ranking["order_correct"] else "WRONG"
            workflow_name = wf[0].get("workflow", f"workflow_{wf_idx + 1}")
            print(f"\n  [{wf_idx + 1}] {workflow_name}")
            print(f"       predicted: {ranking['predicted_order']}  [{status}]")
            print(f"       net_scores: {ranking['net_scores']}")
            for p in wf_scored:
                lbl = f"{p.get('label1','?')}->{p.get('label2','?')}"
                print(f"         {lbl:8s} score={p['score']:.4f}  pred={p['pred']}")
            if ranking["order_correct"]:
                correct_count += 1
        print(f"\nsorting accuracy: {correct_count}/{len(workflows)} workflows correctly ordered")
    else:
        results = flat_scored
        print(f"\n(score = P(text1 causally precedes text2), ensemble alpha={ALPHA})\n")
        for r in results:
            lbl = f"{r.get('label1','?')}->{r.get('label2','?')}"
            print(f"  {lbl:10s} score={r['score']:.4f}  pred={r['pred']}")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote results to {output_path}")


if __name__ == "__main__":
    main()
