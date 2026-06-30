#!/usr/bin/env python3
"""OOD evaluation for CROSS-ENCODER baseline sweep checkpoints.

Cross-encoder counterpart of training/eval_ood.py. Loads a trained
CrossEncoderClassifier from a run directory and scores two held-out test sets:

  1. test_samples_milan.json
       Flat list of {label1, label2, text1, text2} pairs from a multi-step
       workflow. All pair directions are scored; tournament + net-precedence
       (order_events.py) recovers the full event ordering.

  2. ajo_orchestrated_workflows_flat.json
       30 samples, each {id, t1, t2, t3} where t1->t2->t3 is correct.
       All 6 directed pairs are scored; _order_three() recovers the predicted
       order.

The ONLY differences from the bi-encoder eval are model loading and scoring:
the cross-encoder JOINTLY tokenizes ([CLS] t1 [SEP] t2 [SEP]) and the model
returns a single (B,) logit. The Milan/AJO ordering logic and the
ood_results.json output structure are byte-for-byte identical, so the bi-encoder
and cross-encoder sweeps remain directly comparable.

Writes {run_dir}/ood_results.json.

Usage:
  python eval_ood_ce.py --run-dir /mnt/localssd/crossencoder_sweep/runs/aep_dataset/bge-large-en-v1.5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from itertools import combinations
from pathlib import Path

import torch
from transformers import AutoTokenizer

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from classifier.crossencoder.model import CrossEncoderClassifier  # noqa: E402

BATCH_SIZE = 32
_SAFE_ENC_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})

# AJO: always 3 labels, ground truth always [t1, t2, t3]
_AJO_LABELS = ["t1", "t2", "t3"]
_AJO_CORRECT_ORDER = ["t1", "t2", "t3"]
_AJO_DIRECTED_PAIRS = [(a, b) for a in _AJO_LABELS for b in _AJO_LABELS if a != b]


# ---------------------------------------------------------------------------
# Scoring (JOINT tokenization — the cross-encoder difference)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_pairs(
    model: CrossEncoderClassifier,
    tokenizer,
    texts_a: list[str],
    texts_b: list[str],
    max_len: int,
    device: torch.device,
    use_bf16: bool,
) -> list[float]:
    """Return P(text_a precedes text_b) for each pair via joint encoding."""
    from contextlib import nullcontext
    ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16 else nullcontext
    )
    probs: list[float] = []
    for s in range(0, len(texts_a), BATCH_SIZE):
        ba, bb = texts_a[s:s + BATCH_SIZE], texts_b[s:s + BATCH_SIZE]
        enc = tokenizer(ba, bb, padding=True, truncation="longest_first",
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in _SAFE_ENC_KEYS}
        with ctx():
            logits = model(enc)
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs


# ---------------------------------------------------------------------------
# Milan multi-step ordering (order_events.py algorithm) — reused verbatim
# ---------------------------------------------------------------------------
def _label_sort_key(label: str) -> tuple:
    """Natural sort: t1 < t2 < ... < t10 etc."""
    suffix = label[1:]
    return (label[0], int(suffix)) if suffix.isdigit() else (label, 0)


def _build_score_matrix(
    pairs: list[dict], score_key: str = "score"
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Build (labels, P) where P[i][j] = P(i precedes j)."""
    seen: dict[str, None] = {}
    for r in pairs:
        seen.setdefault(r["label1"], None)
        seen.setdefault(r["label2"], None)
    labels = sorted(seen.keys(), key=_label_sort_key)
    P: dict[str, dict[str, float]] = {a: {} for a in labels}
    for r in pairs:
        P[r["label1"]][r["label2"]] = float(r[score_key])
    return labels, P


def _tournament_order(
    labels: list[str], P: dict[str, dict[str, float]]
) -> tuple[list[str], dict[str, int], dict[str, float]]:
    """Sort labels by tournament wins then net precedence (descending)."""
    wins = {a: 0 for a in labels}
    net = {a: 0.0 for a in labels}

    for i in labels:
        for j in labels:
            if i == j:
                continue
            pij = P[i].get(j)
            pji = P[j].get(i)
            if pij is not None and pji is not None:
                net[i] += pij - pji

    for i, j in combinations(labels, 2):
        pij = P[i].get(j, 0.0)
        pji = P[j].get(i, 0.0)
        if pij >= pji:
            wins[i] += 1
        else:
            wins[j] += 1

    order = sorted(labels, key=lambda a: (-wins[a], -net[a]))
    return order, wins, net


def eval_milan(
    model: CrossEncoderClassifier,
    tokenizer,
    milan_path: Path,
    max_len: int,
    device: torch.device,
    use_bf16: bool,
) -> dict:
    """Score flat Milan pairs and recover the full event ordering."""
    raw: list[dict] = json.loads(milan_path.read_text())

    texts_a = [r["text1"] for r in raw]
    texts_b = [r["text2"] for r in raw]
    probs = score_pairs(model, tokenizer, texts_a, texts_b, max_len, device, use_bf16)

    scored_pairs = [
        {
            "label1": r["label1"],
            "label2": r["label2"],
            "score": round(p, 6),
            "pred": int(p > 0.5),
        }
        for r, p in zip(raw, probs)
    ]

    labels, P = _build_score_matrix(scored_pairs)
    predicted_order, wins, net = _tournament_order(labels, P)
    ground_truth = sorted(labels, key=_label_sort_key)

    n_pairs_correct = 0
    n_pairs_total = 0
    for r in scored_pairs:
        i_rank = ground_truth.index(r["label1"])
        j_rank = ground_truth.index(r["label2"])
        if i_rank == j_rank:
            continue
        n_pairs_total += 1
        gt_positive = i_rank < j_rank
        pred_positive = r["score"] > 0.5
        if gt_positive == pred_positive:
            n_pairs_correct += 1

    per_event = {
        lbl: {
            "tournament_wins": wins[lbl],
            "net_precedence": round(net[lbl], 4),
            "predicted_rank": predicted_order.index(lbl) + 1,
            "true_rank": ground_truth.index(lbl) + 1,
        }
        for lbl in labels
    }

    return {
        "n_events": len(labels),
        "ground_truth_order": ground_truth,
        "predicted_order": predicted_order,
        "order_match": predicted_order == ground_truth,
        "n_pairs_correct": n_pairs_correct,
        "n_pairs_total": n_pairs_total,
        "pairwise_acc": round(n_pairs_correct / max(n_pairs_total, 1), 6),
        "per_event": per_event,
        "scored_pairs": scored_pairs,
    }


# ---------------------------------------------------------------------------
# AJO 3-step ordering — reused verbatim
# ---------------------------------------------------------------------------
def _order_three(P: dict[str, dict[str, float]]) -> list[str]:
    """Rank t1/t2/t3 via tournament wins + net precedence."""
    wins = {a: 0 for a in _AJO_LABELS}
    net = {a: 0.0 for a in _AJO_LABELS}
    for i in _AJO_LABELS:
        for j in _AJO_LABELS:
            if i == j:
                continue
            pij, pji = P[i][j], P[j][i]
            net[i] += pij - pji
    for i, j in itertools.combinations(_AJO_LABELS, 2):
        if P[i][j] >= P[j][i]:
            wins[i] += 1
        else:
            wins[j] += 1
    return sorted(_AJO_LABELS, key=lambda a: (-wins[a], -net[a]))


def eval_ajo(
    model: CrossEncoderClassifier,
    tokenizer,
    ajo_path: Path,
    max_len: int,
    device: torch.device,
    use_bf16: bool,
) -> dict:
    """Score all 30 AJO 3-step samples and compute ordering accuracy."""
    samples: list[dict] = json.loads(ajo_path.read_text())

    all_a: list[str] = []
    all_b: list[str] = []
    for sample in samples:
        for lbl_a, lbl_b in _AJO_DIRECTED_PAIRS:
            all_a.append(sample[lbl_a])
            all_b.append(sample[lbl_b])

    flat_scores = score_pairs(model, tokenizer, all_a, all_b, max_len, device, use_bf16)

    per_sample: list[dict] = []
    n_correct = 0
    n_pairs_per_sample = len(_AJO_DIRECTED_PAIRS)  # 6

    for s_idx, sample in enumerate(samples):
        base = s_idx * n_pairs_per_sample
        P: dict[str, dict[str, float]] = {a: {} for a in _AJO_LABELS}
        pair_scores: dict[str, float] = {}
        for k, (lbl_a, lbl_b) in enumerate(_AJO_DIRECTED_PAIRS):
            sc = flat_scores[base + k]
            P[lbl_a][lbl_b] = sc
            pair_scores[f"{lbl_a}_to_{lbl_b}"] = round(sc, 6)

        predicted = _order_three(P)
        is_correct = predicted == _AJO_CORRECT_ORDER
        if is_correct:
            n_correct += 1

        per_sample.append({
            "id": sample["id"],
            "ground_truth_order": _AJO_CORRECT_ORDER,
            "predicted_order": predicted,
            "correct": is_correct,
            "pair_scores": pair_scores,
        })

    return {
        "n_samples": len(samples),
        "n_correct": n_correct,
        "perfect_order_rate": round(n_correct / max(len(samples), 1), 6),
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="OOD evaluation for a cross-encoder baseline sweep checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-dir", required=True,
                    help="Directory containing best.pt and config.json.")
    ap.add_argument("--milan-json", default="/mnt/localssd/test_samples_milan.json")
    ap.add_argument("--ajo-json", default="/mnt/localssd/ajo_orchestrated_workflows_flat.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: best.pt not found in {run_dir}", flush=True)
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- load checkpoint --
    print(f"loading checkpoint <- {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    # Match the precision used at train time (fp32 for DeBERTa-v3).
    use_bf16 = (device.type == "cuda") and bool(cfg.get("bf16", True))
    base_model = cfg["base_model"]
    dropout = cfg.get("dropout", 0.1)
    max_len = cfg["max_seq_len"]
    print(f"device={device}  bf16={use_bf16}  base_model={base_model}  max_seq_len={max_len}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = CrossEncoderClassifier(base_model, dropout=dropout).to(device)
    # Force fp32 master weights before load_state_dict — some backbones load in
    # fp16 (e.g. DeBERTa-v3), which would mismatch the fp32 head.
    model = model.float()
    model.load_state_dict(ckpt["model"])
    model.eval()

    milan_path = Path(args.milan_json)
    ajo_path = Path(args.ajo_json)

    # -- Milan eval --
    print(f"\n=== Milan eval ({milan_path.name}) ===", flush=True)
    milan_results = eval_milan(model, tokenizer, milan_path, max_len, device, use_bf16)
    print(f"  ground_truth : {milan_results['ground_truth_order']}", flush=True)
    print(f"  predicted    : {milan_results['predicted_order']}", flush=True)
    print(f"  order_match  : {milan_results['order_match']}", flush=True)
    print(
        f"  pairwise_acc : {milan_results['pairwise_acc']:.4f} "
        f"({milan_results['n_pairs_correct']}/{milan_results['n_pairs_total']})",
        flush=True,
    )

    # -- AJO eval --
    print(f"\n=== AJO ordering eval ({ajo_path.name}) ===", flush=True)
    ajo_results = eval_ajo(model, tokenizer, ajo_path, max_len, device, use_bf16)
    print(
        f"  correct: {ajo_results['n_correct']}/{ajo_results['n_samples']}  "
        f"perfect_order_rate={ajo_results['perfect_order_rate']:.4f}",
        flush=True,
    )

    # -- write results (drop verbose scored_pairs, keep per_event) --
    milan_compact = {k: v for k, v in milan_results.items() if k != "scored_pairs"}
    ood = {
        "dataset_slug": cfg.get("dataset_slug", ""),
        "model_slug": cfg.get("model_slug", ""),
        "base_model": base_model,
        "model_type": "cross_encoder",
        "milan": milan_compact,
        "ajo_ordering": ajo_results,
    }
    out_path = run_dir / "ood_results.json"
    out_path.write_text(json.dumps(ood, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
