#!/usr/bin/env python3
"""Score text pairs with the trained GPT-2 causal classifier.

Model: /mnt/localssd/automation/internship-causal-embedding/gpt2_causal/checkpoints/best_model

The model was trained with the prompt format:
    [text1 tokens] [SEP] [text2 tokens] <0|1>

Scoring: at the last token position, compare logit(<1>) vs logit(<0>).
    score = softmax([score_0, score_1])[1]  => P(text1 causally precedes text2)

Flat input (default):
  Reads  /mnt/localssd/test_samples_milan.json
  Writes /mnt/localssd/test_samples_milan_scored_gpt2.json

AJO ordering test (--test-ajo flag):
  Reads  /mnt/localssd/ajo_orchestrated_workflows_flat.json
         Each record has {id, t1, t2, t3} where t1->t2->t3 is the correct order.
  Scores all 6 directed pairs (both directions per unordered pair), then ranks
  with tournament + net-precedence (same logic as the bi-encoder script).
  Writes /mnt/localssd/ajo_orchestrated_workflows_flat_scored_gpt2.json

Usage:
  cd /mnt/localssd/automation/internship-causal-embedding
  source .venv/bin/activate

  # flat pairs
  python3 /mnt/localssd/score_milan_gpt2.py

  # AJO 30-sample ordering test
  python3 /mnt/localssd/score_milan_gpt2.py --test-ajo

  # custom input/output paths
  python3 /mnt/localssd/score_milan_gpt2.py --input /path/to/input.json --output /path/to/out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CKPT_DIR = Path(
    "/mnt/localssd/automation/internship-causal-embedding"
    "/gpt2_causal/checkpoints/best_model"
)
SAMPLES = Path("/mnt/localssd/test_samples_milan.json")
AJO_SAMPLES = Path("/mnt/localssd/ajo_orchestrated_workflows_flat.json")
FLAT_OUT = Path("/mnt/localssd/test_samples_milan_scored_gpt2.json")
AJO_OUT = Path("/mnt/localssd/ajo_orchestrated_workflows_flat_scored_gpt2.json")

MAX_LEN = 1024
BATCH_SIZE = 16

_LABELS = ["t1", "t2", "t3"]
_CORRECT_ORDER = ["t1", "t2", "t3"]
_DIRECTED_PAIRS = [(a, b) for a in _LABELS for b in _LABELS if a != b]


# ---------------------------------------------------------------------------
# Tokenizer helpers (mirrors dataset.py exactly)
# ---------------------------------------------------------------------------

def _build_tokenizer(ckpt_dir: Path) -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained(str(ckpt_dir))
    # The tokenizer was saved with the special tokens already added; loading
    # from the checkpoint restores them automatically.  We just verify.
    assert tok.convert_tokens_to_ids("<0>") != tok.unk_token_id, \
        "Special token <0> not found — wrong checkpoint?"
    return tok


def _truncate_symmetric(ids1: list, ids2: list, budget: int):
    excess = len(ids1) + len(ids2) - budget
    if excess <= 0:
        return ids1, ids2
    trim_each = math.ceil(excess / 2)
    ids1 = ids1[:max(1, len(ids1) - trim_each)]
    ids2 = ids2[:max(1, len(ids2) - trim_each)]
    excess2 = len(ids1) + len(ids2) - budget
    if excess2 > 0:
        ids2 = ids2[:max(1, len(ids2) - excess2)]
    return ids1, ids2


def _build_input(tok: GPT2Tokenizer, text1: str, text2: str) -> list[int]:
    """Build [text1] [SEP] [text2] (no label token appended — we score at this pos)."""
    sep_id = tok.convert_tokens_to_ids("[SEP]")
    budget = MAX_LEN - 2  # reserve for [SEP] + label slot
    ids1 = tok.encode(text1, add_special_tokens=False)
    ids2 = tok.encode(text2, add_special_tokens=False)
    ids1, ids2 = _truncate_symmetric(ids1, ids2, budget)
    return ids1 + [sep_id] + ids2


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_pairs(
    model: GPT2LMHeadModel,
    tok: GPT2Tokenizer,
    t1_list: list[str],
    t2_list: list[str],
    device: torch.device,
    use_bf16: bool,
) -> list[float]:
    """Return P(text1 causally precedes text2) for each pair."""
    zero_id = tok.convert_tokens_to_ids("<0>")
    one_id = tok.convert_tokens_to_ids("<1>")
    pad_id = tok.pad_token_id
    probs: list[float] = []

    from contextlib import nullcontext
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    for start in range(0, len(t1_list), BATCH_SIZE):
        bt1 = t1_list[start: start + BATCH_SIZE]
        bt2 = t2_list[start: start + BATCH_SIZE]

        seqs = [_build_input(tok, a, b) for a, b in zip(bt1, bt2)]
        max_seq = max(len(s) for s in seqs)

        input_ids_list, attn_mask_list = [], []
        for seq in seqs:
            pad_len = max_seq - len(seq)
            input_ids_list.append(seq + [pad_id] * pad_len)
            attn_mask_list.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attn_mask = torch.tensor(attn_mask_list, dtype=torch.long, device=device)

        with ctx():
            out = model(input_ids=input_ids, attention_mask=attn_mask)

        logits = out.logits  # [B, seq_len, vocab]
        # Last real token position (the one just before a label would be appended)
        seq_lens = attn_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=device)
        last_logits = logits[batch_idx, seq_lens]  # [B, vocab]

        score_0 = last_logits[:, zero_id]
        score_1 = last_logits[:, one_id]
        # softmax over the two label logits -> P(label=1)
        label_logits = torch.stack([score_0, score_1], dim=1)  # [B, 2]
        p1 = torch.softmax(label_logits, dim=1)[:, 1]
        probs.extend(p1.float().cpu().tolist())

    return probs


# ---------------------------------------------------------------------------
# Flat-pair scoring
# ---------------------------------------------------------------------------

def _score_flat(
    model, tok, pairs: list[dict], device, use_bf16
) -> list[dict]:
    t1 = [p["text1"] for p in pairs]
    t2 = [p["text2"] for p in pairs]
    probs = score_pairs(model, tok, t1, t2, device, use_bf16)
    results = []
    for p, prob in zip(pairs, probs):
        results.append({
            **{k: p[k] for k in ("label1", "label2", "workflow") if k in p},
            "text1": p["text1"],
            "text2": p["text2"],
            "score": round(prob, 6),
            "pred": int(prob > 0.5),
        })
    return results


# ---------------------------------------------------------------------------
# AJO 3-step ordering (tournament + net-precedence, same as bi-encoder script)
# ---------------------------------------------------------------------------

def _order_three(P: dict[str, dict[str, float]]) -> list[str]:
    wins = {a: 0 for a in _LABELS}
    net = {a: 0.0 for a in _LABELS}
    for i in _LABELS:
        for j in _LABELS:
            if i == j:
                continue
            net[i] += P[i][j] - P[j][i]
    for i, j in itertools.combinations(_LABELS, 2):
        if P[i][j] >= P[j][i]:
            wins[i] += 1
        else:
            wins[j] += 1
    return sorted(_LABELS, key=lambda a: (-wins[a], -net[a]))


def _score_ajo(
    model, tok, samples: list[dict], device, use_bf16
) -> tuple[list[dict], int]:
    all_a: list[str] = []
    all_b: list[str] = []
    for sample in samples:
        for lbl_a, lbl_b in _DIRECTED_PAIRS:
            all_a.append(sample[lbl_a])
            all_b.append(sample[lbl_b])

    flat_scores = score_pairs(model, tok, all_a, all_b, device, use_bf16)

    results: list[dict] = []
    correct_count = 0
    n_pairs = len(_DIRECTED_PAIRS)  # 6

    for s_idx, sample in enumerate(samples):
        base = s_idx * n_pairs
        P: dict[str, dict[str, float]] = {a: {} for a in _LABELS}
        pair_records: list[dict] = []
        for k, (lbl_a, lbl_b) in enumerate(_DIRECTED_PAIRS):
            sc = flat_scores[base + k]
            P[lbl_a][lbl_b] = sc
            pair_records.append({
                "label1": lbl_a, "label2": lbl_b,
                "score": round(sc, 6), "pred": int(sc > 0.5),
            })

        predicted = _order_three(P)
        is_correct = predicted == _CORRECT_ORDER
        if is_correct:
            correct_count += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"\n  [sample {sample['id']}]  predicted: {predicted}  [{status}]")
        for pr in pair_records:
            print(f"    {pr['label1']}->{pr['label2']}  score={pr['score']:.4f}  pred={pr['pred']}")

        results.append({
            "id": sample["id"],
            "correct_order": _CORRECT_ORDER,
            "predicted_order": predicted,
            "order_correct": is_correct,
            "pairs": pair_records,
        })

    return results, correct_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score pairs with the GPT-2 causal classifier.")
    parser.add_argument("--test-ajo", action="store_true",
                        help="Score all 6 directed pairs of each AJO 3-step workflow "
                             "and report ordering accuracy.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to input JSON (overrides built-in default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to output JSON (overrides built-in default).")
    args = parser.parse_args()

    if args.test_ajo:
        input_path = args.input or AJO_SAMPLES
        output_path = args.output or AJO_OUT
    else:
        input_path = args.input or SAMPLES
        output_path = args.output or FLAT_OUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"device={device}  bf16={use_bf16}")

    print(f"loading GPT-2 checkpoint <- {CKPT_DIR}")
    tok = _build_tokenizer(CKPT_DIR)
    model = GPT2LMHeadModel.from_pretrained(str(CKPT_DIR))
    model.to(device)
    model.eval()
    print(f"vocab_size={len(tok)}  max_len={MAX_LEN}")

    raw = json.loads(input_path.read_text())

    if args.test_ajo:
        samples: list[dict] = raw
        print(f"loaded {len(samples)} AJO samples from {input_path}  (GPT-2 causal)")
        print("scoring 6 directed pairs per sample (both directions per unordered pair) ...\n")
        results, correct_count = _score_ajo(model, tok, samples, device, use_bf16)
        print(f"\nordering accuracy: {correct_count}/{len(samples)} samples correctly ordered")

    else:
        pairs: list[dict] = raw
        print(f"loaded {len(pairs)} pairs from {input_path}  (GPT-2 causal)")
        results = _score_flat(model, tok, pairs, device, use_bf16)
        print("\n(score = P(text1 causally precedes text2))\n")
        for r in results:
            lbl = f"{r.get('label1','?')}->{r.get('label2','?')}"
            print(f"  {lbl:10s} score={r['score']:.4f}  pred={r['pred']}")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote results to {output_path}")


if __name__ == "__main__":
    main()
