#!/usr/bin/env python3
"""Score text pairs with the trained DUAL-ENCODER (bi-encoder) baseline.

Uses runs/classifier/best_mlp_bge_small_aep (BGE-small, two untied encoders).
Unlike the cross-encoder, this model tokenizes the two texts SEPARATELY, encodes
each (CLS pool + L2 normalize), concatenates [A; B; A-B; A*B], and an MLP head
maps to one directional logit:

    P(text_1 causally precedes text_2) = sigmoid(logit)

The model is rebuilt exactly as trained from the checkpoint's own `cfg`
(same pattern as src/classifier/training/eval_only.py).

Flat input  (default): list of {label1, label2, text1, text2} objects
  Reads  /mnt/localssd/test_samples_milan.json
  Writes /mnt/localssd/test_samples_milan_scored_biencoder.json

3-step input (--three-step flag): list-of-lists, each inner list is one
  3-step workflow with 6 pairwise combos (e.g. test_samples_milan_3step_new5.json)
  Output preserves the nested structure and adds per-workflow ordering accuracy.

AJO ordering test (--test-ajo flag):
  Reads  /mnt/localssd/ajo_orchestrated_workflows_flat.json
         Each record has {id, t1, t2, t3} where t1->t2->t3 is the correct order.
  For each sample the model scores all 6 permutations of the 3 texts using the
  3 directional pairs within each permutation (pos0->pos1, pos1->pos2, pos0->pos2).
  The permutation whose summed scores are highest is the model's predicted order.
  Writes /mnt/localssd/ajo_orchestrated_workflows_flat_scored.json

Usage:
  cd /mnt/localssd/automation/internship-causal-embedding
  source .venv/bin/activate

  # flat pairs (original behaviour)
  python3 /mnt/localssd/score_milan_biencoder.py

  # 3-step workflow file
  python3 /mnt/localssd/score_milan_biencoder.py --three-step \\
      --input  /mnt/localssd/test_samples_milan_3step_new5.json \\
      --output /mnt/localssd/test_samples_milan_3step_new5_scored.json

  # AJO 30-sample ordering test (bi-encoder)
  python3 /mnt/localssd/score_milan_biencoder.py --test-ajo

  # AJO 30-sample ordering test (CrossEncoder2x, runs/crossencoder2x/ce2x_org)
  python3 /mnt/localssd/score_milan_biencoder.py --test-ajo --cross-encoder

  # flat pairs scored with the cross-encoder (test_samples_milan.json)
  python3 /mnt/localssd/score_milan_biencoder.py --cross-encoder

  # any CrossEncoder2x run dir via --ce-ckpt (implies --cross-encoder; reads
  # native_backbone from cfg, so DeBERTa-v3-large loads correctly). AJO ordering test:
  python3 /mnt/localssd/score_milan_biencoder.py --test-ajo --ce-ckpt \\
      /mnt/localssd/automation/internship-causal-embedding/runs/crossencoder2x_deberta/new_aep_workflow_scrap

  # ... or flat pairs with the same model and a custom output:
  python3 /mnt/localssd/score_milan_biencoder.py \\
      --ce-ckpt runs/crossencoder2x_deberta/new_aep_workflow_scrap \\
      --output /mnt/localssd/milan_scored_deberta.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/localssd/automation/internship-causal-embedding")
sys.path.insert(0, str(REPO))
from src.classifier.model import DirectionalClassifier  # noqa: E402
from src.classifier.reasoning_model import ReasoningClassifier  # noqa: E402

CKPT = REPO / "/mnt/localssd/automation/internship-causal-embedding/runs/classifier/merged_workflows_all-mpnet-base-v2" / "best.pt"
SAMPLES = Path("/mnt/localssd/test_samples_milan.json")
OUT = Path("/mnt/localssd/ajo_doc_dataset.json")

# -- cross-encoder 2x (single 24-layer BERT, joint [CLS] t1 [SEP] t2 [SEP]) --
CE2X_MODEL_DIR = Path(
    "/mnt/localssd/new_base_line/internship-causal-embedding"
    "/my_work/kl/cross_encoder_2x"
)
CE_CKPT = REPO / "runs/crossencoder2x/new_ajo_ce2x_org" / "best.pt"
CE_OUT = Path("/mnt/localssd/ajo_orchestrated_workflows_flat_scored_ce2x_claude_data.json")
CE_FLAT_OUT = Path("/mnt/localssd/test_samples_milan_scored_crossencoder.json")

BATCH_SIZE = 16


@torch.no_grad()
def score_pairs(model, tokenizer, t1, t2, max_len, device, use_bf16) -> list[float]:
    """Return P(text_1 precedes text_2) for each pair, texts tokenized separately."""
    probs: list[float] = []
    from contextlib import nullcontext
    ctx_factory = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                   if use_bf16 else nullcontext)
    for s in range(0, len(t1), BATCH_SIZE):
        bt1, bt2 = t1[s:s + BATCH_SIZE], t2[s:s + BATCH_SIZE]
        enc1 = tokenizer(bt1, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        enc2 = tokenizer(bt2, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        enc1 = {k: v.to(device) for k, v in enc1.items()}
        enc2 = {k: v.to(device) for k, v in enc2.items()}
        with ctx_factory():
            logit, _, _ = model(enc1, enc2)
        probs.extend(torch.sigmoid(logit).float().cpu().tolist())
    return probs


@torch.no_grad()
def score_pairs_ce(model, tokenizer, t1, t2, max_len, device, use_bf16) -> list[float]:
    """Cross-encoder 2x: jointly tokenize ([CLS] t1 [SEP] t2 [SEP]) and return
    P(text_1 precedes text_2) = sigmoid(logit) for each pair."""
    probs: list[float] = []
    from contextlib import nullcontext
    ctx_factory = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                   if use_bf16 else nullcontext)
    for s in range(0, len(t1), BATCH_SIZE):
        bt1, bt2 = t1[s:s + BATCH_SIZE], t2[s:s + BATCH_SIZE]
        enc = tokenizer(bt1, bt2, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with ctx_factory():
            logit = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )
        probs.extend(torch.sigmoid(logit).float().cpu().tolist())
    return probs


def _score_flat(model, tokenizer, pairs, max_len, device, use_bf16,
                score_fn=score_pairs) -> list[dict]:
    t1 = [p["text1"] for p in pairs]
    t2 = [p["text2"] for p in pairs]
    probs = score_fn(model, tokenizer, t1, t2, max_len, device, use_bf16)
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


def _rank_workflow(scored_pairs: list[dict]) -> dict:
    """Given 6 scored pairs for one 3-step workflow, rank t1/t2/t3 by model scores.

    For each step label, its 'total score' is the sum of P(step -> other) across
    all pairs where it appears as text1, minus the sum where it appears as text2.
    The label with the highest net score is predicted as step 1, etc.
    """
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


AJO_SAMPLES = Path("/mnt/localssd/ajo_orchestrated_workflows_flat.json")
AJO_OUT = Path("/mnt/localssd/ajo_doc_dataset.json")

_LABELS = ["t1", "t2", "t3"]
_CORRECT_ORDER = ["t1", "t2", "t3"]
# All 6 directed pairs for 3 texts (both directions for each unordered pair)
_DIRECTED_PAIRS = [(a, b) for a in _LABELS for b in _LABELS if a != b]


def _order_three(P: dict[str, dict[str, float]]) -> list[str]:
    """Rank t1/t2/t3 using tournament wins then net precedence (mirrors order_events.py)."""
    wins = {a: 0 for a in _LABELS}
    net = {a: 0.0 for a in _LABELS}
    for i in _LABELS:
        for j in _LABELS:
            if i == j:
                continue
            pij, pji = P[i][j], P[j][i]
            net[i] += pij - pji
    for i, j in itertools.combinations(_LABELS, 2):
        if P[i][j] >= P[j][i]:
            wins[i] += 1
        else:
            wins[j] += 1
    return sorted(_LABELS, key=lambda a: (-wins[a], -net[a]))


def _score_ajo(model, tokenizer, samples: list[dict], max_len, device, use_bf16,
               score_fn=score_pairs) -> tuple[list[dict], int]:
    """For each AJO sample score all 6 directed pairs (both directions for each
    unordered pair among t1/t2/t3), then rank with tournament + net-precedence.

    `score_fn` selects the scorer: score_pairs (bi-encoder) or score_pairs_ce
    (cross-encoder 2x)."""
    # Flatten all directed pairs across all samples for a single batched call
    all_a: list[str] = []
    all_b: list[str] = []
    for sample in samples:
        for lbl_a, lbl_b in _DIRECTED_PAIRS:
            all_a.append(sample[lbl_a])
            all_b.append(sample[lbl_b])

    flat_scores = score_fn(model, tokenizer, all_a, all_b, max_len, device, use_bf16)

    results: list[dict] = []
    correct_count = 0
    n_pairs = len(_DIRECTED_PAIRS)  # 6

    for s_idx, sample in enumerate(samples):
        base = s_idx * n_pairs
        # Build score matrix P[i][j] = P(i precedes j)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pairs with the bi-encoder.")
    parser.add_argument("--three-step", action="store_true",
                        help="Input is a list-of-lists (3-step workflow format).")
    parser.add_argument("--test-ajo", action="store_true",
                        help="Score all 6 permutations of each AJO 3-step workflow "
                             "and report ordering accuracy.")
    parser.add_argument("--cross-encoder", action="store_true",
                        help="Use the CrossEncoder2x model (runs/crossencoder2x/ce2x_org) "
                             "instead of the bi-encoder. Requires --test-ajo.")
    parser.add_argument("--ce-ckpt", type=str, default=None,
                        help="Path to a CrossEncoder2x run dir (uses its best.pt) or a .pt "
                             "file. Implies --cross-encoder. native_backbone is read from the "
                             "checkpoint's cfg, so DeBERTa-v3 and stacked-BERT both load, e.g. "
                             "runs/crossencoder2x_deberta/new_aep_workflow_scrap")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to input JSON (overrides built-in default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to output JSON (overrides built-in default).")
    args = parser.parse_args()
    if args.ce_ckpt is not None:
        args.cross_encoder = True  # --ce-ckpt implies cross-encoder mode

    if args.test_ajo:
        input_path = args.input or AJO_SAMPLES
        output_path = args.output or (CE_OUT if args.cross_encoder else AJO_OUT)
    elif args.cross_encoder:
        input_path = args.input or SAMPLES
        output_path = args.output or CE_FLAT_OUT
    else:
        input_path = args.input or SAMPLES
        output_path = args.output or OUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"device={device}  bf16={use_bf16}")

    if args.cross_encoder:
        # -- CrossEncoder2x: single 24-layer BERT, joint [CLS] t1 [SEP] t2 [SEP] --
        sys.path.insert(0, str(CE2X_MODEL_DIR))
        from model_ce2x import CrossEncoder2x, get_tokenizer  # noqa: E402

        # --ce-ckpt may be a run directory (uses best.pt) or a direct .pt file.
        if args.ce_ckpt is not None:
            p = Path(args.ce_ckpt)
            ce_ckpt_path = p / "best.pt" if p.is_dir() else p
        else:
            ce_ckpt_path = CE_CKPT

        print(f"loading cross-encoder checkpoint <- {ce_ckpt_path}")
        ckpt = torch.load(ce_ckpt_path, map_location=device)
        cfg = ckpt["cfg"]
        base_model = cfg["backbone"]
        n_layers = cfg["n_layers"]
        dropout = cfg.get("dropout", 0.1)
        max_len = cfg["max_seq_len"]
        native_backbone = cfg.get("native_backbone", False)
        print(f"backbone={base_model}  n_layers={n_layers}  native={native_backbone}  "
              f"dropout={dropout}  max_seq_len={max_len}")

        tokenizer = get_tokenizer(base_model)
        model = CrossEncoder2x(backbone=base_model, n_layers=n_layers, dropout=dropout,
                               native_backbone=native_backbone).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        score_fn = score_pairs_ce
    else:
        # -- rebuild bi-encoder exactly as trained, from the checkpoint's own cfg --
        print(f"loading checkpoint <- {CKPT}")
        ckpt = torch.load(CKPT, map_location=device)
        cfg = ckpt["cfg"]
        base_model = cfg["base_model"]
        head_cfg = cfg["head_cfg"]
        n_tiers = cfg.get("n_tiers", 0) or (15 if cfg.get("tier_aux_weight", 0.0) > 0 else 0)
        max_len = cfg["max_seq_len"]
        print(f"base_model={base_model}  head={cfg.get('head')}  "
              f"n_tiers={n_tiers}  max_seq_len={max_len}")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        arch = cfg.get("architecture", "DirectionalClassifier")
        if arch == "ReasoningClassifier":
            model = ReasoningClassifier(base_model, head_cfg=head_cfg).to(device)
        else:
            model = DirectionalClassifier(base_model, head_cfg=head_cfg, n_tiers=n_tiers).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        score_fn = score_pairs

    raw = json.loads(input_path.read_text())

    if args.test_ajo:
        samples: list[dict] = raw
        enc_name = "cross-encoder 2x" if args.cross_encoder else "bi-encoder"
        print(f"loaded {len(samples)} AJO samples from {input_path}  ({enc_name})")
        print("scoring 6 directed pairs per sample (both directions for each unordered pair) ...\n")
        results, correct_count = _score_ajo(model, tokenizer, samples, max_len, device,
                                            use_bf16, score_fn=score_fn)
        print(f"\nordering accuracy: {correct_count}/{len(samples)} samples correctly ordered")

    elif args.three_step:
        # raw is a list of workflows; each workflow is a list of 6 pair dicts
        workflows: list[list[dict]] = raw
        flat_pairs = [p for wf in workflows for p in wf]
        print(f"loaded {len(workflows)} workflows ({len(flat_pairs)} pairs) from {input_path}")

        flat_scored = _score_flat(model, tokenizer, flat_pairs, max_len, device, use_bf16,
                                  score_fn=score_fn)

        # re-nest: split flat results back into per-workflow groups
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
        pairs: list[dict] = raw
        enc_name = "cross-encoder 2x" if args.cross_encoder else "bi-encoder"
        print(f"loaded {len(pairs)} pairs from {input_path}  ({enc_name})")
        results = _score_flat(model, tokenizer, pairs, max_len, device, use_bf16,
                              score_fn=score_fn)
        joint = "jointly" if args.cross_encoder else "separately"
        print(f"\n({enc_name}: texts encoded {joint}; score = P(text1 -> text2))\n")
        for r in results:
            lbl = f"{r.get('label1','?')}->{r.get('label2','?')}"
            print(f"  {lbl:10s} score={r['score']:.4f}  pred={r['pred']}")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote results to {output_path}")


if __name__ == "__main__":
    main()
