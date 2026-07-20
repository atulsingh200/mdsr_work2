"""Pairwise O(n^2) baseline for Phase B: score a cross-encoder on the per-event test docs.

For each reconstructed test document, score every unordered event pair with the CE (one forward per
pair, O(n^2)) and take the higher-probability direction as the predicted directional relation.
Report micro-P/R/F1 vs the gold successor relations (same metric as the O(n) LLM) and the forward
count, so Table B compares LLM per-event O(n) against encoder pairwise O(n^2) on identical docs.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python llmere/eval_crossencoder_perevent.py \
      --run-dir <ce_dir> --tag ce_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from eval_encoder_final import CrossEncoder2xNative      # noqa: E402
from consistency_eval.predict_pairs import score_pairs   # noqa: E402  (batched CE scorer)

DATA = HERE.parent / "outputs/llmere/perevent/test.jsonl"
OUT = HERE.parent / "outputs/llmere"


def load_docs():
    docs = {}
    for ln in open(DATA):
        r = json.loads(ln)
        if r["doc_id"] not in docs:
            docs[r["doc_id"]] = r["events"]      # [{tag,text,gold_rank}]
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = json.loads((Path(args.run_dir) / "config.json").read_text())
    backbone = cfg["backbone"]; max_len = int(cfg.get("max_seq_len", 512))
    use_bf16 = device.type == "cuda"
    tok = AutoTokenizer.from_pretrained(backbone)
    model = CrossEncoder2xNative(backbone, dropout=float(cfg.get("dropout", 0.1)))
    ck = torch.load(Path(args.run_dir) / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()

    docs = load_docs()
    # build all unordered pairs (by tag) per doc; score P(text_a before text_b)
    flat, index = [], []
    for did, events in docs.items():
        by_tag = {e["tag"]: e for e in events}
        tags = sorted(by_tag)
        for ai in range(len(tags)):
            for bi in range(ai + 1, len(tags)):
                a, b = tags[ai], tags[bi]
                flat.append((by_tag[a]["text"], by_tag[b]["text"]))
                index.append((did, a, b, by_tag[a]["gold_rank"], by_tag[b]["gold_rank"]))
    print(f"docs={len(docs)}  pairs(O(n^2) forwards)={len(flat)}", flush=True)
    probs = score_pairs(model, tok, max_len, device, use_bf16, flat, batch_size=256)

    pred, gold = defaultdict(set), defaultdict(set)
    for (did, a, b, ga, gb), p in zip(index, probs):
        # predicted directional relation
        if p > 0.5:
            pred[did].add((a, b))
        else:
            pred[did].add((b, a))
        gold[did].add((a, b) if ga < gb else (b, a))
    tp = fp = fn = 0
    for did in gold:
        P, G = pred[did], gold[did]
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rec_out = {"tag": args.tag, "run_dir": args.run_dir, "method": "pairwise_On2",
               "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
               "n_docs": len(gold), "pairs_On2": len(flat)}
    (OUT / f"crossencoder_perevent_{args.tag}.json").write_text(json.dumps(rec_out, indent=2))
    print(f"[{args.tag}] pairwise O(n^2) relation F1={f1:.4f} P={prec:.4f} R={rec:.4f}  "
          f"forwards={len(flat)}")


if __name__ == "__main__":
    main()
