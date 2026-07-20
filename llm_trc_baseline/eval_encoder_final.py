"""Evaluate the already-trained DeBERTa CrossEncoder2x on final_data test — EVAL ONLY.

The encoder baseline (runs/crossencoder2x_deberta/v2_res_baseline/best.pt) was trained on
`aep_ajo_procedural_workflow_v2_res` (its own test = 4,984 rows). To compare apples-to-apples
with the LLMs we must score it on OUR shared test set:
    /mnt/localssd/causal-embedding-research/final_data/directional_test.jsonl  (4,801 rows)

We reuse the exact model class (CrossEncoder2x) and the dataset/collate used at train time, load
best.pt, and report overall micro-F1 / acc / AUC plus per-`source` F1 (procedural / aep / ajo).
No weights are trained or modified.

Run:
  .venv/bin/python eval_encoder_final.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# ── repo paths (dataset/collate live in the training repo) ───────────────────────
CER = Path("/mnt/localssd/causal-embedding-research")
CE_PKG_ROOT = CER / "automation" / "internship-causal-embedding"          # src.classifier.*
sys.path.insert(0, str(CE_PKG_ROOT))

from src.classifier.crossencoder.data import CECollate, CEPairDataset       # noqa: E402


class CrossEncoder2xNative(nn.Module):
    """Faithful re-implementation of model_ce2x.CrossEncoder2x for native_backbone=True.

    Identical submodule names (`backbone`, `dropout`, `classifier`) and forward path so the
    trained best.pt state_dict loads exactly. Inlined here only to avoid the upstream file's
    `AutoModel.from_pretrained(..., dtype=...)` call, which is a transformers-5.x kwarg not
    accepted by the pinned transformers 4.47.1.
    """

    def __init__(self, backbone: str, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone, torch_dtype=torch.float32)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls)).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(
        CER / "automation/internship-causal-embedding/runs/crossencoder2x_deberta/v2_res_baseline"))
    ap.add_argument("--test-jsonl", default=str(CER / "final_data/directional_test.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent
                                         / "outputs/encoder/deberta_ce2x_on_final_test.json"))
    ap.add_argument("--step-key", default="step")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = json.loads((run_dir / "config.json").read_text())
    backbone = cfg["backbone"]
    max_len = int(cfg.get("max_seq_len", 512))
    native = bool(cfg.get("native_backbone", True))
    n_layers = int(cfg.get("n_layers", 24))
    use_bf16 = device.type == "cuda" and bool(cfg.get("bf16", True))
    print(f"backbone={backbone}  max_len={max_len}  native={native}  bf16={use_bf16}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    ds = CEPairDataset(Path(args.test_jsonl), step_key=args.step_key)
    print(f"test rows: {len(ds)}", flush=True)
    collate = CECollate(tokenizer, max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                    num_workers=4, pin_memory=(device.type == "cuda"))

    assert native, "checkpoint config is not native_backbone; this eval only supports native."
    model = CrossEncoder2xNative(backbone=backbone, dropout=float(cfg.get("dropout", 0.1)))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        print(f"WARN state_dict mismatch missing={missing} unexpected={unexpected}", flush=True)
    model.to(device).eval()
    print(f"loaded best.pt (epoch {ckpt.get('epoch')})", flush=True)

    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    all_probs, all_labels, all_idx = [], [], []
    with torch.no_grad():
        for batch in dl:
            enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            with ctx():
                logits = model(enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"))
            all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_idx.extend(batch["idx"])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs > 0.5).astype(int)

    def _metrics(mask):
        p, y, pr = preds[mask], labels[mask], probs[mask]
        m = {"n": int(mask.sum()), "acc": float((p == y).mean()),
             "f1": float(f1_score(y, p, zero_division=0))}
        try:
            m["auc"] = float(roc_auc_score(y, pr)) if len(set(y)) > 1 else float("nan")
        except Exception:
            m["auc"] = float("nan")
        # per-class F1 (BEFORE=1, NOT_BEFORE=0)
        m["f1_before"] = float(f1_score(y, p, pos_label=1, zero_division=0))
        m["f1_not_before"] = float(f1_score(y, p, pos_label=0, zero_division=0))
        return m

    full = np.ones(len(labels), dtype=bool)
    result = {"model": "deberta-v3-large_crossencoder2x", "run_dir": str(run_dir),
              "test_jsonl": args.test_jsonl, "overall": _metrics(full), "by_source": {}}

    src = np.array([ds.rows[i].get("source", "unknown") for i in all_idx])
    for s in sorted(set(src.tolist())):
        result["by_source"][s] = _metrics(src == s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
