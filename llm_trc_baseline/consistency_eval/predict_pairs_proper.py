"""Score every step pair of every workflow using the correct model classes.

Uses CrossEncoder2x / CrossEncoder2xWithReasoning from model_ce2x.py
(not the simplified CrossEncoder2xNative) to properly load the reasoning model.

Run:
  cd /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    /mnt/localssd/causal-embedding-research/llm_trc_baseline/consistency_eval/predict_pairs_proper.py \
    --tag baseline --run-dir runs/crossencoder2x_deberta/v2_res_baseline

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    /mnt/localssd/causal-embedding-research/llm_trc_baseline/consistency_eval/predict_pairs_proper.py \
    --tag reasoning --run-dir runs/crossencoder2x_deberta_reasoning/v2_fixed_alpha02_ep10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from contextlib import nullcontext

import torch

MODEL_SRC = Path("/mnt/localssd/causal-embedding-research/new_base_line/internship-causal-embedding/my_work/kl/cross_encoder_2x")
sys.path.insert(0, str(MODEL_SRC))
from model_ce2x import CrossEncoder2x, CrossEncoder2xWithReasoning, get_tokenizer  # noqa: E402

BASE = Path("/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/runs")
DEFAULT_WORKFLOWS = Path("/mnt/localssd/causal-embedding-research/llm_trc_baseline/consistency_eval/workflows_curated_697.json")
DEFAULT_OUT = Path("/mnt/localssd/causal-embedding-research/llm_trc_baseline/outputs/consistency_curated")


def load_model(run_dir: Path, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    with_reasoning = cfg.get("model_type", "") == "cross_encoder_2x_with_reasoning"
    if with_reasoning:
        m = CrossEncoder2xWithReasoning(
            backbone=cfg["backbone"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
            native_backbone=cfg["native_backbone"],
            decoder_layers=cfg["decoder_layers"], max_reason_len=cfg["max_reason_len"],
        ).to(device)
    else:
        m = CrossEncoder2x(
            backbone=cfg["backbone"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
            native_backbone=cfg["native_backbone"],
        ).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    print(f"loaded {run_dir.name}: with_reasoning={with_reasoning} "
          f"epoch={ckpt['epoch']} val_acc={ckpt['val']['acc']:.4f}", flush=True)
    return m, cfg, with_reasoning


@torch.no_grad()
def score_pairs(model, tok, max_len, device, cfg, with_reasoning, pairs_a, pairs_b, batch_size=128):
    use_bf16 = device.type == "cuda" and cfg.get("bf16", True)
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    probs = []
    for k in range(0, len(pairs_a), batch_size):
        ba = pairs_a[k:k+batch_size]
        bb = pairs_b[k:k+batch_size]
        enc = tok(ba, bb, return_tensors="pt", truncation=True,
                  max_length=max_len, padding=True).to(device)
        with ctx():
            logits = model(enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"))
        if with_reasoning:
            logits = logits[0]   # (cls_logits, decoder_logits) — use cls only
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="baseline or reasoning")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workflows", default=str(DEFAULT_WORKFLOWS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    workflows = json.loads(Path(args.workflows).read_text())
    print(f"workflows: {len(workflows)}  device: {device}", flush=True)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = BASE / run_dir

    model, cfg, with_reasoning = load_model(run_dir, device)
    tok = get_tokenizer(cfg["backbone"])
    max_len = cfg["max_seq_len"]

    # Build flat list of oriented (a, b) pairs — randomised per pair to remove position bias
    rng = random.Random(args.seed)
    flat_a, flat_b, index = [], [], []
    for wi, wf in enumerate(workflows):
        steps = wf["steps"]
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                flip = rng.random() < 0.5
                if flip:
                    flat_a.append(steps[j]); flat_b.append(steps[i])
                else:
                    flat_a.append(steps[i]); flat_b.append(steps[j])
                index.append((wi, i, j, flip))

    print(f"scoring {len(flat_a)} pairs across {len(workflows)} workflows ...", flush=True)
    probs = score_pairs(model, tok, max_len, device, cfg, with_reasoning,
                        flat_a, flat_b, args.batch_size)

    # Build per-workflow edge lists
    for wf in workflows:
        wf["edges"] = []
    for (wi, i, j, flip), p in zip(index, probs):
        p_ij = (1.0 - p) if flip else p   # convert back to P(step_i before step_j)
        workflows[wi]["edges"].append({
            "i": i, "j": j, "p_ij": round(float(p_ij), 6),
            "pred_before": bool(p_ij > 0.5),
            "confidence": round(float(max(p_ij, 1.0 - p_ij)), 6),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pairs_{args.tag}.json"
    out = {"model": str(run_dir), "tag": args.tag, "seed": args.seed,
           "n_workflows": len(workflows), "n_pairs": len(flat_a), "workflows": workflows}
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
