"""Score every step pair of every workflow with a cross-encoder checkpoint.

For each workflow with gold-ordered steps [s0, s1, ..., s_{N-1}] (gold: s_i before s_j iff i<j),
we run the CE on every unordered pair {i,j}. To neutralize any first-position bias, each pair is
presented to the model in a RANDOMIZED orientation (seeded per pair); we then convert back to the
canonical quantity  p_ij = P(step_i before step_j)  and record:
    predicted_before : p_ij > 0.5   (model thinks i comes before j)
    confidence        : max(p_ij, 1 - p_ij)   (prob of the predicted label)

Output JSON: {"model":..., "run_dir":..., "workflows":[{id, source, origin, n_steps,
              edges:[{i, j, p_ij, pred_before, confidence}]}]}

Reuses the inline CrossEncoder2xNative loader (works for the plain baseline AND the reasoning
checkpoint via strict=False, which drops the unused reasoning-decoder weights at inference).

Run:
  .venv/bin/python consistency_eval/predict_pairs.py --run-dir <ckpt_dir> --tag baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))            # llm_trc_baseline/
from eval_encoder_final import CrossEncoder2xNative   # noqa: E402


def load_model(run_dir: Path, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    backbone = cfg["backbone"]
    max_len = int(cfg.get("max_seq_len", 512))
    use_bf16 = device.type == "cuda" and bool(cfg.get("bf16", True))
    tok = AutoTokenizer.from_pretrained(backbone)
    model = CrossEncoder2xNative(backbone=backbone, dropout=float(cfg.get("dropout", 0.1)))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    # missing should be empty (backbone+classifier all present); unexpected = reasoning-decoder keys.
    kept = [k for k in unexpected if not (k.startswith("reason") or k.startswith("lm_bias"))]
    assert not missing, f"missing keys: {missing[:5]}"
    assert not kept, f"unexpected non-decoder keys: {kept[:5]}"
    model.to(device).eval()
    print(f"loaded {run_dir.name}: backbone={backbone} max_len={max_len} "
          f"bf16={use_bf16} (ignored {len(unexpected)} reasoning-decoder tensors)", flush=True)
    return model, tok, max_len, use_bf16


@torch.no_grad()
def score_pairs(model, tok, max_len, device, use_bf16, pairs_ab, batch_size=128):
    """pairs_ab: list of (text_a, text_b). Returns P(text_a before text_b) per pair."""
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    probs = []
    for k in range(0, len(pairs_ab), batch_size):
        chunk = pairs_ab[k:k + batch_size]
        enc = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                  truncation="longest_first", max_length=max_len, return_tensors="pt").to(device)
        with ctx():
            logits = model(enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"))
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True, help="short name, e.g. baseline / reasoning")
    ap.add_argument("--workflows", default=str(HERE / "workflows_eval.json"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "outputs/consistency"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    workflows = json.loads(Path(args.workflows).read_text())
    model, tok, max_len, use_bf16 = load_model(Path(args.run_dir), device)

    # Build a flat list of oriented (text_a, text_b) pairs; remember the flip per pair.
    rng = random.Random(args.seed)
    flat, index = [], []            # index[k] = (wf_idx, i, j, flipped)
    for wi, wf in enumerate(workflows):
        steps = wf["steps"]
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                flip = rng.random() < 0.5
                if flip:
                    flat.append((steps[j], steps[i]))     # model sees (j, i)
                else:
                    flat.append((steps[i], steps[j]))     # model sees (i, j)
                index.append((wi, i, j, flip))

    print(f"scoring {len(flat)} pairs across {len(workflows)} workflows ...", flush=True)
    probs = score_pairs(model, tok, max_len, device, use_bf16, flat, args.batch_size)

    for wf in workflows:
        wf["edges"] = []
    for (wi, i, j, flip), p in zip(index, probs):
        # p = P(shown_a before shown_b). Convert to p_ij = P(step_i before step_j).
        p_ij = (1.0 - p) if flip else p
        workflows[wi]["edges"].append({
            "i": i, "j": j, "p_ij": round(float(p_ij), 6),
            "pred_before": bool(p_ij > 0.5),
            "confidence": round(float(max(p_ij, 1.0 - p_ij)), 6),
        })

    out = {"model": str(Path(args.run_dir)), "tag": args.tag, "seed": args.seed,
           "n_workflows": len(workflows), "n_pairs": len(flat), "workflows": workflows}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pairs_{args.tag}.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
