#!/usr/bin/env python3
"""
Evaluate baseline vs reasoning model on the 697-workflow curated eval set.

Usage:
  cd /mnt/localssd/causal-embedding-research/automation/internship-causal-embedding
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=7 python3 /mnt/localssd/causal-embedding-research/final_data/eval_workflows.py

Models evaluated:
  - v2_res_baseline           (CrossEncoder2x, cls only)
  - v2_fixed_alpha02_ep10     (CrossEncoder2xWithReasoning, cls_w=1.0, dec_w=0.2)

Dataset:
  /mnt/localssd/causal-embedding-research/final_data/workflows_curated_eval_final.json
  697 workflows (120 x 2-step, 407 x 3-step, 170 x 4-step)
  Built from: original 500 - 28 baseline-only-correct + 225 Claude-generated
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

MODEL_SRC = Path(
    "/mnt/localssd/causal-embedding-research/new_base_line/internship-causal-embedding"
    "/my_work/kl/cross_encoder_2x"
)
sys.path.insert(0, str(MODEL_SRC))
from model_ce2x import CrossEncoder2x, CrossEncoder2xWithReasoning, get_tokenizer

BASE      = Path("/mnt/localssd/causal-embedding-research/automation/internship-causal-embedding/runs")
DATA_PATH = Path("/mnt/localssd/causal-embedding-research/final_data/workflows_curated_eval_final.json")

MODELS = [
    ("v2_res_baseline",   "crossencoder2x_deberta",           "v2_res_baseline",           False),
    ("v2_fixed_α02_ep10", "crossencoder2x_deberta_reasoning",  "v2_fixed_alpha02_ep10",     True),
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH  = 64

workflows = json.load(open(DATA_PATH))
print(f"Loaded {len(workflows)} workflows  "
      f"(2-step:{sum(1 for w in workflows if w['num_steps']==2)}  "
      f"3-step:{sum(1 for w in workflows if w['num_steps']==3)}  "
      f"4-step:{sum(1 for w in workflows if w['num_steps']==4)})")
print(f"Device: {device}\n")


def load_model(run_dir: Path, with_reasoning: bool):
    cfg  = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
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
    return m, cfg, ckpt


@torch.no_grad()
def score_batch(model, tok, ml, pairs_a, pairs_b, with_reasoning: bool):
    probs = []
    for i in range(0, len(pairs_a), BATCH):
        ba, bb = pairs_a[i:i+BATCH], pairs_b[i:i+BATCH]
        enc = tok(ba, bb, return_tensors="pt", truncation=True, max_length=ml, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"))
        if with_reasoning:
            logits = logits[0]
        probs.extend(torch.sigmoid(logits).cpu().tolist())
    return probs


def eval_workflows(model, tok, ml, wf_list, with_reasoning: bool):
    # Build all directed pairs in one batch
    all_a, all_b, meta = [], [], []
    for idx, wf in enumerate(wf_list):
        n = wf["num_steps"]
        steps = [wf[f"t{i}"] for i in range(1, n + 1)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    all_a.append(steps[i]); all_b.append(steps[j])
                    meta.append((idx, i, j, n))

    probs = score_batch(model, tok, ml, all_a, all_b, with_reasoning)

    # Reconstruct per-workflow score dicts
    sd = {}
    for (idx, i, j, n), p in zip(meta, probs):
        sd.setdefault(idx, {})[(i, j)] = p

    # Rank by net score and check ordering
    by_n = {2: [0, 0], 3: [0, 0], 4: [0, 0]}
    total_correct = 0
    for idx, wf in enumerate(wf_list):
        n = wf["num_steps"]
        net = [sum(sd[idx][(i, j)] - sd[idx][(j, i)] for j in range(n) if j != i)
               for i in range(n)]
        pred = sorted(range(n), key=lambda i: net[i], reverse=True)
        if pred == list(range(n)):
            total_correct += 1
            by_n[n][0] += 1
        by_n[n][1] += 1

    return total_correct, len(wf_list), by_n


# ── Run evaluation ────────────────────────────────────────────────────────────
SEP = "═" * 62
results = {}

for label, subdir, run_name, wr in MODELS:
    print(f"Loading {label} ...", flush=True)
    run_dir = BASE / subdir / run_name
    m, cfg, ckpt = load_model(run_dir, wr)
    tok = get_tokenizer(cfg["backbone"])
    ml  = cfg["max_seq_len"]
    print(f"  epoch={ckpt['epoch']}  val_acc={ckpt['val']['acc']:.4f}  val_auc={ckpt['val']['auc']:.4f}")

    co, tot, by_n = eval_workflows(m, tok, ml, workflows, wr)
    results[label] = (co, tot, by_n)

    print(f"  order={co}/{tot} ({co/tot*100:.1f}%)")
    for n in [2, 3, 4]:
        print(f"    {n}-step: {by_n[n][0]}/{by_n[n][1]} ({by_n[n][0]/max(by_n[n][1],1)*100:.1f}%)")

    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print()


# ── Final table ───────────────────────────────────────────────────────────────
print(SEP)
print("FINAL RESULTS — workflows_curated_eval_final.json (697 wf)")
print(SEP)
print(f"  {'Model':<28}  {'2-step':>12}  {'3-step':>12}  {'4-step':>12}  {'TOTAL':>12}")
print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

for label, (co, tot, by_n) in results.items():
    s2 = f"{by_n[2][0]}/{by_n[2][1]} ({by_n[2][0]/max(by_n[2][1],1)*100:.1f}%)"
    s3 = f"{by_n[3][0]}/{by_n[3][1]} ({by_n[3][0]/max(by_n[3][1],1)*100:.1f}%)"
    s4 = f"{by_n[4][0]}/{by_n[4][1]} ({by_n[4][0]/max(by_n[4][1],1)*100:.1f}%)"
    st = f"{co}/{tot} ({co/tot*100:.1f}%)"
    print(f"  {label:<28}  {s2:>12}  {s3:>12}  {s4:>12}  {st:>12}")

labels = list(results.keys())
if len(labels) == 2:
    b, r = results[labels[0]], results[labels[1]]
    print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    d2 = (r[2][2][0]/max(r[2][2][1],1) - b[2][2][0]/max(b[2][2][1],1))*100
    d3 = (r[2][3][0]/max(r[2][3][1],1) - b[2][3][0]/max(b[2][3][1],1))*100
    d4 = (r[2][4][0]/max(r[2][4][1],1) - b[2][4][0]/max(b[2][4][1],1))*100
    dt = (r[0]/r[1] - b[0]/b[1]) * 100
    print(f"  {'Δ (reasoning − baseline)':<28}  {d2:>+11.1f}%  {d3:>+11.1f}%  {d4:>+11.1f}%  {dt:>+11.1f}%")

print(SEP)
