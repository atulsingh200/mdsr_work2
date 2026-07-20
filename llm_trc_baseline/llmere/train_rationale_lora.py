"""LLMERE Phase A — rationale-augmented pairwise LoRA on final_data (self-contained).

Fine-tunes a decoder-only LLM to GENERATE A RATIONALE THEN THE VERDICT (LLMERE's core idea), using
the `reasoning` field already present in final_data as rationale supervision. Contrast with paper-1's
label-only LoRA (same backbone) to reproduce LLMERE's "rationales help" claim on our causal task.

LLMERE recipe: LoRA rank 64, lr 2e-4, cosine schedule, ~3 epochs, bf16, loss on completion tokens.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python llmere/train_rationale_lora.py \
      --model Qwen/Qwen2.5-7B-Instruct --epochs 3
Smoke:
  ... --max-train 120 --max-eval 40 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))          # llm_trc_baseline/ (for prompts.py)
import prompts as P                            # noqa: E402

CER = Path("/mnt/localssd/causal-embedding-research")
DEF = {"train": CER / "final_data/directional_train.jsonl",
       "val": CER / "final_data/directional_val.jsonl",
       "test": CER / "final_data/directional_test.jsonl"}


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
            if limit and len(rows) >= limit:
                break
    return rows


class RationaleDataset(Dataset):
    """prompt = 'explain then VERDICT'; target = the reasoning trace (ends with VERDICT: ...)."""

    def __init__(self, rows, tok, max_len=2048):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = P.build_cot_messages(r["text_1"], r["text_2"], few_shot=None)
        reason = (r.get("reasoning") or "").strip()
        if "VERDICT" not in reason.upper():
            reason += f"\nVERDICT: {'BEFORE' if int(r['label'])==1 else 'NOT_BEFORE'}"
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        target = reason + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


class Collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, ll, aa = [], [], []
        for b in batch:
            n = len(b["input_ids"]); pad = m - n
            ii.append(b["input_ids"] + [self.pad_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            aa.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(aa),
                "labels": torch.tensor(ll)}


@torch.no_grad()
def evaluate(model, tok, rows, device, max_new_tokens=256, batch_size=16, keep_samples=8):
    model.eval()
    preds, unparsed, samples = [], 0, []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        texts = [tok.apply_chat_template(P.build_cot_messages(r["text_1"], r["text_2"], None),
                                         tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(chunk, gen):
            lab = P.parse_verdict(g)
            if lab is None:
                unparsed += 1
                lab = 0
            preds.append(lab)
            if len(samples) < keep_samples:
                samples.append({"gen": g[:300], "pred": lab, "gold": int(r["label"])})
    y = np.array([int(r["label"]) for r in rows]); p = np.array(preds)
    src = np.array([r.get("source", "unknown") for r in rows])

    def _m(mask):
        yy, pp = y[mask], p[mask]
        return {"n": int(mask.sum()), "acc": float((yy == pp).mean()),
                "f1": float(f1_score(yy, pp, zero_division=0)),
                "f1_before": float(f1_score(yy, pp, pos_label=1, zero_division=0)),
                "f1_not_before": float(f1_score(yy, pp, pos_label=0, zero_division=0))}
    res = {"overall": _m(np.ones(len(y), bool)), "by_source": {},
           "unparsed": unparsed, "unparsed_rate": unparsed / max(len(y), 1), "samples": samples}
    for s in sorted(set(src.tolist())):
        res["by_source"][s] = _m(src == s)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=3)          # LLMERE: 3 (MAVEN)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)      # eff 8
    ap.add_argument("--lr", type=float, default=2e-4)         # LLMERE: 2e-4
    ap.add_argument("--lora-r", type=int, default=64)         # LLMERE: 64
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])  # LLMERE: cosine
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None, help="cap TEST rows (smoke only)")
    ap.add_argument("--val-subset", type=int, default=1200,
                    help="cap val rows used for per-epoch checkpoint selection (speed)")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=str(HERE.parent / "outputs/llmere"))
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}__rationale__lora"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    tr = load_rows(DEF["train"], args.max_train)
    va = load_rows(DEF["val"], args.max_eval or args.val_subset)
    te = load_rows(DEF["test"], args.max_eval)     # full test unless smoke
    print(f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, init_lora_weights="gaussian",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    dl = DataLoader(RationaleDataset(tr, tok, args.max_len), batch_size=args.batch_size,
                    shuffle=True, collate_fn=Collate(tok.pad_token_id))
    steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warm = int(args.warmup_frac * steps)
    sched = (get_cosine_schedule_with_warmup if args.schedule == "cosine"
             else get_linear_schedule_with_warmup)(opt, warm, steps)

    best_f1, history = -1.0, []
    for ep in range(args.epochs):
        model.train(); opt.zero_grad(); run = 0.0
        for it, b in enumerate(dl):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            loss = loss / args.grad_accum
            loss.backward(); run += loss.item() * args.grad_accum
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (it + 1) % 300 == 0:
                print(f"  ep{ep} it{it+1}/{len(dl)} loss={run/(it+1):.4f}", flush=True)
        val = evaluate(model, tok, va, device, batch_size=args.eval_batch_size)
        f1 = val["overall"]["f1"]
        history.append({"epoch": ep, "train_loss": run / len(dl), "val": val})
        print(f"[ep {ep}] train_loss={run/len(dl):.4f} val_f1={f1:.4f} "
              f"val_acc={val['overall']['acc']:.4f} unparsed={val['unparsed_rate']:.3f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir / f"{tag}_adapter")

    test = evaluate(model, tok, te, device, batch_size=args.eval_batch_size)
    rec = {"model": args.model, "method": "lora_rationale", "lora_r": args.lora_r,
           "lr": args.lr, "epochs": args.epochs, "schedule": args.schedule,
           "best_val_f1": best_f1, "history": history, "test": test}
    (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"\n[{tag}] TEST acc={test['overall']['acc']:.4f} f1={test['overall']['f1']:.4f} "
          f"unparsed={test['unparsed_rate']:.3f}")
    print(f"wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
