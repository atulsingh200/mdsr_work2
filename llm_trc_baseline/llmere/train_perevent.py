"""LLMERE Phase B — per-event O(n) extraction: LoRA train + eval (self-contained).

Fine-tunes Qwen2.5-7B (LoRA) on the per-event queries built by build_perevent_docs.py, then evaluates
on the test docs: one generation per event (O(n) queries/doc), parse the extracted 'AFTER' set,
reconstruct directional relations, and report micro-P/R/F1 vs gold + inference efficiency
(O(n) forwards/doc vs the pairwise O(n^2)).

Run:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python llmere/train_perevent.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
DATA = HERE.parent / "outputs/llmere/perevent"
OUT = HERE.parent / "outputs/llmere"
SYS = "You are an expert at ordering the steps of a procedure."


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


class PEDataset(Dataset):
    def __init__(self, rows, tok, max_len=2048):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        target = r["target"] + self.tok.eos_token
        p = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = (p + t)[: self.max_len]
        lab = ([-100] * len(p) + t)[: self.max_len]
        return {"input_ids": ids, "labels": lab}


class Collate:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, b):
        m = max(len(x["input_ids"]) for x in b)
        ii, ll, aa = [], [], []
        for x in b:
            n = len(x["input_ids"]); pad = m - n
            ii.append(x["input_ids"] + [self.pad] * pad)
            ll.append(x["labels"] + [-100] * pad)
            aa.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(aa),
                "labels": torch.tensor(ll)}


_AFTER_RE = re.compile(r"AFTER\s*:\s*(.*)", re.I)
_TAG_RE = re.compile(r"<?e(\d+)>?")


def parse_after(gen):
    m = _AFTER_RE.search(gen)
    seg = m.group(1) if m else gen
    seg = seg.split("Reasoning")[0]
    if "none" in seg.lower():
        return set()
    return {int(x) for x in _TAG_RE.findall(seg)}


@torch.no_grad()
def evaluate(model, tok, rows, device, max_new_tokens=128, batch_size=16):
    model.eval()
    # predicted directional relations per doc: (query_tag -> after_tag)
    pred_rel, gold_rel = defaultdict(set), defaultdict(set)
    docs_nevents = {}
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(chunk, gen):
            d = r["doc_id"]; q = r["query_tag"]
            docs_nevents[d] = r["n_events"]
            for t in parse_after(g):
                pred_rel[d].add((q, t))
            for t in r["gold_after_tags"]:
                gold_rel[d].add((q, t))
    tp = fp = fn = 0
    for d in gold_rel:
        P, G = pred_rel.get(d, set()), gold_rel[d]
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # efficiency: per-event O(n) queries vs pairwise O(n^2)
    on = sum(n for n in docs_nevents.values())
    on2 = sum(n * (n - 1) // 2 for n in docs_nevents.values())
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "n_docs": len(gold_rel), "queries_On": on, "pairs_On2": on2,
            "speedup_x": round(on2 / on, 2) if on else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=24)
    args = ap.parse_args()

    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}__perevent__lora"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    tr = load(DATA / "train.jsonl")[: args.max_train] if args.max_train else load(DATA / "train.jsonl")
    te = load(DATA / "test.jsonl")
    print(f"train_queries={len(tr)} test_queries={len(te)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, init_lora_weights="gaussian",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    dl = DataLoader(PEDataset(tr, tok, args.max_len), batch_size=args.batch_size, shuffle=True,
                    collate_fn=Collate(tok.pad_token_id))
    steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    for ep in range(args.epochs):
        model.train(); opt.zero_grad(); run = 0.0
        for it, b in enumerate(dl):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            if not torch.isfinite(loss):          # skip bad batch before it corrupts params
                opt.zero_grad(); continue
            loss = loss / args.grad_accum
            loss.backward(); run += loss.item() * args.grad_accum
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (it + 1) % 300 == 0:
                print(f"  ep{ep} it{it+1}/{len(dl)} loss={run/(it+1):.4f}", flush=True)
        print(f"[ep {ep}] train_loss={run/len(dl):.4f}", flush=True)

    res = evaluate(model, tok, te, device, batch_size=args.eval_batch_size)
    rec = {"model": args.model, "method": "perevent_On", **res}
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    model.save_pretrained(OUT / f"{tag}_adapter")
    print(f"\n[{tag}] relation F1={res['f1']:.4f} P={res['precision']:.4f} R={res['recall']:.4f} "
          f"| O(n) queries={res['queries_On']} vs O(n^2) pairs={res['pairs_On2']} "
          f"({res['speedup_x']}x fewer forwards)")
    print(f"wrote {OUT / f'{tag}.json'}")


if __name__ == "__main__":
    main()
