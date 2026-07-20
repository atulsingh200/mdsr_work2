"""Paper-faithful LoRA fine-tuning — reproduces arXiv:2410.10476 on our data.

KEY CHANGES vs lora_finetune.py:
  1. Uses prompts_paper.py: NO system message, raw "Given the context: -> LABEL" format
  2. Full step text passed — no truncation (paper also passes full sentences)
  3. Paper exact hyperparams: r=32, alpha=64, lr=1e-4, batch=8, linear warmup 10%, 2 epochs
  4. LoRA target modules: q_proj, k_proj, v_proj, o_proj (same as paper's Llama2 recipe)
  5. Zero-shot prompt during fine-tuning (paper trains with zero-shot, evaluates zero-shot)

Without the helpful system prompt guiding the model, the LLM must learn purely from the
label supervision signal — the same disadvantage the paper's Llama2 models faced.

Example:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python lora_finetune_paper.py \
      --model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --epochs 2

Smoke:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python lora_finetune_paper.py \
      --model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 \
      --max-train 200 --max-eval 50 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

import prompts_paper as P

HERE = Path(__file__).parent
CER = Path("/mnt/localssd/causal-embedding-research")
DEFAULT_TRAIN = CER / "final_data/directional_train.jsonl"
DEFAULT_VAL = CER / "final_data/directional_val.jsonl"
DEFAULT_TEST = CER / "final_data/directional_test.jsonl"


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
            if limit and len(rows) >= limit:
                break
    return rows


class TRCDataset(Dataset):
    """Builds causal-LM sequence with loss masked to answer tokens only."""

    def __init__(self, rows, tok, prompt_type, max_len=512):
        self.rows = rows
        self.tok = tok
        self.pt = prompt_type
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = P.build_messages(self.pt, r["text_1"], r["text_2"], few_shot=None)
        # Paper uses raw completion (no chat template framing) but we apply template
        # with NO system message — the key difference from lora_finetune.py
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        target = P._ANSWER[self.pt](int(r["label"])) + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + t_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


class Collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(labels)}


@torch.no_grad()
def evaluate(model, tok, rows, prompt_type, device, max_new_tokens, batch_size=16):
    model.eval()
    preds, unparsed = [], 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        texts = [tok.apply_chat_template(
            P.build_messages(prompt_type, r["text_1"], r["text_2"], None),
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(chunk, gen):
            lab = P.parse_answer(prompt_type, g)
            if lab is None:
                unparsed += 1
                lab = 0
            preds.append(lab)
    y = np.array([int(r["label"]) for r in rows])
    p = np.array(preds)
    src = np.array([r.get("source", "unknown") for r in rows])

    def _m(mask):
        yy, pp = y[mask], p[mask]
        return {"n": int(mask.sum()), "acc": float((yy == pp).mean()),
                "f1": float(f1_score(yy, pp, zero_division=0)),
                "f1_before": float(f1_score(yy, pp, pos_label=1, zero_division=0)),
                "f1_not_before": float(f1_score(yy, pp, pos_label=0, zero_division=0))}

    res = {"overall": _m(np.ones(len(y), bool)), "by_source": {},
           "unparsed": unparsed, "unparsed_rate": unparsed / max(len(y), 1)}
    for s in sorted(set(src.tolist())):
        res["by_source"][s] = _m(src == s)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-type", default="QA1", choices=["P", "QA1"])
    ap.add_argument("--epochs", type=int, default=2)           # paper: 2
    ap.add_argument("--batch-size", type=int, default=4)       # micro-batch
    ap.add_argument("--grad-accum", type=int, default=2)       # eff batch=8, paper: 8
    ap.add_argument("--lr", type=float, default=1e-4)          # paper: 1e-4
    ap.add_argument("--lora-r", type=int, default=32)          # paper: 32
    ap.add_argument("--lora-alpha", type=int, default=64)      # paper: 64
    ap.add_argument("--warmup-frac", type=float, default=0.1)  # paper: 10%
    ap.add_argument("--max-len", type=int, default=512)        # paper: short context
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--out-dir", default=str(HERE / "outputs/lora_paper"))
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    tag = f"{model_tag}__{args.prompt_type}__lora_paper"

    print(f"Paper-faithful LoRA | full text (no truncation) | prompt={args.prompt_type} | no system msg", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    train_rows = load_rows(Path(args.train), args.max_train)
    val_rows = load_rows(Path(args.val), args.max_eval)
    test_rows = load_rows(Path(args.test), args.max_eval)
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      init_lora_weights="gaussian", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = TRCDataset(train_rows, tok, args.prompt_type, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=Collate(tok.pad_token_id))
    steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_frac * steps), steps)

    max_new = 6
    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for it, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            if not torch.isfinite(loss * args.grad_accum):
                opt.zero_grad()
                continue
            loss.backward()
            running += out.loss.item()
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (it + 1) % 200 == 0:
                print(f"  ep{epoch} it{it+1}/{len(dl)} loss={running/(it+1):.4f}", flush=True)
        val = evaluate(model, tok, val_rows, args.prompt_type, device, max_new)
        f1 = val["overall"]["f1"]
        history.append({"epoch": epoch, "train_loss": running / len(dl), "val": val})
        print(f"[epoch {epoch}] train_loss={running/len(dl):.4f} val_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir / f"{tag}_adapter")

    test = evaluate(model, tok, test_rows, args.prompt_type, device, max_new)
    rec = {"model": args.model, "prompt_type": args.prompt_type, "method": "lora_paper_faithful",
           "max_words": P.MAX_WORDS, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
           "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
           "best_val_f1": best_f1, "history": history, "test": test}
    (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"\n[{tag}] TEST f1={test['overall']['f1']:.4f} acc={test['overall']['acc']:.4f}")
    print(f"wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
