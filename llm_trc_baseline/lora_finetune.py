"""LoRA fine-tuning of a decoder-only LLM on final_data directional TRC (LLM upper bound).

Reproduces the paper's recipe (arXiv 2410.10476, Appendix A.1): zero-shot prompt format,
LoRA r=32 / alpha=64 (gaussian init), AdamW lr 1e-4, batch size 8, linear warmup 10%, causal-LM
loss on the completion tokens only. After training, evaluates on the test set with the same parser
as icl_eval (generation-based), reporting overall + per-source F1.

Example:
  .venv/bin/python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --epochs 2

Smoke:
  .venv/bin/python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 \
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

import prompts as P

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
    """Builds (prompt, target) as a single causal-LM sequence; loss masked to the target tokens.

    rationale=False (paper-1): target = the label word only.
    rationale=True  (LLMERE):  prompt asks to explain then verdict; target = the reasoning trace
                               (which already ends in 'VERDICT: BEFORE/NOT_BEFORE').
    """

    def __init__(self, rows, tok, prompt_type, max_len=1024, rationale=False):
        self.rows, self.tok, self.pt, self.max_len = rows, tok, prompt_type, max_len
        self.rationale = rationale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.rationale:
            msgs = P.build_cot_messages(r["text_1"], r["text_2"], few_shot=None)
            reason = (r.get("reasoning") or "").strip()
            if "VERDICT" not in reason.upper():          # guarantee the verdict is present
                reason += f"\nVERDICT: {'BEFORE' if int(r['label'])==1 else 'NOT_BEFORE'}"
            target = reason + self.tok.eos_token
        else:
            msgs = P.build_messages(self.pt, r["text_1"], r["text_2"], few_shot=None)
            target = P._ANSWER[self.pt](int(r["label"])) + self.tok.eos_token
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
def evaluate(model, tok, rows, prompt_type, device, max_new_tokens, batch_size=16, rationale=False):
    model.eval()
    preds, unparsed = [], 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        if rationale:
            texts = [tok.apply_chat_template(
                P.build_cot_messages(r["text_1"], r["text_2"], None),
                tokenize=False, add_generation_prompt=True) for r in chunk]
        else:
            texts = [tok.apply_chat_template(
                P.build_messages(prompt_type, r["text_1"], r["text_2"], None),
                tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(chunk, gen):
            lab = P.parse_verdict(g) if rationale else P.parse_answer(prompt_type, g)
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
    ap.add_argument("--rationale", action="store_true",
                    help="LLMERE mode: train to generate the reasoning trace then the VERDICT.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)      # micro-batch
    ap.add_argument("--grad-accum", type=int, default=2)      # eff batch = 4*2 = 8 (paper: 8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--cosine", action="store_true", help="cosine schedule (LLMERE uses cosine).")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--out-dir", default=str(HERE / "outputs/lora"))
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    mode = "rationale" if args.rationale else args.prompt_type
    tag = f"{model_tag}__{mode}__lora"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left-pad for generation

    train_rows = load_rows(Path(args.train), args.max_train)
    val_rows = load_rows(Path(args.val), args.max_eval)
    test_rows = load_rows(Path(args.test), args.max_eval)
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # required for grad-checkpointing + PEFT
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      init_lora_weights="gaussian", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = TRCDataset(train_rows, tok, args.prompt_type, args.max_len, rationale=args.rationale)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=Collate(tok.pad_token_id))
    steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup = int(args.warmup_frac * steps)
    if args.cosine:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(opt, warmup, steps)
    else:
        sched = get_linear_schedule_with_warmup(opt, warmup, steps)

    max_new = 256 if args.rationale else (16 if args.prompt_type == "QA2" else 8)
    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for it, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (it + 1) % 200 == 0:
                print(f"  ep{epoch} it{it+1}/{len(dl)} loss={running/(it+1):.4f}", flush=True)
        val = evaluate(model, tok, val_rows, args.prompt_type, device, max_new, rationale=args.rationale)
        f1 = val["overall"]["f1"]
        history.append({"epoch": epoch, "train_loss": running / len(dl), "val": val})
        print(f"[epoch {epoch}] train_loss={running/len(dl):.4f} val_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir / f"{tag}_adapter")

    test = evaluate(model, tok, test_rows, args.prompt_type, device, max_new, rationale=args.rationale)
    rec = {"model": args.model, "prompt_type": args.prompt_type, "method": "lora",
           "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lr": args.lr,
           "epochs": args.epochs, "batch_size": args.batch_size,
           "best_val_f1": best_f1, "history": history, "test": test}
    (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"\n[{tag}] TEST f1={test['overall']['f1']:.4f} acc={test['overall']['acc']:.4f}")
    print(f"wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
