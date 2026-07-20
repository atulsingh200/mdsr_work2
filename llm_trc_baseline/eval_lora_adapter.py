"""Evaluate a saved LoRA adapter on the final_data test set.

Usage:
  .venv/bin/python eval_lora_adapter.py \
      --adapter outputs/lora/Qwen2.5-7B-Instruct__P__lora_adapter \
      --model Qwen/Qwen2.5-7B-Instruct \
      --prompt-type P
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import prompts as P

CER = Path("/mnt/localssd/causal-embedding-research")
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


@torch.no_grad()
def evaluate(model, tok, rows, prompt_type, device, max_new_tokens=8, batch_size=16):
    model.eval()
    preds, unparsed = [], 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        texts = [tok.apply_chat_template(
            P.build_messages(prompt_type, r["text_1"], r["text_2"], None),
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(chunk, gen):
            lab = P.parse_answer(prompt_type, g)
            if lab is None:
                unparsed += 1
                lab = 0
            preds.append(lab)
        if (i // batch_size) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(rows)} rows done...", flush=True)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to saved LoRA adapter directory")
    ap.add_argument("--model", required=True, help="Base model name or path")
    ap.add_argument("--prompt-type", required=True, choices=["P", "QA1", "QA2"])
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--out-dir", default="outputs/lora")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    model_tag = args.model.split("/")[-1]
    tag = f"{model_tag}__{args.prompt_type}__lora"

    print(f"Loading tokenizer: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"Loading base model: {args.model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)

    print(f"Loading adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    test_rows = load_rows(args.test, args.limit)
    print(f"test rows: {len(test_rows)}", flush=True)

    max_new = 16 if args.prompt_type == "QA2" else 8
    print("Running evaluation...", flush=True)
    result = evaluate(model, tok, test_rows, args.prompt_type, device, max_new, args.batch_size)

    rec = {"model": args.model, "adapter": args.adapter, "prompt_type": args.prompt_type,
           "method": "lora_eval_only", **result}

    out_path = Path(args.out_dir) / f"{tag}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2))

    print(f"\n[{tag}] TEST f1={result['overall']['f1']:.4f}  acc={result['overall']['acc']:.4f}  unparsed={result['unparsed_rate']:.4f}")
    print("by source:")
    for s, m in result["by_source"].items():
        print(f"  {s}: f1={m['f1']:.4f}  acc={m['acc']:.4f}  n={m['n']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
