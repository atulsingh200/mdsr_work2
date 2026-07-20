"""Fast LoRA eval using vLLM: merges adapter into base model, saves merged weights,
then runs vLLM inference over the full test set in one shot.

Usage:
  .venv/bin/python eval_lora_vllm.py \
      --adapter outputs/lora/Qwen2.5-7B-Instruct__P__lora_adapter \
      --model Qwen/Qwen2.5-7B-Instruct \
      --prompt-type P
"""

from __future__ import annotations

import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

import prompts as P

CER = Path("/mnt/localssd/causal-embedding-research")
DEFAULT_TEST = CER / "final_data/directional_test.jsonl"
MERGED_DIR = Path("/tmp/opencode/merged_lora_model")


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


def merge_and_save(base_model, adapter_path, merged_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

    print(f"Loading + merging adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}", flush=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(merged_dir)
    print("Merge done.", flush=True)


def compute_metrics(rows, preds, unparsed):
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
           "unparsed": int(unparsed), "unparsed_rate": unparsed / max(len(y), 1)}
    for s in sorted(set(src.tolist())):
        res["by_source"][s] = _m(src == s)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-type", required=True, choices=["P", "QA1", "QA2"])
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--out-dir", default="outputs/lora")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-merge", action="store_true",
                    help="Skip merge step if merged model already exists at MERGED_DIR")
    args = ap.parse_args()

    model_tag = args.model.split("/")[-1]
    tag = f"{model_tag}__{args.prompt_type}__lora"
    merged_dir = Path(f"/tmp/opencode/merged_{tag}")

    # Step 1: merge adapter into base weights (on CPU, no GPU needed)
    if not args.skip_merge or not merged_dir.exists():
        merge_and_save(args.model, args.adapter, merged_dir)
    else:
        print(f"Skipping merge, using existing {merged_dir}", flush=True)

    # Step 2: load merged model with vLLM and run fast inference
    from vllm import LLM, SamplingParams

    print(f"Loading merged model into vLLM (tp={args.tp})...", flush=True)
    llm = LLM(model=str(merged_dir), tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=False, trust_remote_code=True)
    tok = llm.get_tokenizer()

    rows = load_rows(args.test, args.limit)
    print(f"test rows: {len(rows)}", flush=True)

    max_tokens = 16 if args.prompt_type == "QA2" else 8
    chats = [P.build_messages(args.prompt_type, r["text_1"], r["text_2"], None) for r in rows]
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    print("Running vLLM inference...", flush=True)
    outputs = llm.generate(prompts, sp)

    preds, unparsed, samples = [], 0, []
    for r, o in zip(rows, outputs):
        gen = o.outputs[0].text
        lab = P.parse_answer(args.prompt_type, gen)
        if lab is None:
            unparsed += 1
            lab = 0
        preds.append(lab)
        if len(samples) < 8:
            samples.append({"gen": gen[:120], "pred": lab, "gold": int(r["label"])})

    result = compute_metrics(rows, preds, unparsed)
    result["samples"] = samples

    rec = {"model": args.model, "adapter": args.adapter, "prompt_type": args.prompt_type,
           "method": "lora_vllm_eval", **result}

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
