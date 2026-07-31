"""vLLM inference for local open-weight models on the workflow step-ordering task.

Loads one model once, sweeps shot settings (0/3/5), scores all eval rows, and writes
per-sample outputs + aggregate metrics. Follows the pattern in llm_trc_baseline/icl_eval.py.

Hyperparams match arXiv:2511.04688v2 Table 2: temperature=0.9, top_p=0.9.
Qwen3 uses thinking mode + max_tokens 2048; other models use max_tokens 512.

Example:
  .venv/bin/python run_inference.py --model Qwen/Qwen3-8B --tp 1
  .venv/bin/python run_inference.py --model mistralai/Mistral-7B-Instruct-v0.2 --tp 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metrics as M
import prompts as P

HERE = Path(__file__).parent
DATA = HERE / "data"


def load_eval() -> list[dict]:
    rows = []
    with open(DATA / "eval_prepared.jsonl") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def load_fewshot() -> dict:
    return json.load(open(DATA / "fewshot_examples.json"))


def run_setting(llm, tok, rows, few_shot, max_tokens, thinking: bool) -> list[dict]:
    from vllm import SamplingParams
    chats = [P.build_messages(r["title"], r["shuffled_steps"], few_shot) for r in rows]
    kw = {}
    if thinking:
        kw["enable_thinking"] = True
    text_prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **kw)
                    for c in chats]
    sp = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=max_tokens)
    outs = llm.generate(text_prompts, sp, use_tqdm=True)
    results = []
    for r, o in zip(rows, outs):
        gen = o.outputs[0].text
        pred = M.extract_order(gen, r["shuffled_steps"])
        results.append({"id": r["id"], "num_steps": r["num_steps"],
                        "gold_order": r["gold_order"], "pred_order": pred,
                        "raw": gen[:2000]})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 3, 5])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    thinking = False   # Qwen3 thinking not supported in vllm 0.6.6; using Qwen2.5 instead
    max_tokens = 512

    rows = load_eval()
    if args.limit:
        rows = rows[:args.limit]
    fewshot = load_fewshot()
    print(f"eval rows: {len(rows)}  model: {args.model}  thinking: {thinking}", flush=True)

    from vllm import LLM
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=False, trust_remote_code=True)
    tok = llm.get_tokenizer()

    summary = {}
    for k in args.shots:
        fs = None if k == 0 else fewshot[f"shots_{k}"]
        results = run_setting(llm, tok, rows, fs, max_tokens, thinking)
        agg = M.aggregate(results)
        tag = f"{model_tag}__k{k}"
        (out_dir / f"{tag}.json").write_text(json.dumps(
            {"model": args.model, "shots": k, "metrics": agg, "results": results}, indent=2))
        summary[tag] = agg
        print(f"[{tag}] acc={agg['acc']:.4f} nlcs={agg['nlcs']:.4f} ktau={agg['ktau']:.4f} "
              f"ned={agg['ned']:.4f} unparsed={agg['unparsed_rate']:.3f}", flush=True)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
