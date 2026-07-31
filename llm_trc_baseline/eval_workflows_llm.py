#!/usr/bin/env python3
"""
Evaluate LLM models (finetuned LoRA adapters + ICL) on the 697-workflow ordering task.

Method: same net-score ranking as the encoder model, but scores come from the
LLM's YES-token logit probability for the QA1 prompt:
  P(step_A before step_B) = softmax([logit_YES, logit_NO])[YES]

For each workflow, score all directed pairs (i→j and j→i), rank by net score,
check if recovered order == ground truth t1 < t2 < ... < tN.

YES and NO are confirmed single tokens for all Qwen2.5 and Phi-3.5 models.

Usage examples:
  # Finetuned LoRA model (merge + vLLM):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_workflows_llm.py \\
      --adapter outputs/lora/Qwen2.5-7B-Instruct__QA1__lora_adapter \\
      --base-model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --tp 4

  # ICL (no adapter, just base model):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_workflows_llm.py \\
      --base-model Qwen/Qwen2.5-7B-Instruct --prompt-type QA1 --shots 0 --tp 4
"""
from __future__ import annotations
import argparse, json, sys, shutil
from pathlib import Path
import numpy as np

CER      = Path("/mnt/localssd/causal-embedding-research")
WF_PATH  = CER / "final_data/workflows_curated_eval_final.json"
TRAIN    = CER / "final_data/directional_train.jsonl"
HF_HOME  = Path("/mnt/localssd/.cache/huggingface")

sys.path.insert(0, str(Path(__file__).parent))
import prompts as P


# ── helpers ────────────────────────────────────────────────────────────────────
def load_workflows():
    wfs = json.load(open(WF_PATH))
    # build all directed pairs per workflow
    records = []   # {wf_id, wf_title, n, step_i, step_j, i, j, text_1, text_2}
    for wf in wfs:
        n = wf["num_steps"]
        steps = [wf[f"t{k}"] for k in range(1, n+1)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    records.append({
                        "wf_id": wf["id"], "wf_title": wf["title"],
                        "wf_source": wf.get("source",""), "n": n,
                        "i": i, "j": j,
                        "text_1": steps[i], "text_2": steps[j],
                    })
    return wfs, records


def get_yes_no_ids(tokenizer, model_name):
    """Return (yes_id, no_id) single-token ids for this tokenizer."""
    model_lower = model_name.lower()
    if "phi" in model_lower:
        yes_candidates = ["YES", "Yes"]
        no_candidates  = ["NO",  "No"]
    else:
        yes_candidates = ["YES", "Yes"]
        no_candidates  = ["NO",  "No"]

    def _single(cands):
        for c in cands:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0], c
        raise ValueError(f"No single-token candidate found among {cands}")

    yes_id, yes_str = _single(yes_candidates)
    no_id,  no_str  = _single(no_candidates)
    print(f"  YES token: '{yes_str}' -> id={yes_id}")
    print(f"  NO  token: '{no_str}'  -> id={no_id}")
    return yes_id, no_id


def merge_adapter(base_model, adapter_path, merged_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"  Loading base {base_model} on CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    print(f"  Merging adapter {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(merged_dir)
    print(f"  Saved merged model to {merged_dir}", flush=True)


def score_pairs_vllm(llm, tokenizer, records, prompt_type, yes_id, no_id,
                     few_shot, batch_size=512):
    """Score all pairs: return list of P(YES) for each record."""
    from vllm import SamplingParams

    # Build prompts
    all_prompts = []
    for r in records:
        msgs = P.build_messages(prompt_type, r["text_1"], r["text_2"], few_shot)
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        all_prompts.append(txt)

    # Request logprobs for the first generated token
    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        logprobs=20)   # top-20 logprobs

    print(f"  Scoring {len(all_prompts)} pairs...", flush=True)
    outputs = llm.generate(all_prompts, sp, use_tqdm=True)

    probs = []
    for o in outputs:
        lps = o.outputs[0].logprobs
        if lps and lps[0]:
            top = lps[0]  # dict token_id -> Logprob
            # get logits for YES and NO
            lp_yes = top[yes_id].logprob if yes_id in top else -100.0
            lp_no  = top[no_id].logprob  if no_id  in top else -100.0
            # softmax of just these two
            m = max(lp_yes, lp_no)
            e_yes = np.exp(lp_yes - m); e_no = np.exp(lp_no - m)
            p_yes = e_yes / (e_yes + e_no)
        else:
            p_yes = 0.5  # fallback
        probs.append(float(p_yes))
    return probs


def evaluate_ordering(wfs, records, probs):
    """Given P(YES) for each directed pair, rank steps and compute ordering accuracy."""
    # Build score dict: {wf_id: {(i,j): p}}
    sd = {}
    for r, p in zip(records, probs):
        sd.setdefault(r["wf_id"], {})[(r["i"], r["j"])] = p

    by_n = {2:[0,0], 3:[0,0], 4:[0,0]}
    total_co = 0
    for wf in wfs:
        wid = wf["id"]; n = wf["num_steps"]
        if wid not in sd: continue
        net = [sum(sd[wid][(i,j)] - sd[wid][(j,i)] for j in range(n) if j!=i) for i in range(n)]
        pred = sorted(range(n), key=lambda i: net[i], reverse=True)
        if pred == list(range(n)):
            total_co += 1; by_n[n][0] += 1
        by_n[n][1] += 1

    total = sum(v[1] for v in by_n.values())
    print(f"  2-step: {by_n[2][0]}/{by_n[2][1]} ({by_n[2][0]/max(by_n[2][1],1)*100:.1f}%)")
    print(f"  3-step: {by_n[3][0]}/{by_n[3][1]} ({by_n[3][0]/max(by_n[3][1],1)*100:.1f}%)")
    print(f"  4-step: {by_n[4][0]}/{by_n[4][1]} ({by_n[4][0]/max(by_n[4][1],1)*100:.1f}%)")
    print(f"  TOTAL : {total_co}/{total} ({total_co/total*100:.1f}%)")
    return {"by_n": {str(n): {"correct": by_n[n][0], "total": by_n[n][1]} for n in [2,3,4]},
            "total_correct": total_co, "total": total,
            "order_acc": round(total_co/total, 4)}


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (None = ICL/base model)")
    ap.add_argument("--prompt-type", default="QA1", choices=["P","QA1","QA2"])
    ap.add_argument("--shots", type=int, default=0, help="ICL shots (0 for finetuned models)")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out-dir", default="outputs/workflow_ordering")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--tag", default=None, help="Override output filename tag")
    args = ap.parse_args()

    model_tag = args.base_model.split("/")[-1]
    if args.adapter:
        adapter_tag = Path(args.adapter).name.replace("_lora_adapter","")
        run_tag = args.tag or f"{adapter_tag}__wf_order"
    else:
        run_tag = args.tag or f"{model_tag}__ICL_k{args.shots}_{args.prompt_type}__wf_order"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_tag}.json"

    print(f"\n{'='*65}")
    print(f"RUN: {run_tag}")
    print(f"{'='*65}")

    wfs, records = load_workflows()
    print(f"Workflows: {len(wfs)}  Directed pairs: {len(records)}")

    # Few-shot examples for ICL
    few_shot = None
    if args.shots > 0 and not args.adapter:
        few_shot = P.sample_few_shot(TRAIN, k_per_class=args.shots//2, seed=42)
        print(f"Using {len(few_shot)} few-shot examples")

    # Determine model path
    if args.adapter:
        merged_dir = Path(f"/mnt/localssd/causal-embedding-research/llm_trc_baseline/outputs/merged_models/{run_tag}")
        if not args.skip_merge or not merged_dir.exists():
            merge_adapter(args.base_model, args.adapter, merged_dir)
        else:
            print(f"  Using cached merged model at {merged_dir}")
        model_path = str(merged_dir)
    else:
        model_path = args.base_model

    # Load vLLM
    from vllm import LLM
    print(f"Loading vLLM (tp={args.tp}, model={model_path})...", flush=True)
    llm = LLM(model=model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len,
              dtype="bfloat16", trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    yes_id, no_id = get_yes_no_ids(tokenizer, args.base_model)

    # Score all pairs
    probs = score_pairs_vllm(llm, tokenizer, records, args.prompt_type,
                              yes_id, no_id, few_shot)

    # Evaluate ordering
    print("\nOrdering results:")
    metrics = evaluate_ordering(wfs, records, probs)

    # Save
    result = {
        "run_tag": run_tag, "base_model": args.base_model,
        "adapter": args.adapter, "prompt_type": args.prompt_type,
        "shots": args.shots, "yes_token_id": yes_id, "no_token_id": no_id,
        **metrics
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")
    return metrics


if __name__ == "__main__":
    main()
