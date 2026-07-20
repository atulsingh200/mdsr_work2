"""ICL chain-of-thought reference (non-fine-tuned): rationale-then-VERDICT via vLLM.

Shows how much the rationale format alone (no fine-tuning) buys, as a reference row for the LLMERE
comparison. Zero-shot and few-shot (exemplars carry their reasoning trace).

Run:
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python llmere/icl_cot.py --model Qwen/Qwen2.5-7B-Instruct --shots 0 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import prompts as P  # noqa: E402

CER = Path("/mnt/localssd/causal-embedding-research")
TEST = CER / "final_data/directional_test.jsonl"
TRAIN = CER / "final_data/directional_train.jsonl"


def load_rows(path, limit=None):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
        if limit and len(rows) >= limit:
            break
    return rows


def few_shot(train_path, k_per_class, seed):
    rng = random.Random(seed)
    pos, neg = [], []
    for r in load_rows(train_path):
        ex = {"text_1": r["text_1"], "text_2": r["text_2"], "label": int(r["label"]),
              "reasoning": (r.get("reasoning") or "").strip()}
        (pos if ex["label"] == 1 else neg).append(ex)
    rng.shuffle(pos); rng.shuffle(neg)
    shots = pos[:k_per_class] + neg[:k_per_class]
    rng.shuffle(shots)
    return shots


def metrics(rows, preds, unparsed):
    y = np.array([int(r["label"]) for r in rows]); p = np.array(preds)
    src = np.array([r.get("source", "unknown") for r in rows])
    def _m(mask):
        yy, pp = y[mask], p[mask]
        return {"n": int(mask.sum()), "acc": float((yy == pp).mean()),
                "f1": float(f1_score(yy, pp, zero_division=0)),
                "f1_before": float(f1_score(yy, pp, pos_label=1, zero_division=0)),
                "f1_not_before": float(f1_score(yy, pp, pos_label=0, zero_division=0))}
    out = {"overall": _m(np.ones(len(y), bool)), "by_source": {}, "unparsed": unparsed,
           "unparsed_rate": unparsed / max(len(y), 1)}
    for s in sorted(set(src.tolist())):
        out["by_source"][s] = _m(src == s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out-dir", default=str(HERE.parent / "outputs/llmere"))
    args = ap.parse_args()

    rows = load_rows(TEST, args.limit)
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=4096, dtype="bfloat16", trust_remote_code=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=256, stop=None)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mtag = args.model.split("/")[-1]
    summary = {}
    for k in args.shots:
        fs = None if k == 0 else few_shot(TRAIN, k, args.seed)
        prompts = [tok.apply_chat_template(P.build_cot_messages(r["text_1"], r["text_2"], fs),
                                           tokenize=False, add_generation_prompt=True) for r in rows]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        preds, unparsed = [], 0
        for o in outs:
            lab = P.parse_verdict(o.outputs[0].text)
            if lab is None:
                unparsed += 1; lab = 0
            preds.append(lab)
        m = metrics(rows, preds, unparsed)
        tag = f"{mtag}__cot__k{k}"
        (out_dir / f"icl_{tag}.json").write_text(json.dumps(
            {"model": args.model, "method": "icl_cot", "shots": k, **m}, indent=2))
        summary[tag] = {"acc": m["overall"]["acc"], "f1": m["overall"]["f1"],
                        "unparsed": m["unparsed_rate"]}
        print(f"[{tag}] acc={m['overall']['acc']:.4f} f1={m['overall']['f1']:.4f} "
              f"unparsed={m['unparsed_rate']:.3f}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
