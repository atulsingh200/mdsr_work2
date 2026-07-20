"""Paper-faithful ICL eval — reproduces arXiv:2410.10476 setting on our data.

KEY CHANGES vs icl_eval.py:
  1. Uses prompts_paper.py: NO system message, raw example-label format, 20-word truncation
  2. k_per_class=1 (paper uses 1 example per class for MATRES/TIMELINE, so k=1 per class = 2 examples)
  3. Seeds 0-4 as before
  4. This is the HARD setting — LLMs should underperform the encoder

Example:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python icl_eval_paper.py \
      --model Qwen/Qwen2.5-7B-Instruct --prompt-types P QA1 QA2 --shots 0 1 --tp 1

Smoke:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python icl_eval_paper.py \
      --model Qwen/Qwen2.5-7B-Instruct --prompt-types QA1 --shots 0 --limit 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import prompts_paper as P

HERE = Path(__file__).parent
CER = Path("/mnt/localssd/causal-embedding-research")
DEFAULT_TEST = CER / "final_data/directional_test.jsonl"
DEFAULT_TRAIN = CER / "final_data/directional_train.jsonl"


def load_rows(path: Path, limit: int | None) -> list[dict]:
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

    full = np.ones(len(y), dtype=bool)
    out = {"overall": _m(full), "by_source": {}, "n": int(len(y)),
           "unparsed": int(unparsed), "unparsed_rate": float(unparsed / max(len(y), 1)),
           "pred_pos_rate": float(p.mean())}
    for s in sorted(set(src.tolist())):
        out["by_source"][s] = _m(src == s)
    return out


def run_setting(llm, tok, rows, prompt_type, few_shot, max_tokens):
    from vllm import SamplingParams
    chats = [P.build_messages(prompt_type, r["text_1"], r["text_2"], few_shot) for r in rows]
    # Paper uses NO system message — raw text completion style
    text_prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                    for c in chats]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.generate(text_prompts, sp, use_tqdm=False)
    preds, unparsed, samples = [], 0, []
    for r, o in zip(rows, outs):
        gen = o.outputs[0].text
        lab = P.parse_answer(prompt_type, gen)
        if lab is None:
            unparsed += 1
            lab = 0
        preds.append(lab)
        if len(samples) < 8:
            samples.append({"gen": gen[:120], "pred": lab, "gold": int(r["label"])})
    m = compute_metrics(rows, preds, unparsed)
    m["samples"] = samples
    return m, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-types", nargs="+", default=list(P.PROMPT_TYPES))
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 1],
                    help="k per class; 0=zero-shot. Paper uses k=1 (1 per class).")
    ap.add_argument("--few-shot-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(HERE / "outputs/icl_paper"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]

    rows = load_rows(Path(args.test), args.limit)
    print(f"test rows: {len(rows)}  model: {args.model}  MAX_WORDS={P.MAX_WORDS}", flush=True)

    from vllm import LLM
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=False, trust_remote_code=True)
    tok = llm.get_tokenizer()

    summary = {}
    for ptype in args.prompt_types:
        max_tokens = 12 if ptype == "QA2" else 6
        for k in args.shots:
            if k == 0:
                m, _ = run_setting(llm, tok, rows, ptype, None, max_tokens)
                tag = f"{model_tag}__{ptype}__k0__paper"
                rec = {"model": args.model, "prompt_type": ptype, "shots": 0,
                       "max_words": P.MAX_WORDS, "paper_faithful": True, **m}
                (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
                summary[tag] = {"f1": m["overall"]["f1"], "acc": m["overall"]["acc"],
                                 "unparsed_rate": m["unparsed_rate"]}
                print(f"[{tag}] f1={m['overall']['f1']:.4f} acc={m['overall']['acc']:.4f} "
                      f"unparsed={m['unparsed_rate']:.3f} pred_pos={m['pred_pos_rate']:.3f}",
                      flush=True)
            else:
                seed_f1s, seed_accs = [], []
                for seed in args.few_shot_seeds:
                    fs = P.sample_few_shot(args.train, k_per_class=k, seed=seed)
                    m, _ = run_setting(llm, tok, rows, ptype, fs, max_tokens)
                    tag = f"{model_tag}__{ptype}__k{k}__s{seed}__paper"
                    rec = {"model": args.model, "prompt_type": ptype, "shots": k,
                           "seed": seed, "max_words": P.MAX_WORDS, "paper_faithful": True, **m}
                    (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
                    seed_f1s.append(m["overall"]["f1"])
                    seed_accs.append(m["overall"]["acc"])
                    print(f"[{tag}] f1={m['overall']['f1']:.4f} acc={m['overall']['acc']:.4f} "
                          f"unparsed={m['unparsed_rate']:.3f}", flush=True)
                agg_tag = f"{model_tag}__{ptype}__k{k}__paper"
                agg = {"model": args.model, "prompt_type": ptype, "shots": k,
                       "seeds": args.few_shot_seeds,
                       "f1_mean": statistics.mean(seed_f1s),
                       "f1_std": statistics.pstdev(seed_f1s) if len(seed_f1s) > 1 else 0.0,
                       "acc_mean": statistics.mean(seed_accs),
                       "seed_f1s": seed_f1s, "max_words": P.MAX_WORDS}
                (out_dir / f"{agg_tag}__agg.json").write_text(json.dumps(agg, indent=2))
                summary[agg_tag] = {"f1_mean": agg["f1_mean"], "f1_std": agg["f1_std"]}
                print(f"[{agg_tag}] f1={agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f}", flush=True)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
