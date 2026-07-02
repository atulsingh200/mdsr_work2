"""
Rerank generated_follow_ups in match_results.jsonl using the perplexity of a
causal language model (GPT-2-medium fine-tuned on AEP transcripts).

For each (question, follow_up) pair we build the string
    question + " " + follow_up
(joined by a single space — no special separators) and compute the
FULL-SEQUENCE perplexity under the LM. Lower perplexity = more fluent/likely
continuation = ranked higher. Follow-ups are sorted ascending by perplexity.

Metrics (hit@k, recall@k for k=1,3,5) are computed only on records that have
at least one follow_up with clicked == 1 — identical methodology to
rerank_ce2x.py so the results are directly comparable.

Usage:
    python rerank_gpt2_ppl.py \\
        --input  match_results.jsonl \\
        --model-dir <path>/output_gpt2_smooth \\
        --output reranked_gpt2_ppl.jsonl \\
        [--batch-size 32] [--device cuda] [--max-len 1024]
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class PairDataset(Dataset):
    """Flat list of (concatenated_text, rec_idx, fup_idx)."""

    def __init__(self, records: list):
        self.pairs: list[tuple] = []
        for rec_idx, rec in enumerate(records):
            question = rec["question"]
            for fup_idx, fup in enumerate(rec["generated_follow_ups"]):
                text = question + " " + fup["follow_up"]   # join by a single space
                self.pairs.append((text, rec_idx, fup_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def make_collate(tokenizer, max_len: int):
    def _collate(batch):
        texts, rec_idxs, fup_idxs = zip(*batch)
        enc = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        return enc, list(rec_idxs), list(fup_idxs)
    return _collate


@torch.no_grad()
def sequence_perplexity(model, input_ids, attention_mask, use_bf16):
    """Full-sequence perplexity per row. Returns tensor (B,)."""
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.autocast(device_type="cpu", enabled=False))
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # shift: position t predicts token t+1
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    loss_mask = attention_mask[:, 1:].float()          # 1 for real next-token targets

    B, Lm1, V = shift_logits.shape
    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(B, Lm1)

    tok_counts = loss_mask.sum(dim=1).clamp(min=1.0)
    mean_loss = (per_tok * loss_mask).sum(dim=1) / tok_counts
    return torch.exp(mean_loss)                        # perplexity (B,)


def compute_metrics(records: list, scores: list[dict], ks=(1, 3, 5)):
    """Lower score (perplexity) ranks higher. Eval only on records with a click."""
    hit = {k: [] for k in ks}
    recall = {k: [] for k in ks}

    for rec_idx, rec in enumerate(records):
        fups = rec["generated_follow_ups"]
        gt_set = {f["follow_up"] for f in fups if f.get("clicked", 0) == 1}
        if not gt_set:
            continue  # no ground truth → skip

        # ascending perplexity = best first
        ranked = sorted(range(len(fups)),
                        key=lambda i: scores[rec_idx].get(i, float("inf")))
        ranked_texts = [fups[i]["follow_up"] for i in ranked]

        for k in ks:
            top_k = set(ranked_texts[:k])
            overlap = gt_set & top_k
            hit[k].append(1 if overlap else 0)
            recall[k].append(len(overlap) / len(gt_set))

    result = {}
    for k in ks:
        n = len(hit[k])
        result[f"hit@{k}"]    = sum(hit[k]) / n    if n else float("nan")
        result[f"recall@{k}"] = sum(recall[k]) / n if n else float("nan")
    result["n_eval"] = len(hit[ks[0]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="match_results.jsonl")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output",    default="reranked_gpt2_ppl.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-len",   type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"
    print(f"Device: {device}  bf16={use_bf16}")

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16 if use_bf16 else torch.float32,
    ).to(device).eval()
    print(f"  n_params={sum(p.numel() for p in model.parameters()):,}  "
          f"max_len={args.max_len}")

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Records loaded: {len(records)}")

    dataset = PairDataset(records)
    print(f"Total pairs to score: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, args.max_len),
        num_workers=0,
    )

    all_scores: list[dict] = [{} for _ in range(len(records))]

    for enc, rec_idxs, fup_idxs in tqdm(loader, desc="Scoring (perplexity)"):
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        ppl = sequence_perplexity(model, input_ids, attention_mask, use_bf16)
        ppl = ppl.float().cpu().tolist()
        for score, rec_idx, fup_idx in zip(ppl, rec_idxs, fup_idxs):
            all_scores[rec_idx][fup_idx] = score

    # write reranked output (ascending perplexity), preserving schema
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for rec_idx, rec in enumerate(records):
            n = len(rec["generated_follow_ups"])
            order = sorted(range(n), key=lambda i: all_scores[rec_idx].get(i, float("inf")))
            reranked_fups = [rec["generated_follow_ups"][i] for i in order]
            out_rec = {**rec, "generated_follow_ups": reranked_fups}
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print(f"\nReranked output written to: {out_path}")

    # evaluation
    metrics = compute_metrics(records, all_scores)
    n_eval = metrics.pop("n_eval")
    print(f"\nEvaluation  (n={n_eval} records with ≥1 clicked follow-up)")
    print(f"  {'Metric':<12}  {'Score':>8}")
    print("  " + "-" * 22)
    for k in (1, 3, 5):
        print(f"  hit@{k:<8}   {metrics[f'hit@{k}']:.4f}")
    print()
    for k in (1, 3, 5):
        print(f"  recall@{k:<6}   {metrics[f'recall@{k}']:.4f}")

    # first-record example
    rec = records[0]
    fups = rec["generated_follow_ups"]
    print(f"\nExample — question: {rec['question'][:80]}")
    n = len(fups)
    order = sorted(range(n), key=lambda i: all_scores[0].get(i, float("inf")))
    print("Reranked (ascending perplexity = best first):")
    for pos, idx in enumerate(order):
        fup = fups[idx]
        ppl = all_scores[0][idx]
        print(f"  rank{pos+1} [ck={fup.get('clicked',0)}] [ppl={ppl:8.2f}] {fup['follow_up'][:65]}")


if __name__ == "__main__":
    main()
