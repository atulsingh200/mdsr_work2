"""
Evaluate a trained GPT-2 causal classifier on any split.

Usage:
    python evaluate.py --checkpoint ./checkpoints/best_model \
                       --data ../data/aep_causal_classification_34/directional_test.jsonl \
                       --config config.yaml
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset import CausalPairDataset, label_token_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to saved model dir")
    p.add_argument("--data",       required=True, help="Path to .jsonl split")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    batch_size = args.batch_size or cfg["batch_size"] * 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    zero_id, one_id = label_token_ids(tokenizer)

    ds = CausalPairDataset(args.data, tokenizer, cfg["max_len"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=ds.collate_fn,
    )

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B, seq_len, vocab]

            seq_lens  = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=device)
            last_logits = logits[batch_idx, seq_lens]

            score_0 = last_logits[:, zero_id]
            score_1 = last_logits[:, one_id]
            preds   = (score_1 > score_0).long()

            gt_tokens = labels[batch_idx, seq_lens]
            gt = (gt_tokens == one_id).long()

            tp += ((preds == 1) & (gt == 1)).sum().item()
            tn += ((preds == 0) & (gt == 0)).sum().item()
            fp += ((preds == 1) & (gt == 0)).sum().item()
            fn += ((preds == 0) & (gt == 1)).sum().item()

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Results on: {args.data}")
    print(f"  Total examples : {total}")
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  Confusion matrix:")
    print(f"             Pred 0   Pred 1")
    print(f"  True  0  :  {tn:>6}   {fp:>6}")
    print(f"  True  1  :  {fn:>6}   {tp:>6}")


if __name__ == "__main__":
    main()
