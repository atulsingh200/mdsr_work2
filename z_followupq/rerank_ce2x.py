"""
Rerank generated_follow_ups in match_results.jsonl using a trained
CrossEncoder2x (DeBERTa-v3-large) model.

The cross-encoder jointly tokenizes each (question, follow_up) pair as:
    [CLS] question [SEP] follow_up [SEP]
and returns a scalar relevance score. Follow-ups are sorted
highest-score first. Metrics (hit@k, recall@k for k=1,3,5) are computed
only on records that have at least one follow_up with clicked == 1.

Usage:
    python rerank_ce2x.py \\
        --input  match_results.jsonl \\
        --model  <path>/best.pt \\
        --config <path>/config.json \\
        --output reranked_ce2x.jsonl \\
        [--batch-size 32] [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

_CE2X_MODEL_DIR = Path(
    "/mnt/localssd/new_base_line/internship-causal-embedding"
    "/my_work/kl/cross_encoder_2x"
)


def _add_ce2x_path(override: str | None = None) -> None:
    p = str(Path(override) if override else _CE2X_MODEL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(checkpoint_path: str, config: dict, device: torch.device, ce2x_dir: str | None):
    _add_ce2x_path(ce2x_dir)
    from model_ce2x import CrossEncoder2x  # noqa: E402

    model = CrossEncoder2x(
        backbone=config["backbone"],
        n_layers=config.get("n_layers", 24),
        dropout=config.get("dropout", 0.1),
        native_backbone=config.get("native_backbone", True),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


class PairDataset(Dataset):
    """Flat list of (question, follow_up, rec_idx, fup_idx)."""

    def __init__(self, records: list):
        self.pairs: list[tuple] = []
        for rec_idx, rec in enumerate(records):
            question = rec["question"]
            for fup_idx, fup in enumerate(rec["generated_follow_ups"]):
                self.pairs.append((question, fup["follow_up"], rec_idx, fup_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def make_collate(tokenizer, max_len: int):
    def _collate(batch):
        questions, follow_ups, rec_idxs, fup_idxs = zip(*batch)
        enc = tokenizer(
            list(questions), list(follow_ups),
            padding=True, truncation="longest_first",
            max_length=max_len, return_tensors="pt",
        )
        return enc, list(rec_idxs), list(fup_idxs)
    return _collate


def compute_metrics(records: list, scores: list[dict], ks=(1, 3, 5)):
    """
    records : original record list
    scores  : scores[rec_idx][fup_idx] = float score
    Returns : dict of hit@k and recall@k for k in ks
    """
    hit = {k: [] for k in ks}
    recall = {k: [] for k in ks}

    for rec_idx, rec in enumerate(records):
        fups = rec["generated_follow_ups"]
        # ground truth: follow_up texts that were clicked
        gt_set = {f["follow_up"] for f in fups if f.get("clicked", 0) == 1}
        if not gt_set:
            continue  # no ground truth → skip

        # rank follow_ups by score descending
        ranked = sorted(range(len(fups)), key=lambda i: scores[rec_idx].get(i, 0.0), reverse=True)
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
    parser.add_argument("--input",   default="match_results.jsonl")
    parser.add_argument("--model",   required=True)
    parser.add_argument("--config",  required=True)
    parser.add_argument("--output",  default="reranked_ce2x.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-ce2x-dir", default=None,
                        help="Override path to model_ce2x.py directory.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    with open(args.config) as f:
        config = json.load(f)

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model, config, device, args.model_ce2x_dir)
    print(f"  backbone={config['backbone']}  n_params={sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
    max_seq_len = config.get("max_seq_len", 512)
    print(f"  max_seq_len={max_seq_len}")

    input_path = Path(args.input)
    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Records loaded: {len(records)}")

    dataset = PairDataset(records)
    print(f"Total pairs to score: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_seq_len),
        num_workers=0,
    )

    all_scores: list[dict] = [{} for _ in range(len(records))]

    with torch.no_grad():
        for enc, rec_idxs, fup_idxs in tqdm(loader, desc="Scoring"):
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            token_type_ids = enc.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            batch_scores = torch.sigmoid(logits).cpu().tolist()

            for score, rec_idx, fup_idx in zip(batch_scores, rec_idxs, fup_idxs):
                all_scores[rec_idx][fup_idx] = score

    # write reranked output
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for rec_idx, rec in enumerate(records):
            n = len(rec["generated_follow_ups"])
            ranked_order = sorted(range(n), key=lambda i: all_scores[rec_idx].get(i, 0.0), reverse=True)
            reranked_fups = [rec["generated_follow_ups"][i] for i in ranked_order]
            out_rec = {**rec, "generated_follow_ups": reranked_fups}
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print(f"\nReranked output written to: {out_path}")

    # evaluation
    metrics = compute_metrics(records, all_scores)
    n_eval = metrics.pop("n_eval")
    print(f"\nEvaluation  (n={n_eval} records with ≥1 clicked follow-up)")
    print(f"{'Metric':<12}  {'Score':>8}")
    print("-" * 24)
    for k in (1, 3, 5):
        print(f"  hit@{k:<5}   {metrics[f'hit@{k}']:.4f}")
    print()
    for k in (1, 3, 5):
        print(f"  recall@{k:<3}   {metrics[f'recall@{k}']:.4f}")

    # show first record example
    rec = records[0]
    fups = rec["generated_follow_ups"]
    print(f"\nExample — question: {rec['question'][:80]}")
    print("Original order:")
    for fup in fups:
        orig_idx = next(j for j, x in enumerate(fups) if x["follow_up"] == fup["follow_up"])
        sc = all_scores[0].get(orig_idx, float("nan"))
        print(f"  [ck={fup.get('clicked',0)}] [{sc:.4f}] {fup['follow_up'][:70]}")
    print("Reranked (sorted by score):")
    n = len(fups)
    ranked_order = sorted(range(n), key=lambda i: all_scores[0].get(i, 0.0), reverse=True)
    for pos, idx in enumerate(ranked_order):
        fup = fups[idx]
        sc = all_scores[0][idx]
        print(f"  rank{pos+1} [ck={fup.get('clicked',0)}] [{sc:.4f}] {fup['follow_up'][:70]}")


if __name__ == "__main__":
    main()
