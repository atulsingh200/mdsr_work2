"""
Rerank follow-up queries using the finetuned DirectionalClassifier.

For each record: score(question, follow_up_query) for all follow-up queries,
then sort descending by score.

Usage:
    python rerank_followup.py \
        --input  prod_jan_feb_26-aep-ajo_follow-up-queries.json \
        --model  /mnt/localssd/automation/internship-causal-embedding/runs/classifier/ajo_doc_dataset/best.pt \
        --output reranked_followup.json \
        [--batch-size 64] [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ── make the project src importable ──────────────────────────────────────────
SRC = Path("/mnt/localssd/automation/internship-causal-embedding/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.model import DirectionalClassifier  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, config: dict, device: torch.device) -> tuple:
    model = DirectionalClassifier(
        base_model_name=config["base_model"],
        head_cfg=config.get("head_cfg"),
        n_tiers=config.get("n_tiers", 0),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # checkpoint saves state under 'model'; fall back to flat dict for older saves
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.model_max_length = config.get("max_seq_len", 512)
    return model, tokenizer


class PairDataset(Dataset):
    """Flat list of (question, follow_up, record_idx, query_idx) tuples."""

    def __init__(self, records: list):
        self.pairs = []
        for rec_idx, rec in enumerate(records):
            question = rec["question"]
            # follow_up_queries is [[q1, q2, ...]] — unwrap outer list
            queries = rec["follow_up_queries"]
            if queries and isinstance(queries[0], list):
                queries = queries[0]
            for q_idx, fq in enumerate(queries):
                self.pairs.append((question, fq, rec_idx, q_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def _hard_truncate(enc: dict, max_len: int) -> dict:
    return {k: v[:, :max_len] for k, v in enc.items()}


def collate_fn(tokenizer, max_len):
    def _collate(batch):
        questions, follow_ups, rec_idxs, q_idxs = zip(*batch)
        enc1 = tokenizer(
            list(questions),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc2 = tokenizer(
            list(follow_ups),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc1 = _hard_truncate(enc1, max_len)
        enc2 = _hard_truncate(enc2, max_len)
        return enc1, enc2, list(rec_idxs), list(q_idxs)
    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="prod_jan_feb_26-aep-ajo_follow-up-queries.json")
    parser.add_argument("--model",  default="/mnt/localssd/automation/internship-causal-embedding/runs/classifier/ajo_doc_dataset/best.pt")
    parser.add_argument("--config", default="/mnt/localssd/automation/internship-causal-embedding/runs/classifier/ajo_doc_dataset/config.json")
    parser.add_argument("--output", default="reranked_followup.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # load config
    with open(args.config) as f:
        config = json.load(f)

    # load model
    print(f"Loading model from {args.model} ...")
    model, tokenizer = load_model(args.model, config, device)
    print("Model loaded.")

    # load data
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path
    with open(input_path) as f:
        records = json.load(f)
    print(f"Records: {len(records)}")

    # build flat pair dataset
    dataset = PairDataset(records)
    print(f"Total pairs to score: {len(dataset)}")

    # Cap at the encoder's actual position-embedding capacity (some configs
    # record max_seq_len > 512, which the BERT/BGE position table can't handle).
    model_max_pos = model.src.config.max_position_embeddings
    max_len = min(config.get("max_seq_len", 512), model_max_pos)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, max_len),
        num_workers=0,
    )

    # score all pairs
    # scores[rec_idx][q_idx] = sigmoid(logit)
    num_records = len(records)
    all_scores: list[dict] = [{} for _ in range(num_records)]

    with torch.no_grad():
        for enc1, enc2, rec_idxs, q_idxs in tqdm(loader, desc="Scoring"):
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            enc2 = {k: v.to(device) for k, v in enc2.items()}
            logits, _, _ = model(enc1, enc2)
            scores = torch.sigmoid(logits).cpu().tolist()
            for score, rec_idx, q_idx in zip(scores, rec_idxs, q_idxs):
                all_scores[rec_idx][q_idx] = score

    # build reranked output
    output_records = []
    for rec_idx, rec in enumerate(records):
        question = rec["question"]
        queries = rec["follow_up_queries"]
        if queries and isinstance(queries[0], list):
            queries = queries[0]

        scored = [
            {"query": q, "score": all_scores[rec_idx][q_idx]}
            for q_idx, q in enumerate(queries)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        output_records.append({
            "question": question,
            "follow_up_queries": [item["query"] for item in scored],
            "scores": [round(item["score"], 6) for item in scored],
        })

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    with open(out_path, "w") as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Reranked output written to: {out_path}")
    print(f"Example (first record):")
    ex = output_records[0]
    print(f"  Question: {ex['question']}")
    for q, s in zip(ex["follow_up_queries"], ex["scores"]):
        print(f"    [{s:.4f}] {q}")


if __name__ == "__main__":
    main()
