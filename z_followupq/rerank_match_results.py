"""
Rerank generated_follow_ups in match_results.jsonl using the finetuned
DirectionalClassifier. Each follow_up is scored against the question;
the list is sorted most→least relevant. All original fields are preserved
exactly; only the order of generated_follow_ups changes.

Usage:
    python rerank_match_results.py \
        --input  match_results.jsonl \
        --model  <path>/best.pt \
        --config <path>/config.json \
        --output reranked_match_results.jsonl \
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

SRC = Path("/mnt/localssd/automation/internship-causal-embedding/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.model import DirectionalClassifier  # noqa: E402
from classifier.reasoning_model import ReasoningClassifier  # noqa: E402


def build_model(arch: str, config: dict):
    """Construct the right architecture. Both expose forward(enc1, enc2) ->
    (dir_logit, _, _), so scoring is identical downstream."""
    arch = (arch or "directional").lower()
    if arch in ("reasoning", "reasoningclassifier"):
        return ReasoningClassifier(
            base_model_name=config["base_model"],
            head_cfg=config.get("head_cfg"),
        )
    return DirectionalClassifier(
        base_model_name=config["base_model"],
        head_cfg=config.get("head_cfg"),
        n_tiers=config.get("n_tiers", 0),
    )


def load_model(checkpoint_path: str, config: dict, device: torch.device, arch: str):
    model = build_model(arch, config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.model_max_length = config.get("max_seq_len", 512)
    return model, tokenizer


class PairDataset(Dataset):
    """Flat list of (question, follow_up, rec_idx, fup_idx)."""

    def __init__(self, records: list):
        self.pairs = []
        for rec_idx, rec in enumerate(records):
            question = rec["question"]
            for fup_idx, fup in enumerate(rec["generated_follow_ups"]):
                self.pairs.append((question, fup["follow_up"], rec_idx, fup_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def _hard_truncate(enc: dict, max_len: int) -> dict:
    return {k: v[:, :max_len] for k, v in enc.items()}


def collate_fn(tokenizer, max_len):
    def _collate(batch):
        questions, follow_ups, rec_idxs, fup_idxs = zip(*batch)
        enc1 = tokenizer(list(questions), padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        enc2 = tokenizer(list(follow_ups), padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        enc1 = _hard_truncate(enc1, max_len)
        enc2 = _hard_truncate(enc2, max_len)
        return enc1, enc2, list(rec_idxs), list(fup_idxs)
    return _collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="match_results.jsonl")
    parser.add_argument("--model",  required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="reranked_match_results.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--arch",
        choices=["auto", "directional", "reasoning"],
        default="auto",
        help="Model architecture. 'auto' reads config['architecture'] "
             "(ReasoningClassifier -> reasoning, else directional).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    with open(args.config) as f:
        config = json.load(f)

    arch = args.arch
    if arch == "auto":
        arch = "reasoning" if config.get("architecture") == "ReasoningClassifier" else "directional"
    print(f"Architecture: {arch}")

    print(f"Loading model from {args.model} ...")
    model, tokenizer = load_model(args.model, config, device, arch)
    print("Model loaded.")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path

    with open(input_path) as f:
        records = [json.loads(line) for line in f]
    print(f"Records: {len(records)}")

    dataset = PairDataset(records)
    print(f"Total pairs to score: {len(dataset)}")

    # Cap at the encoder's actual position-embedding capacity. Some configs
    # (e.g. best_mlp_bge_small) record max_seq_len=524, which exceeds BERT/BGE's
    # 512 position table and crashes the embedding add.
    model_max_pos = model.src.config.max_position_embeddings
    max_len = min(config.get("max_seq_len", 512), model_max_pos)
    print(f"max_len = {max_len} (config={config.get('max_seq_len')}, model_max_pos={model_max_pos})")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, max_len),
        num_workers=0,
    )

    # scores[rec_idx][fup_idx] = sigmoid score
    all_scores: list[dict] = [{} for _ in range(len(records))]

    with torch.no_grad():
        for enc1, enc2, rec_idxs, fup_idxs in tqdm(loader, desc="Scoring"):
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            enc2 = {k: v.to(device) for k, v in enc2.items()}
            logits, _, _ = model(enc1, enc2)
            scores = torch.sigmoid(logits).cpu().tolist()
            for score, rec_idx, fup_idx in zip(scores, rec_idxs, fup_idxs):
                all_scores[rec_idx][fup_idx] = score

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    with open(out_path, "w") as f:
        for rec_idx, rec in enumerate(records):
            scored = sorted(
                range(len(rec["generated_follow_ups"])),
                key=lambda i: all_scores[rec_idx][i],
                reverse=True,
            )
            reranked_fups = [rec["generated_follow_ups"][i] for i in scored]
            out_rec = {**rec, "generated_follow_ups": reranked_fups}
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Written to: {out_path}")

    # show first record diff
    rec = records[0]
    print(f"\nExample — question: {rec['question'][:80]}")
    print("Original order:")
    for fup in rec["generated_follow_ups"]:
        print(f"  [{fup['rating']:7s}] {fup['follow_up'][:70]}")
    with open(out_path) as f:
        first_out = json.loads(f.readline())
    print("Reranked order:")
    for i, fup in enumerate(first_out["generated_follow_ups"]):
        score = all_scores[0][
            next(j for j, x in enumerate(rec["generated_follow_ups"]) if x["follow_up"] == fup["follow_up"])
        ]
        print(f"  [{score:.4f}] [{fup['rating']:7s}] {fup['follow_up'][:70]}")


if __name__ == "__main__":
    main()
