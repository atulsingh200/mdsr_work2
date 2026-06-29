import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class CausalPairDataset(Dataset):
    """Each item: (anchor_text, positive_text, [negative_texts])."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512,
                 max_negatives: int = 7):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                assert "anchor" in ex and "positive" in ex
                if "negatives" not in ex:
                    ex["negatives"] = []
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_negatives = max_negatives

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict:
        ex = self.examples[idx]
        negs = ex["negatives"][:self.max_negatives]
        return {
            "anchor": ex["anchor"],
            "positive": ex["positive"],
            "negatives": negs,
            "anchor_id": ex.get("anchor_id", idx),
            "positive_id": ex.get("positive_id", -1),
        }
