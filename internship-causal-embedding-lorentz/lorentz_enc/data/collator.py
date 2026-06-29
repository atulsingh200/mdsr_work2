import torch
from typing import List, Dict


class TripletCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tok = tokenizer
        self.max_length = max_length

    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

    def __call__(self, batch: List[Dict]) -> Dict:
        B = len(batch)
        N = max(len(ex["negatives"]) for ex in batch)
        if N == 0:
            N = 1  # at least 1 slot to avoid empty tensors
        neg_mask = torch.zeros(B, N, dtype=torch.bool)
        anchors, positives, all_negatives = [], [], []
        for i, ex in enumerate(batch):
            anchors.append(ex["anchor"])
            positives.append(ex["positive"])
            negs = list(ex["negatives"])
            for j in range(N):
                if j < len(negs):
                    all_negatives.append(negs[j])
                    neg_mask[i, j] = True
                else:
                    all_negatives.append("")
                    neg_mask[i, j] = False
        return {
            "anchor": self._encode(anchors),
            "positive": self._encode(positives),
            "negatives": self._encode(all_negatives),
            "neg_mask": neg_mask,
            "B": B, "N": N,
            "anchor_ids": torch.tensor([ex.get("anchor_id", i) for i, ex in enumerate(batch)]),
            "positive_ids": torch.tensor([ex.get("positive_id", -1) for ex in batch]),
        }
