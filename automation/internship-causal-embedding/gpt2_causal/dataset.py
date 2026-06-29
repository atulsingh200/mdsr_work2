import json
import math
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def build_tokenizer(model_name: str) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "additional_special_tokens": ["[SEP]", "<0>", "<1>"],
    })
    return tokenizer


def label_token_ids(tokenizer: GPT2Tokenizer):
    zero_id = tokenizer.convert_tokens_to_ids("<0>")
    one_id  = tokenizer.convert_tokens_to_ids("<1>")
    return zero_id, one_id


def _truncate_symmetric(ids1: list, ids2: list, budget: int):
    """Trim ids1 and ids2 symmetrically so their combined length <= budget."""
    excess = len(ids1) + len(ids2) - budget
    if excess <= 0:
        return ids1, ids2
    trim_each = math.ceil(excess / 2)
    ids1 = ids1[:max(1, len(ids1) - trim_each)]
    ids2 = ids2[:max(1, len(ids2) - trim_each)]
    # Second pass in case one side was already very short
    excess2 = len(ids1) + len(ids2) - budget
    if excess2 > 0:
        ids2 = ids2[:max(1, len(ids2) - excess2)]
    return ids1, ids2


class CausalPairDataset(Dataset):
    """
    Each sequence:  [text_1 tokens] [SEP] [text_2 tokens] <0|1>
    Loss is computed only on the final label token.
    """

    def __init__(self, path: str, tokenizer: GPT2Tokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sep_id   = tokenizer.convert_tokens_to_ids("[SEP]")
        self.zero_id, self.one_id = label_token_ids(tokenizer)
        self.pad_id   = tokenizer.pad_token_id

        # budget: leave 2 slots for [SEP] and label
        self.text_budget = max_len - 2

        self.records = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                self.records.append((d["text_1"], d["text_2"], int(d["label"])))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text1, text2, label = self.records[idx]

        ids1 = self.tokenizer.encode(text1, add_special_tokens=False)
        ids2 = self.tokenizer.encode(text2, add_special_tokens=False)
        ids1, ids2 = _truncate_symmetric(ids1, ids2, self.text_budget)

        label_id = self.one_id if label == 1 else self.zero_id
        input_ids = ids1 + [self.sep_id] + ids2 + [label_id]

        # labels tensor: -100 everywhere except the final label position
        labels = [-100] * (len(input_ids) - 1) + [label_id]

        return input_ids, labels

    def collate_fn(self, batch):
        input_ids_list, labels_list = zip(*batch)
        max_seq = max(len(x) for x in input_ids_list)

        padded_input  = []
        padded_labels = []
        attention_masks = []

        for inp, lbl in zip(input_ids_list, labels_list):
            pad_len = max_seq - len(inp)
            padded_input.append(inp + [self.pad_id] * pad_len)
            padded_labels.append(lbl + [-100] * pad_len)
            attention_masks.append([1] * len(inp) + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(padded_input,    dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels":         torch.tensor(padded_labels,   dtype=torch.long),
        }
