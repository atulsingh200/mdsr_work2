"""
Step 1: Build and save the merged dataset (transcripts + C4) to disk.

Outputs
-------
  <output_dir>/merged_docs.jsonl   -- one raw document per line (inspectable)
  <output_dir>/train/              -- Arrow dataset of packed 1024-token chunks
  <output_dir>/eval/               -- Arrow dataset of packed 1024-token chunks

Run once; train.py loads from the Arrow files on every training run.

Usage
-----
    python prepare_data.py \
        --transcript_path /path/to/aep_transcripts.json \
        --output_dir      /path/to/save/dataset
"""

import argparse
import json
import logging
import os
import random
from typing import List

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_transcripts(path: str, eos: str) -> List[dict]:
    with open(path) as f:
        records = json.load(f)
    docs = []
    for r in records:
        title = r.get("title", "").strip()
        text  = r.get("transcript", "").strip()
        if text:
            docs.append({"source": "transcript", "text": f"[Title: {title}]\n{text}"})
    logger.info("Loaded %d transcript documents", len(docs))
    return docs


def load_c4_docs(token_budget: int, tokenizer, eos: str) -> List[dict]:
    logger.info("Streaming C4 until ~%dK token budget …", token_budget // 1000)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    docs, collected = [], 0
    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue
        collected += len(tokenizer.encode(text + eos, add_special_tokens=False))
        docs.append({"source": "c4", "text": text})
        if collected >= token_budget:
            break
    logger.info("Collected %d C4 documents (~%d tokens)", len(docs), collected)
    return docs


def pack_into_chunks(docs: List[dict], tokenizer, max_length: int, eos: str) -> List[List[int]]:
    """Tokenise each doc, append EOS, concatenate, split into fixed-length chunks."""
    all_ids: List[int] = []
    for doc in docs:
        all_ids.extend(tokenizer.encode(doc["text"] + eos, add_special_tokens=False))
    n_chunks = len(all_ids) // max_length
    logger.info("Total tokens: %d → %d chunks of %d", len(all_ids), n_chunks, max_length)
    return [all_ids[i * max_length:(i + 1) * max_length] for i in range(n_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript_path", required=True)
    ap.add_argument("--output_dir",      required=True)
    ap.add_argument("--model_name",      default="EleutherAI/pythia-410m")
    ap.add_argument("--c4_ratio",        type=float, default=0.30,
                    help="Fraction of total tokens from C4 (default 30%%)")
    ap.add_argument("--max_length",      type=int,   default=1024)
    ap.add_argument("--eval_fraction",   type=float, default=0.05)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--reuse_docs_jsonl", default=None,
                    help="If set, load the exact documents from this merged_docs.jsonl "
                         "instead of re-reading transcripts and streaming C4. Use this to "
                         "train a second model (e.g. GPT-2) on identical text for a fair "
                         "comparison; only the tokenizer/packing differs.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token

    rng = random.Random(args.seed)

    if args.reuse_docs_jsonl:
        # ── reuse exact documents from a previous run (identical text) ─────────
        logger.info("Reusing documents from %s", args.reuse_docs_jsonl)
        with open(args.reuse_docs_jsonl) as f:
            all_docs = [json.loads(line) for line in f if line.strip()]
        n_tr = sum(d.get("source") == "transcript" for d in all_docs)
        n_c4 = sum(d.get("source") == "c4" for d in all_docs)
        logger.info("Loaded %d documents (%d transcript + %d C4) — identical text",
                    len(all_docs), n_tr, n_c4)
    else:
        # ── transcripts ────────────────────────────────────────────────────────
        transcript_docs = load_transcripts(args.transcript_path, eos)
        transcript_tokens = sum(
            len(tokenizer.encode(d["text"] + eos, add_special_tokens=False))
            for d in transcript_docs
        )
        logger.info("Transcript tokens: %d", transcript_tokens)

        # ── C4 ─────────────────────────────────────────────────────────────────
        c4_budget = int(transcript_tokens * args.c4_ratio / (1.0 - args.c4_ratio))
        logger.info(
            "C4 token budget: %d  (%.0f%% transcript / %.0f%% C4)",
            c4_budget, (1 - args.c4_ratio) * 100, args.c4_ratio * 100,
        )
        c4_docs = load_c4_docs(c4_budget, tokenizer, eos)

        # ── merge & shuffle ──────────────────────────────────────────────────────
        all_docs = transcript_docs + c4_docs
        rng.shuffle(all_docs)

    # ── save raw JSONL (static, inspectable) ─────────────────────────────────
    jsonl_path = os.path.join(args.output_dir, "merged_docs.jsonl")
    with open(jsonl_path, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Saved %d raw documents → %s", len(all_docs), jsonl_path)

    # ── pack into 1024-token chunks ───────────────────────────────────────────
    chunks = pack_into_chunks(all_docs, tokenizer, args.max_length, eos)
    rng.shuffle(chunks)

    # ── train / eval split & save Arrow datasets ──────────────────────────────
    n_eval = max(1, int(len(chunks) * args.eval_fraction))
    train_chunks = chunks[n_eval:]
    eval_chunks  = chunks[:n_eval]

    train_ds = Dataset.from_dict({"input_ids": train_chunks})
    eval_ds  = Dataset.from_dict({"input_ids": eval_chunks})

    train_ds.save_to_disk(os.path.join(args.output_dir, "train"))
    eval_ds.save_to_disk(os.path.join(args.output_dir, "eval"))

    logger.info("Train: %d chunks | Eval: %d chunks", len(train_ds), len(eval_ds))
    logger.info("Arrow datasets saved to %s/{train,eval}", args.output_dir)

    # ── sanity decode ─────────────────────────────────────────────────────────
    sample = train_ds[0]["input_ids"]
    logger.info("Sample chunk (first 60 tokens decoded):\n  %s …",
                tokenizer.decode(sample[:60]))


if __name__ == "__main__":
    main()
