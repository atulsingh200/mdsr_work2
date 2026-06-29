"""
Step 2: Train Pythia-410M on the pre-built dataset from prepare_data.py.

Single-GPU training (no DDP / torchrun).

Usage
-----
    # quick smoke test
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --dataset_dir /path/to/dataset --output_dir /path/to/output --max_steps 2

    # full run
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --dataset_dir /path/to/dataset --output_dir /path/to/output
"""

import argparse
import logging
import math
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",   default="EleutherAI/pythia-410m")
    p.add_argument("--dataset_dir",  required=True,
                   help="Directory with train/ and eval/ Arrow splits (from prepare_data.py)")
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--num_train_epochs",              type=int,   default=20)
    p.add_argument("--per_device_train_batch_size",   type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps",   type=int,   default=8)
    p.add_argument("--learning_rate",                 type=float, default=2e-5)
    p.add_argument("--weight_decay",                  type=float, default=0.0)
    p.add_argument("--warmup_steps",                  type=int,   default=100)
    p.add_argument("--save_steps",                    type=int,   default=200)
    p.add_argument("--eval_steps",                    type=int,   default=200)
    p.add_argument("--logging_steps",                 type=int,   default=50)
    p.add_argument("--seed",                          type=int,   default=42)
    p.add_argument("--max_steps",                     type=int,   default=-1,
                   help="Set to 2 for a quick smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── model + tokenizer ────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    logger.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # ── dataset ──────────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.dataset_dir)
    train_dataset = load_from_disk(os.path.join(args.dataset_dir, "train"))
    eval_dataset  = load_from_disk(os.path.join(args.dataset_dir, "eval"))
    logger.info("Train: %d chunks | Eval: %d chunks", len(train_dataset), len(eval_dataset))

    # ── trainer ──────────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,               # gradient clipping for stability
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,   # single-GPU; keep it simple
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,        # 0 avoids forked-worker SIGSEGV on this host
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # attach perplexity to logged metrics
    _orig_log = trainer.log
    def _log_ppl(logs, *args, **kwargs):
        for loss_key, ppl_key in [("loss", "train_perplexity"), ("eval_loss", "eval_perplexity")]:
            if loss_key in logs:
                try:
                    logs[ppl_key] = round(math.exp(logs[loss_key]), 2)
                except OverflowError:
                    pass
        _orig_log(logs, *args, **kwargs)
    trainer.log = _log_ppl

    logger.info("Starting training …")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # dump full log history (loss / lr / eval) for plotting
    import json
    log_path = os.path.join(args.output_dir, "log_history.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    logger.info("Saved log history to %s", log_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
