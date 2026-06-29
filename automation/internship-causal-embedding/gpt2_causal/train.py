import argparse
import csv
import os
import math
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from dataset import build_tokenizer, CausalPairDataset, label_token_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run only 10 steps on 100 examples to verify wiring")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, zero_id, one_id, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, seq_len, vocab]

        # The label token is always the last non-pad position.
        # Because we padded on the right, find last real token per row.
        seq_lens = attention_mask.sum(dim=1) - 1  # index of label token
        batch_idx = torch.arange(input_ids.size(0), device=device)
        last_logits = logits[batch_idx, seq_lens]  # [B, vocab]

        score_0 = last_logits[:, zero_id]
        score_1 = last_logits[:, one_id]
        preds   = (score_1 > score_0).long()

        # Ground truth: 1 if label token == one_id, else 0
        gt_tokens = labels[batch_idx, seq_lens]
        gt = (gt_tokens == one_id).long()

        correct += (preds == gt).sum().item()
        total   += input_ids.size(0)

    model.train()
    return correct / total if total else 0.0


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenizer + model
    tokenizer = build_tokenizer(cfg["model_name"])
    zero_id, one_id = label_token_ids(tokenizer)

    model = GPT2LMHeadModel.from_pretrained(cfg["model_name"])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Datasets
    train_ds = CausalPairDataset(cfg["train_path"], tokenizer, cfg["max_len"])
    val_ds   = CausalPairDataset(cfg["val_path"],   tokenizer, cfg["max_len"])

    if args.smoke_test:
        train_ds.records = train_ds.records[:100]
        val_ds.records   = val_ds.records[:50]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=train_ds.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=4,
        collate_fn=val_ds.collate_fn,
    )

    # Optimizer + scheduler
    num_update_steps = math.ceil(len(train_loader) / cfg["grad_accum_steps"]) * cfg["num_epochs"]
    warmup_steps     = int(num_update_steps * cfg["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"])

    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_path = os.path.join(cfg["output_dir"], "train_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_acc"])

    best_val_acc = 0.0
    global_step  = 0
    max_steps    = 10 if args.smoke_test else None

    model.train()
    optimizer.zero_grad()

    for epoch in range(cfg["num_epochs"]):
        for step, batch in enumerate(train_loader):
            if max_steps and global_step >= max_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg["fp16"]):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    print(f"epoch {epoch+1}  step {global_step}  loss {loss.item() * cfg['grad_accum_steps']:.4f}")

        if max_steps and global_step >= max_steps:
            break

        val_acc = evaluate(model, val_loader, zero_id, one_id, device)
        print(f"=== Epoch {epoch+1} | val_acc={val_acc:.4f} ===")
        log_writer.writerow([global_step, "", val_acc])
        log_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(cfg["output_dir"], "best_model")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

    log_file.close()
    print(f"Training done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
