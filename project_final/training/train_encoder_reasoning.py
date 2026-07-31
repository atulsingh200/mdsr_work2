

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# ── path setup ─────────────────────────────────────────────────────────────────
# All helper modules are in the same directory as this script (training/)
_TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TRAINING_DIR))

from data import CEReasoningCollate, CEPairReasoningDataset, _parse_step, strip_verdict  # noqa: E402
from train_utils import device_auto, make_logger, set_seed                               # noqa: E402

from model_ce2x import CrossEncoder2xWithReasoning, get_tokenizer  # noqa: E402  (after sys.path)


# ── helpers (mirrors train_ce2x.py) ────────────────────────────────────────────

def scan_max_seq_len(jsonls: list[Path], tokenizer) -> int:
    """Longest joint-tokenised length ([CLS] text_1 [SEP] text_2 [SEP]) across splits."""
    mx = 0
    for p in jsonls:
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                n = len(tokenizer.encode(r["text_1"], r["text_2"],
                                         add_special_tokens=True,
                                         max_length=512, truncation=True))
                if n > mx:
                    mx = n
    return mx


def scan_max_reason_len(jsonls: list[Path], tokenizer) -> int:
    """Longest tokenised (verdict-stripped) reasoning trace across splits."""
    mx = 0
    for p in jsonls:
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                reasoning = strip_verdict(r["reasoning"])
                n = len(tokenizer.encode(reasoning, add_special_tokens=True,
                                         max_length=1024, truncation=True))
                if n > mx:
                    mx = n
    return mx


@torch.no_grad()
def generate_reasoning(model, enc, attention_mask, tokenizer, device, use_bf16,
                        max_new_tokens: int) -> str:
    """Greedy-decode a single reasoning trace for qualitative inspection."""
    model.eval()
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    cur = torch.tensor([[tokenizer.cls_token_id]], device=device, dtype=torch.long)
    with ctx():
        enc_out = model.backbone(input_ids=enc, attention_mask=attention_mask)
        hidden_states = enc_out.last_hidden_state
        embed_w = model.backbone.get_input_embeddings().weight
        for _ in range(max_new_tokens):
            seq_len = cur.size(1)
            positions = torch.arange(seq_len, device=device)
            tgt = torch.nn.functional.embedding(cur, embed_w) + model.reason_pos_emb(positions)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
            )
            dec_out = model.reason_decoder(
                tgt, hidden_states, tgt_mask=causal_mask,
                memory_key_padding_mask=(attention_mask == 0),
            )
            logits = torch.nn.functional.linear(dec_out[:, -1, :], embed_w, model.lm_bias)
            next_id = logits.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_id], dim=1)
            if next_id.item() == tokenizer.sep_token_id:
                break
    return tokenizer.decode(cur[0, 1:], skip_special_tokens=True)


@torch.no_grad()
def evaluate(model, dl, device, use_bf16, cls_loss_fn, decoder_loss_fn,
             cls_w, dec_w, dataset=None, return_predictions=False):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = total_cls_loss = total_dec_loss = 0.0
    total_n = total_dec_tok = 0
    rows = [] if return_predictions else None
    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    for batch in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        labels = batch["labels"].to(device, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
        dec_lbl = batch["decoder_labels"].to(device, non_blocking=True)
        dec_pad = batch["decoder_padding_mask"].to(device, non_blocking=True)

        with ctx():
            cls_logits, dec_logits = model(
                enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"),
                decoder_input_ids=dec_in, decoder_padding_mask=dec_pad,
            )
            cls_loss = cls_loss_fn(cls_logits, labels)
            dec_loss = decoder_loss_fn(dec_logits.reshape(-1, dec_logits.size(-1)), dec_lbl.reshape(-1))
            loss = cls_w * cls_loss + dec_w * dec_loss

        n_tok = (dec_lbl != -100).sum().item()
        probs = torch.sigmoid(cls_logits).float().cpu().numpy()
        lbls = labels.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(lbls)
        total_loss += loss.item() * labels.size(0)
        total_cls_loss += cls_loss.item() * labels.size(0)
        total_dec_loss += dec_loss.item() * n_tok
        total_n += labels.size(0)
        total_dec_tok += n_tok
        if return_predictions and dataset is not None:
            for j, i_ds in enumerate(batch["idx"]):
                r = dataset.rows[i_ds]
                k1 = f"{dataset.step_key}_1"
                k2 = f"{dataset.step_key}_2"
                rows.append({
                    "url_1": r.get("url_1", ""), "url_2": r.get("url_2", ""),
                    "sub_1": r.get("sub_1", ""), "sub_2": r.get("sub_2", ""),
                    "step_1": r.get(k1), "step_2": r.get(k2),
                    "label": int(lbls[j]), "prob": float(probs[j]),
                })

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels).astype(int)
    preds = (probs_arr > 0.5).astype(int)
    acc = float((preds == labels_arr).mean())
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        auc = float(roc_auc_score(labels_arr, probs_arr))
        f1 = float(f1_score(labels_arr, preds))
    except Exception:
        auc = f1 = float("nan")
    dec_loss_avg = total_dec_loss / max(total_dec_tok, 1)
    return ({"loss": total_loss / max(total_n, 1),
             "cls_loss": total_cls_loss / max(total_n, 1),
             "decoder_loss": dec_loss_avg,
             "decoder_ppl": math.exp(min(dec_loss_avg, 20.0)),
             "acc": acc, "auc": auc, "f1": f1, "n": total_n}, rows)


def _per_step_pair(preds_rows: list[dict]) -> dict:
    pair_stats: dict = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in preds_rows:
        s1 = _parse_step(row.get("step_1"))
        s2 = _parse_step(row.get("step_2"))
        if s1 is None or s2 is None:
            continue
        key = (min(s1, s2), max(s1, s2))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int((row["prob"] > 0.5) == (row["label"] == 1))
    return {f"S{a}-S{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
            for (a, b), s in sorted(pair_stats.items())}


def _write_preds(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train CrossEncoder2xWithReasoning (CE2x + auxiliary reasoning decoder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True,
                    help="Directory with <split-prefix>{train,val,test}.jsonl (rows need a 'reasoning' field)")
    ap.add_argument("--split-prefix", default="",
                    help="Filename prefix for split files: <prefix>{train,val,test}.jsonl.")
    ap.add_argument("--out-dir", default="runs/crossencoder2x_deberta_reasoning")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--backbone", default="microsoft/deberta-v3-large")
    ap.add_argument("--n-layers", type=int, default=24,
                    help="Number of transformer layers (24 = 2× BERT-base, matches BiEncoder param budget).")
    ap.add_argument("--native-backbone", action="store_true",
                    help="Use AutoModel.from_pretrained directly instead of the BERT-stacking logic. "
                         "Required for non-BERT backbones such as DeBERTa-v3-large.")
    ap.add_argument("--decoder-layers", type=int, default=4,
                    help="Number of Transformer decoder layers for the reasoning head.")
    ap.add_argument("--cls-loss-weight", type=float, default=1.0,
                    help="Weight on the BCE classification loss. Spec loss is "
                         "L = L_BCE + alpha*L_LM, so keep this at 1.0 and set "
                         "--decoder-loss-weight to alpha.")
    ap.add_argument("--decoder-loss-weight", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="Override joint sequence length cap; default = auto-scan.")
    ap.add_argument("--max-reason-len", type=int, default=None,
                    help="Override reasoning target length cap; default = auto-scan.")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Cap the number of training rows (smoke testing only).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--num-qual-samples", type=int, default=3,
                    help="Number of fixed val examples to greedy-decode and log each epoch.")
    ap.add_argument("--model-ce2x-dir", default=str(_CE2X_MODEL_DIR),
                    help="Directory that contains model_ce2x.py.")
    ap.add_argument("--step-key", default="tier",
                    help="Field-name prefix for step/tier columns. "
                         "Use 'tier' for datasets with tier_1/tier_2 (int), "
                         "'step' for datasets with step_1/step_2 (e.g. 't2' strings).")
    args = ap.parse_args()

    if args.model_ce2x_dir != str(_CE2X_MODEL_DIR):
        sys.path.insert(0, args.model_ce2x_dir)

    set_seed(args.seed)
    device = device_auto()
    use_bf16 = device.type == "cuda"

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}  device={device}  bf16={use_bf16}")

    tokenizer = get_tokenizer(args.backbone)
    log.info(f"tokenizer <- {args.backbone}")

    data_dir = Path(args.data_dir)
    jsonls = [data_dir / f"{args.split_prefix}{s}.jsonl" for s in ("train", "val", "test")]

    if args.max_seq_len is None:
        log.info("scanning max joint token length across all splits …")
        t0 = time.time()
        mx = scan_max_seq_len(jsonls, tokenizer)
        model_cap = getattr(tokenizer, "model_max_length", 512)
        if model_cap is None or model_cap > 100_000:
            model_cap = 512
        args.max_seq_len = min(model_cap, max(128, int(np.ceil(mx * 1.05)) + 2))
        log.info(f"  observed max={mx}  -> max_seq_len={args.max_seq_len}  (scan {time.time()-t0:.1f}s)")
    else:
        log.info(f"max_seq_len={args.max_seq_len} (user-provided)")

    if args.max_reason_len is None:
        log.info("scanning max reasoning token length across all splits …")
        t0 = time.time()
        mx = scan_max_reason_len(jsonls, tokenizer)
        args.max_reason_len = max(32, int(np.ceil(mx * 1.05)) + 2)
        log.info(f"  observed max={mx}  -> max_reason_len={args.max_reason_len}  (scan {time.time()-t0:.1f}s)")
    else:
        log.info(f"max_reason_len={args.max_reason_len} (user-provided)")

    log.info("loading datasets …")
    ds_train = CEPairReasoningDataset(jsonls[0], step_key=args.step_key)
    if args.max_train_samples is not None:
        ds_train.rows = ds_train.rows[:args.max_train_samples]
    ds_val   = CEPairReasoningDataset(jsonls[1], step_key=args.step_key)
    ds_test  = CEPairReasoningDataset(jsonls[2], step_key=args.step_key)
    log.info(f"  train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    collate = CEReasoningCollate(tokenizer, args.max_seq_len, args.max_reason_len)
    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)

    log.info(f"building CrossEncoder2xWithReasoning  backbone={args.backbone}  "
             f"n_layers={args.n_layers}  native={args.native_backbone}  "
             f"decoder_layers={args.decoder_layers} …")
    model = CrossEncoder2xWithReasoning(
        backbone=args.backbone, n_layers=args.n_layers, dropout=args.dropout,
        native_backbone=args.native_backbone, decoder_layers=args.decoder_layers,
        max_reason_len=args.max_reason_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  total params={n_params:,}")

    cls_loss_fn = nn.BCEWithLogitsLoss()
    decoder_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=args.lr)

    total_steps  = (len(dl_train) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(f"  total_steps={total_steps}  warmup_steps={warmup_steps}")

    cfg = {
        "backbone": args.backbone, "n_layers": args.n_layers,
        "native_backbone": args.native_backbone,
        "decoder_layers": args.decoder_layers,
        "cls_loss_weight": args.cls_loss_weight, "decoder_loss_weight": args.decoder_loss_weight,
        "data_dir": str(args.data_dir), "split_prefix": args.split_prefix,
        "run_name": args.run_name,
        "model_type": "cross_encoder_2x_with_reasoning",
        "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum, "eff_batch_size": args.batch_size * args.grad_accum,
        "epochs": args.epochs, "warmup_frac": args.warmup_frac,
        "grad_clip": args.grad_clip, "dropout": args.dropout, "seed": args.seed,
        "max_seq_len": args.max_seq_len, "max_reason_len": args.max_reason_len,
        "device": str(device), "bf16": use_bf16,
        "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "n_params": n_params,
        "pool": "cls", "input_form": "[CLS] text_1 [SEP] text_2 [SEP]",
        "head": "Linear(hidden, 1) + TransformerDecoder(reasoning, verdict-stripped)",
        "loss": "cls_w*BCEWithLogitsLoss + dec_w*CrossEntropyLoss",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    best_val_auc = -1.0
    opt_step = 0
    t_start = time.time()
    qual_batch = next(iter(dl_val))  # fixed batch for qualitative decoding each epoch

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = running_cls_loss = running_dec_loss = 0.0
        running_correct = running_total = running_dec_tok = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(dl_train, 1):
            enc    = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device, non_blocking=True)
            dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
            dec_lbl = batch["decoder_labels"].to(device, non_blocking=True)
            dec_pad = batch["decoder_padding_mask"].to(device, non_blocking=True)

            with ctx():
                cls_logits, dec_logits = model(
                    enc["input_ids"], enc.get("attention_mask"), enc.get("token_type_ids"),
                    decoder_input_ids=dec_in, decoder_padding_mask=dec_pad,
                )
                cls_loss = cls_loss_fn(cls_logits, labels)
                dec_loss = decoder_loss_fn(dec_logits.reshape(-1, dec_logits.size(-1)), dec_lbl.reshape(-1))
                loss = (args.cls_loss_weight * cls_loss + args.decoder_loss_weight * dec_loss) / args.grad_accum

            loss.backward()
            bs = labels.size(0)
            n_tok = (dec_lbl != -100).sum().item()
            running_loss     += float(loss.item()) * args.grad_accum * bs
            running_cls_loss += float(cls_loss.item()) * bs
            running_dec_loss += float(dec_loss.item()) * n_tok
            preds = (torch.sigmoid(cls_logits.detach()) > 0.5).float()
            running_correct += (preds == labels).sum().item()
            running_total   += bs
            running_dec_tok += n_tok

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    log.info(f"  e{epoch} opt_step={opt_step}/{total_steps}  "
                             f"loss={running_loss / running_total:.4f}  "
                             f"cls_loss={running_cls_loss / running_total:.4f}  "
                             f"decoder_loss={running_dec_loss / max(running_dec_tok, 1):.4f}  "
                             f"acc={running_correct / running_total:.4f}  "
                             f"lr={sched.get_last_lr()[0]:.2e}")

        val_metrics, _ = evaluate(model, dl_val, device, use_bf16, cls_loss_fn, decoder_loss_fn,
                                   args.cls_loss_weight, args.decoder_loss_weight)
        log.info(f"[epoch {epoch}/{args.epochs}]  "
                 f"train_loss={running_loss / running_total:.4f}  "
                 f"train_cls_loss={running_cls_loss / running_total:.4f}  "
                 f"train_decoder_loss={running_dec_loss / max(running_dec_tok, 1):.4f}  "
                 f"train_acc={running_correct / running_total:.4f}  "
                 f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
                 f"val_auc={val_metrics['auc']:.4f}  val_decoder_ppl={val_metrics['decoder_ppl']:.2f}  "
                 f"elapsed={time.time()-t_epoch:.1f}s")

        for i in range(min(args.num_qual_samples, qual_batch["enc"]["input_ids"].size(0))):
            enc_i = qual_batch["enc"]["input_ids"][i:i+1].to(device)
            am_i = qual_batch["enc"]["attention_mask"][i:i+1].to(device)
            gen = generate_reasoning(model, enc_i, am_i, tokenizer, device, use_bf16,
                                      max_new_tokens=args.max_reason_len)
            gold = ds_val.rows[qual_batch["idx"][i]]["reasoning"]
            gold = strip_verdict(gold)
            log.info(f"  [qual {i}] gen : {gen}")
            log.info(f"  [qual {i}] gold: {gold}")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "cfg": cfg}
        torch.save(ckpt, run_dir / "checkpoint_latest.pt")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(ckpt, run_dir / "best.pt")
            log.info(f"  * saved best.pt  val_auc={best_val_auc:.4f}")

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
               run_dir / "final.pt")
    log.info(f"saved final.pt  total_time={time.time()-t_start:.0f}s")

    # ── reload best for val + test eval ───────────────────────────────────────
    log.info("loading best.pt for final val/test evaluation …")
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    val_metrics, val_rows = evaluate(model, dl_val, device, use_bf16, cls_loss_fn, decoder_loss_fn,
                                     args.cls_loss_weight, args.decoder_loss_weight,
                                     dataset=ds_val, return_predictions=True)
    _write_preds(run_dir / "val_predictions.jsonl", val_rows)
    log.info(f"[val]  acc={val_metrics['acc']:.4f}  auc={val_metrics['auc']:.4f}  "
             f"f1={val_metrics['f1']:.4f}  decoder_ppl={val_metrics['decoder_ppl']:.2f}")

    test_metrics, test_rows = evaluate(model, dl_test, device, use_bf16, cls_loss_fn, decoder_loss_fn,
                                       args.cls_loss_weight, args.decoder_loss_weight,
                                       dataset=ds_test, return_predictions=True)
    log.info(f"[test] loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
             f"auc={test_metrics['auc']:.4f}  f1={test_metrics['f1']:.4f}  "
             f"decoder_ppl={test_metrics['decoder_ppl']:.2f}")

    per_pair = _per_step_pair(test_rows)
    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_step_pair": per_pair}, indent=2))
    _write_preds(run_dir / "test_predictions.jsonl", test_rows)

    log.info(f"[done] best_val_auc={best_val_auc:.4f}  "
             f"test_acc={test_metrics['acc']:.4f}  test_auc={test_metrics['auc']:.4f}  "
             f"results in {run_dir}")


if __name__ == "__main__":
    main()
