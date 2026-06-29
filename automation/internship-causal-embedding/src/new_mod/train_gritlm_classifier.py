"""
GritLM-7B full fine-tune: directional causal classification.

Architecture:
  GritLM-7B backbone (all weights trainable, no LoRA)
  instruction prefix + "PASSAGE 1: {text_1}\\n\\nPASSAGE 2: {text_2}"
  mean-pool over content tokens (instruction prefix masked out)
  -> Linear(4096, 1) -> raw logit (unbounded scalar)

Training loss: BCEWithLogitsLoss (sigmoid applied internally)

Inference:
  prob = sigmoid(logit)        # 0-1, model confidence
  pred = int(prob > 0.5)       # hard decision
  label=1: text_1 tier causally precedes text_2 tier
  label=0: reversed/non-causal

Outputs (--out-dir/<run-name>/):
  config.json            hyperparams + architecture spec
  best.pt                best val-acc checkpoint (backbone + head state dict)
  final.pt               end-of-training checkpoint
  train.log              per-optimizer-step + per-epoch metrics
  test_metrics.json      acc / AUC / F1 + per-tier-pair breakdown
  test_predictions.jsonl one row per test sample: url, tiers, label, prob

Memory (4x A100-80GB, full fine-tune):
  BF16 model ~14 GB + BF16 gradients ~14 GB + 8-bit Adam ~14 GB = ~42 GB/GPU
  with DataParallel each GPU holds a full copy -> comfortably fits in 80 GB.

CLI example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gritlm_classifier.py \\
      --data-dir  data/aep_causal_classification \\
      --out-dir   runs/gritlm_classifier \\
      --epochs 3  --batch-size 8  --accum-steps 8 \\
      --lr-backbone 5e-6  --lr-head 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "<|user|>\n"
    "Given two passages from Adobe Journey Optimizer documentation, classify whether "
    "PASSAGE 1 is a causal prerequisite that must be understood before PASSAGE 2 in "
    "the learning hierarchy. Represent this pair so that a score closer to 1 means "
    "PASSAGE 1 causally precedes PASSAGE 2, and a score closer to 0 means the "
    "ordering is reversed or non-causal.\n"
    "<|embed|>\n"
)
PASSAGE_SEP = "\n\n"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GritLMClassifier(nn.Module):
    """GritLM backbone + mean-pool (content tokens only) + linear head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, instr_len: int):
        super().__init__()
        self.backbone  = backbone
        self.head      = nn.Linear(hidden_size, 1, bias=True)
        self.instr_len = instr_len
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_causal=False,    # bidirectional attention for representation
            use_cache=False,    # DynamicCache.from_legacy_cache removed in transformers 5.x
        )
        # last_hidden_state or out[0] depending on model version
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = h.float()                                  # (B, seq, 4096) in float32

        # mask instruction prefix so it doesn't pollute the content representation
        mask = attention_mask.clone().float()          # (B, seq)
        if self.instr_len > 0:
            mask[:, :self.instr_len] = 0.0
        mask = mask.unsqueeze(-1)                      # (B, seq, 1)

        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 4096)
        # L2-normalize onto the unit sphere before the linear head.
        # This bounds |logit| <= ||w||₂ (Cauchy-Schwarz) regardless of embedding
        # magnitude, preventing the all-positive-logit collapse that occurs when
        # GritLM embeddings live in a biased half-space.
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)   # (B, 4096), unit norm
        logit  = self.head(pooled).squeeze(-1)         # (B,)  bounded scalar
        return logit


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, path: Path):
        self.rows: list[dict] = []
        with open(path) as f:
            for ln in f:
                if ln.strip():
                    self.rows.append(json.loads(ln))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {
            "text_1": r["text_1"],
            "text_2": r["text_2"],
            "label":  float(r["label"]),
            "tier_1": r["tier_1"],
            "tier_2": r["tier_2"],
            "url_1":  r.get("url_1", ""),
            "url_2":  r.get("url_2", ""),
            "idx":    i,
        }


class Collate:
    def __init__(self, tokenizer, max_len: int, instruction: str):
        self.tok         = tokenizer
        self.max_len     = max_len
        self.instruction = instruction

    def _format(self, t1: str, t2: str) -> str:
        return self.instruction + "PASSAGE 1: " + t1 + PASSAGE_SEP + "PASSAGE 2: " + t2

    def __call__(self, batch: list[dict]) -> dict:
        texts = [self._format(b["text_1"], b["text_2"]) for b in batch]
        enc   = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":  torch.tensor([b["label"]  for b in batch], dtype=torch.float32),
            "tier_1":  [b["tier_1"] for b in batch],
            "tier_2":  [b["tier_2"] for b in batch],
            "url_1":   [b["url_1"]  for b in batch],
            "url_2":   [b["url_2"]  for b in batch],
            "idx":     [b["idx"]    for b in batch],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    dataset: PairDataset | None = None,
    return_predictions: bool = False,
) -> tuple[dict, list[dict] | None]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss, total_n = 0.0, 0
    rows: list[dict] | None = [] if return_predictions else None

    for batch in dl:
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lbl  = batch["labels"].to(device, non_blocking=True)

        logits = model(ids, mask)
        loss   = loss_fn(logits, lbl)

        # sigmoid here only to get calibrated probabilities for metrics
        probs = torch.sigmoid(logits).float().cpu().numpy()
        lbls  = lbl.cpu().numpy()

        all_probs.append(probs)
        all_labels.append(lbls)
        total_loss += loss.item() * lbl.size(0)
        total_n    += lbl.size(0)

        if return_predictions and dataset is not None:
            for j, idx in enumerate(batch["idx"]):
                r = dataset.rows[idx]
                rows.append({
                    "url_1":  r.get("url_1", ""),
                    "url_2":  r.get("url_2", ""),
                    "tier_1": r["tier_1"],
                    "tier_2": r["tier_2"],
                    "label":  int(lbls[j]),
                    "prob":   float(probs[j]),
                })

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    # threshold at 0.5 on prob == threshold at 0 on raw logit (mathematically identical)
    preds  = (probs > 0.5).astype(int)
    acc    = float((preds == labels).mean())

    # prediction-distribution diagnostics — catches "all positive" collapse
    frac_pred_pos  = float(preds.mean())           # fraction predicted as class 1
    mean_prob      = float(probs.mean())
    mean_prob_pos  = float(probs[labels == 1].mean()) if (labels == 1).any() else float("nan")
    mean_prob_neg  = float(probs[labels == 0].mean()) if (labels == 0).any() else float("nan")

    try:
        from sklearn.metrics import f1_score, roc_auc_score
        auc = float(roc_auc_score(labels, probs))
        f1  = float(f1_score(labels, preds))
    except Exception:
        auc = f1 = float("nan")

    return (
        {
            "loss":          total_loss / max(total_n, 1),
            "acc":           acc,
            "auc":           auc,
            "f1":            f1,
            "n":             total_n,
            "frac_pred_pos": frac_pred_pos,    # should be ~0.5 for a healthy model
            "mean_prob":     mean_prob,         # should be ~0.5 overall
            "mean_prob_pos": mean_prob_pos,     # prob when true label=1 (want >0.5)
            "mean_prob_neg": mean_prob_neg,     # prob when true label=0 (want <0.5)
        },
        rows,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("train_gritlm")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full fine-tune GritLM-7B for directional causal classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir",        default="data/aep_causal_classification")
    ap.add_argument("--out-dir",         default="runs/gritlm_classifier")
    ap.add_argument("--base-model",      default="GritLM/GritLM-7B")
    ap.add_argument("--run-name",        default=None)
    ap.add_argument("--epochs",          type=int,   default=3)
    ap.add_argument("--batch-size",      type=int,   default=4,
                    help="Total batch size per gradient-accumulation micro-step "
                         "(split evenly across GPUs by DataParallel).")
    ap.add_argument("--eval-batch-size", type=int,   default=8)
    ap.add_argument("--accum-steps",     type=int,   default=16,
                    help="Gradient accumulation steps. "
                         "Effective batch = batch-size * accum-steps.")
    ap.add_argument("--max-seq-len",     type=int,   default=512)
    ap.add_argument("--lr-backbone",     type=float, default=5e-6,
                    help="Learning rate for the GritLM backbone.")
    ap.add_argument("--lr-head",         type=float, default=1e-4,
                    help="Learning rate for the linear classification head.")
    ap.add_argument("--warmup-frac",     type=float, default=0.06)
    ap.add_argument("--weight-decay",    type=float, default=0.01)
    ap.add_argument("--grad-clip",       type=float, default=1.0)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--num-workers",     type=int,   default=2)
    ap.add_argument("--log-every",       type=int,   default=20,
                    help="Log every N optimizer steps.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gritlm"
    run_dir  = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}")
    log.info(f"device={device}  n_gpus={n_gpus}  optimizer={'AdamW8bit' if HAS_BNB else 'AdamW'}")

    # -- tokenizer --
    log.info(f"Loading tokenizer <- {args.base_model}")
    tok = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right"
    )
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    instr_len = len(tok(INSTRUCTION, add_special_tokens=True)["input_ids"])
    log.info(f"Instruction prefix length: {instr_len} tokens  max_seq_len={args.max_seq_len}")

    # -- datasets --
    data_dir = Path(args.data_dir)
    ds_train = PairDataset(data_dir / "directional_train.jsonl")
    ds_val   = PairDataset(data_dir / "directional_val.jsonl")
    ds_test  = PairDataset(data_dir / "directional_test.jsonl")
    log.info(f"train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    collate  = Collate(tok, args.max_seq_len, INSTRUCTION)
    pin      = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size,      shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=pin)

    # -- backbone --
    log.info(f"Loading backbone {args.base_model} (bfloat16) ...")
    # Load config first and patch any attributes the cached GritLM code expects
    # but that newer transformers versions no longer expose on MistralConfig.
    from transformers import AutoConfig
    backbone_cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if not hasattr(backbone_cfg, "rope_theta"):
        backbone_cfg.rope_theta = 10000.0          # Mistral default
    if not hasattr(backbone_cfg, "sliding_window"):
        backbone_cfg.sliding_window = None

    backbone = AutoModel.from_pretrained(
        args.base_model,
        config=backbone_cfg,
        trust_remote_code=True,
        dtype=torch.bfloat16,                      # `dtype` replaces deprecated `torch_dtype`
    ).to(device)

    # Gradient checkpointing is intentionally NOT used here.
    # With DataParallel, enable_input_require_grads() hooks are not replicated
    # to per-GPU copies, silently freezing the backbone. Each A100-80GB has
    # enough headroom (model 14GB + grad 14GB + 8-bit Adam 7GB + activations
    # ~4GB = ~39GB) to hold full activations without checkpointing.

    model = GritLMClassifier(backbone, backbone.config.hidden_size, instr_len).to(device)

    if n_gpus > 1:
        log.info(f"DataParallel across {n_gpus} GPUs.")
        model = nn.DataParallel(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_trainable:,}")

    # -- optimizer: 8-bit Adam to keep optimizer states ~7 GB instead of 56 GB --
    raw_model  = unwrap(model)
    param_groups = [
        {"params": list(raw_model.backbone.parameters()),
         "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": list(raw_model.head.parameters()),
         "lr": args.lr_head,     "weight_decay": 0.0},
    ]
    if HAS_BNB:
        opt = bnb.optim.AdamW8bit(param_groups)
        log.info("Optimizer: bitsandbytes AdamW8bit")
    else:
        opt = torch.optim.AdamW(param_groups)
        log.warning("bitsandbytes not found; using torch AdamW (expect higher GPU memory use).")

    # steps counted at optimizer level (after accumulation)
    steps_per_epoch = len(dl_train) // args.accum_steps
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * args.warmup_frac)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    log.info(
        f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
        f"warmup={warmup_steps}  effective_batch={args.batch_size * args.accum_steps}"
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # -- config --
    cfg = {
        "base_model":      args.base_model,
        "run_name":        run_name,
        "architecture":    "GritLM-7B + mean-pool (content only) + Linear(4096,1)",
        "instruction":     INSTRUCTION,
        "instr_len_tokens": instr_len,
        "pool":            "mean (attention-mask, instruction prefix zeroed) then L2-normalize",
        "head":            "L2-norm(4096) -> Linear(4096, 1) -> raw logit",
        "loss":            "BCEWithLogitsLoss",
        "inference":       "prob = sigmoid(logit); pred = int(prob > 0.5)",
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "accum_steps":     args.accum_steps,
        "effective_batch": args.batch_size * args.accum_steps,
        "eval_batch_size": args.eval_batch_size,
        "max_seq_len":     args.max_seq_len,
        "lr_backbone":     args.lr_backbone,
        "lr_head":         args.lr_head,
        "warmup_frac":     args.warmup_frac,
        "weight_decay":    args.weight_decay,
        "grad_clip":       args.grad_clip,
        "seed":            args.seed,
        "n_gpus":          n_gpus,
        "optimizer":       "AdamW8bit" if HAS_BNB else "AdamW",
        "n_train":         len(ds_train),
        "n_val":           len(ds_val),
        "n_test":          len(ds_test),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # -- train loop --
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = running_correct = running_total = 0.0
        opt.zero_grad()

        for micro_step, batch in enumerate(dl_train, 1):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbl  = batch["labels"].to(device, non_blocking=True)

            logits = model(ids, mask)
            # divide loss by accum_steps before backward so gradients average correctly
            loss = loss_fn(logits, lbl) / args.accum_steps
            loss.backward()

            bs = lbl.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits.detach().float())
                running_correct += ((probs > 0.5).float() == lbl).sum().item()
            running_loss  += loss.item() * args.accum_steps * bs
            running_total += bs

            # optimizer step every accum_steps micro-batches
            if micro_step % args.accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()

                opt_step = micro_step // args.accum_steps
                if opt_step % args.log_every == 0 or opt_step == 1:
                    lrs = " ".join(
                        f"lr{i}={pg['lr']:.2e}" for i, pg in enumerate(opt.param_groups)
                    )
                    log.info(
                        f"  e{epoch} step {opt_step:>4}/{steps_per_epoch}  "
                        f"loss={running_loss / running_total:.4f}  "
                        f"acc={running_correct / running_total:.4f}  {lrs}"
                    )

        val_metrics, _ = evaluate(model, dl_val, device, loss_fn)
        log.info(
            f"[epoch {epoch}] "
            f"train_loss={running_loss / running_total:.4f}  "
            f"train_acc={running_correct / running_total:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}  elapsed={time.time() - t_epoch:.1f}s"
        )
        # collapse check: healthy model should have frac_pred_pos ≈ 0.5 and
        # mean_prob_pos > mean_prob_neg (model separates the two classes)
        log.info(
            f"  [dist] frac_pred_pos={val_metrics['frac_pred_pos']:.3f}  "
            f"mean_prob={val_metrics['mean_prob']:.3f}  "
            f"mean_prob|label=1={val_metrics['mean_prob_pos']:.3f}  "
            f"mean_prob|label=0={val_metrics['mean_prob_neg']:.3f}"
        )

        state = {
            "model": unwrap(model).state_dict(),
            "epoch": epoch,
            "val":   val_metrics,
            "cfg":   cfg,
        }
        torch.save(state, run_dir / "final.pt")
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(state, run_dir / "best.pt")
            log.info(f"  saved best.pt  val_acc={val_metrics['acc']:.4f}")

    # -- test with best checkpoint --
    log.info("Loading best.pt for test evaluation ...")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    unwrap(model).load_state_dict(ckpt["model"])

    test_metrics, pred_rows = evaluate(
        model, dl_test, device, loss_fn, dataset=ds_test, return_predictions=True
    )
    log.info(
        f"[test] loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
        f"auc={test_metrics['auc']:.4f}  f1={test_metrics['f1']:.4f}"
    )

    # per-tier-pair accuracy (same breakdown as the BGE-small classifier)
    pair_stats: dict[tuple[int, int], dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in pred_rows:
        key = (min(row["tier_1"], row["tier_2"]), max(row["tier_1"], row["tier_2"]))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int((row["prob"] > 0.5) == (row["label"] == 1))
    per_pair = {
        f"T{a}-T{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
        for (a, b), s in sorted(pair_stats.items())
    }
    log.info("[test] per-tier-pair accuracy:")
    for k, v in per_pair.items():
        log.info(f"    {k:>9}  n={v['n']:>5}  acc={v['acc']:.4f}")

    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_tier_pair": per_pair}, indent=2)
    )
    with open(run_dir / "test_predictions.jsonl", "w") as f:
        for row in pred_rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"[done] results in {run_dir}")


if __name__ == "__main__":
    main()
