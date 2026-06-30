"""Train a CROSS-ENCODER directional classifier on text pairs (baseline sweep).

Cross-encoder counterpart of training/train_baselines.py:
  - Single transformer jointly encodes  [CLS] text_1 [SEP] text_2 [SEP],
    CLS-pools, and a Linear head maps to one directional logit.
  - Supports all 7 Adobe datasets (handles 3 different JSONL schemas).
  - Works with 6 BERT-type backbones including DeBERTa-v3-large.
  - --out-dir is the final run directory (no run-name subdirectory).
  - Writes run_summary.json + ood-compatible config.json for aggregation.

Outputs (in --out-dir/):
  config.json, best.pt, final.pt, train.log,
  test_metrics.json, test_predictions.jsonl, run_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Support both installed package and direct script execution
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[3] / "src"  # .../internship-causal-embedding/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from classifier.crossencoder.model import CrossEncoderClassifier  # noqa: E402

_SAFE_ENC_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})


# ---------------------------------------------------------------------------
# Schema detection (mirrors training/train_baselines.py)
# ---------------------------------------------------------------------------
def _detect_schema(first_row: dict) -> str:
    """'standard' (tier_1/2), 'newstyle' (heading_1/2), or 'step' (step_1/2)."""
    if "tier_1" in first_row:
        return "standard"
    if "heading_1" in first_row:
        return "newstyle"
    if "step_1" in first_row:
        return "step"
    return "standard"


# ---------------------------------------------------------------------------
# Dataset + joint-tokenizing collate
# ---------------------------------------------------------------------------
class CEPairDataset(Dataset):
    """In-memory labelled text pairs, normalised from any of the 3 schemas."""

    def __init__(self, path: Path):
        self.rows: list[dict] = []
        self.schema = "standard"
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    self.rows.append(json.loads(ln))
        if not self.rows:
            return
        self.schema = _detect_schema(self.rows[0])
        if self.schema == "newstyle":
            for r in self.rows:
                r.setdefault("tier_1", 0)
                r.setdefault("tier_2", 0)
                r.setdefault("sub_1", r.get("heading_1", ""))
                r.setdefault("sub_2", r.get("heading_2", ""))
        elif self.schema == "step":
            for r in self.rows:
                r.setdefault("tier_1", 0)
                r.setdefault("tier_2", 0)
                r.setdefault("sub_1", r.get("step_1", ""))
                r.setdefault("sub_2", r.get("step_2", ""))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {
            "text_1": r["text_1"],
            "text_2": r["text_2"],
            "label": float(r["label"]),
            "idx": i,
        }


class CECollate:
    """Joint-tokenize text pairs: [CLS] text_1 [SEP] text_2 [SEP]."""

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict:
        t1 = [b["text_1"] for b in batch]
        t2 = [b["text_2"] for b in batch]
        enc = self.tok(
            t1, t2, padding=True, truncation="longest_first",
            max_length=self.max_len, return_tensors="pt",
        )
        enc = {k: v for k, v in enc.items() if k in _SAFE_ENC_KEYS}
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return {"enc": enc, "labels": labels, "idx": [b["idx"] for b in batch]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def device_auto() -> torch.device:
    """Always cuda:0 inside the process; GPU isolation via CUDA_VISIBLE_DEVICES."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scan_max_seq_len(jsonls: list[Path], tokenizer) -> int:
    """Longest JOINT tokenized length (text_1 [SEP] text_2) across splits."""
    mx = 0
    for p in jsonls:
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                n = len(tokenizer.encode(
                    r["text_1"], r["text_2"],
                    add_special_tokens=True, max_length=512, truncation=True,
                ))
                if n > mx:
                    mx = n
    return mx


def find_split_files(data_dir: Path, split_prefix: str | None) -> list[Path]:
    """Return [train, val, test] paths, auto-detecting prefix if not given."""
    if split_prefix is not None:
        return [data_dir / f"{split_prefix}{s}.jsonl" for s in ("train", "val", "test")]
    for prefix in ("directional_", ""):
        paths = [data_dir / f"{prefix}{s}.jsonl" for s in ("train", "val", "test")]
        if all(p.exists() for p in paths):
            return paths
    raise FileNotFoundError(
        f"Cannot find train/val/test JSONL files in {data_dir}. "
        "Try --split-prefix directional_ or --split-prefix ''"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("train_ce_baselines")
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: CrossEncoderClassifier,
    dl: DataLoader,
    device: torch.device,
    use_bf16: bool,
    loss_fn: nn.Module,
    dataset: CEPairDataset | None = None,
    return_predictions: bool = False,
) -> tuple[dict, list[dict] | None]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    rows: list[dict] | None = [] if return_predictions else None

    ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16 else nullcontext
    )

    for batch in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
        labels = batch["labels"].to(device, non_blocking=True)
        with ctx_factory():
            logits = model(enc)
            loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        lbls = labels.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(lbls)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        if return_predictions and dataset is not None:
            for j, i_ds in enumerate(batch["idx"]):
                r = dataset.rows[i_ds]
                rows.append({
                    "url_1": r.get("url_1", r.get("url", "")),
                    "url_2": r.get("url_2", r.get("url", "")),
                    "sub_1": r.get("sub_1", ""),
                    "sub_2": r.get("sub_2", ""),
                    "tier_1": r.get("tier_1"),
                    "tier_2": r.get("tier_2"),
                    "label": int(lbls[j]),
                    "prob": float(probs[j]),
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
        auc = float("nan")
        f1 = float("nan")
    return (
        {"loss": total_loss / max(total_n, 1), "acc": acc, "auc": auc, "f1": f1, "n": total_n},
        rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a cross-encoder directional classifier (baseline sweep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- Data ----
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-prefix", default=None,
                    help="Filename prefix for splits (e.g. 'directional_' or ''). "
                         "Auto-detected if omitted.")
    # ---- Output ----
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (used directly, no subdirectory created).")
    ap.add_argument("--dataset-slug", default="")
    ap.add_argument("--model-slug", default="")
    # ---- Model ----
    ap.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--gpu-id", type=int, default=None,
                    help="Physical GPU index (informational/logging only). GPU "
                         "isolation is via CUDA_VISIBLE_DEVICES; process uses cuda:0.")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable bf16 autocast and train in fp32. Required for "
                         "DeBERTa-v3 (NaN loss under bf16).")
    ap.add_argument("--dropout", type=float, default=0.1)
    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = device_auto()
    use_bf16 = (device.type == "cuda") and not args.no_amp

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    log.info(f"run_dir={run_dir}")
    log.info(
        f"device={device}  bf16={use_bf16}  physical_gpu={args.gpu_id}  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    log.info(f"dataset={args.dataset_slug}  model={args.model_slug}  backbone={args.base_model}")

    # -- tokenizer --
    log.info(f"tokenizer <- {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # -- locate split files --
    data_dir = Path(args.data_dir)
    jsonls = find_split_files(data_dir, args.split_prefix)
    log.info(f"split files: {[p.name for p in jsonls]}")

    # -- max_seq_len (cap at model's positional embedding limit) --
    model_cap = getattr(tokenizer, "model_max_length", 512)
    if model_cap is None or model_cap > 100_000:
        model_cap = 512
    if args.max_seq_len is None:
        log.info("scanning max JOINT token length across all splits ...")
        t0 = time.time()
        mx = scan_max_seq_len(jsonls, tokenizer)
        args.max_seq_len = min(model_cap, max(128, int(np.ceil(mx * 1.05)) + 2))
        log.info(
            f"  observed_max_joint={mx} -> max_seq_len={args.max_seq_len} "
            f"(cap={model_cap}, scan={time.time() - t0:.1f}s)"
        )
    else:
        args.max_seq_len = min(args.max_seq_len, model_cap)
        log.info(f"max_seq_len={args.max_seq_len} (provided, capped at {model_cap})")

    # -- datasets --
    log.info("loading datasets ...")
    ds_train = CEPairDataset(jsonls[0])
    ds_val = CEPairDataset(jsonls[1])
    ds_test = CEPairDataset(jsonls[2])
    log.info(
        f"  schema={ds_train.schema}  train={len(ds_train)}  "
        f"val={len(ds_val)}  test={len(ds_test)}"
    )

    collate = CECollate(tokenizer, args.max_seq_len)
    pin = device.type == "cuda"
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=pin,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=pin,
    )
    dl_test = DataLoader(
        ds_test, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=pin,
    )

    # -- model --
    log.info(f"building CrossEncoderClassifier  backbone={args.base_model}")
    model = CrossEncoderClassifier(args.base_model, dropout=args.dropout).to(device)
    # Some backbones (e.g. microsoft/deberta-v3-large) ship fp16 weights on the
    # Hub, which mismatches the fp32 Linear head and destabilises training. Force
    # fp32 master weights; bf16 autocast (when enabled) still handles compute.
    model = model.float()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  hidden_size={model.hidden_size}  trainable_params={n_trainable:,}")

    # -- optimizer: two param groups (encoder, head) --
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.classifier.parameters())
    optim = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.0},
    ])
    total_steps = len(dl_train) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    log.info(f"  total_steps={total_steps}  warmup_steps={warmup_steps}")

    loss_fn = nn.BCEWithLogitsLoss()

    # -- config dump --
    cfg = {
        "base_model": args.base_model,
        "dataset_slug": args.dataset_slug,
        "model_slug": args.model_slug,
        "data_dir": str(args.data_dir),
        "schema": ds_train.schema,
        "model_type": "cross_encoder",
        "gpu_id": args.gpu_id,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "warmup_frac": args.warmup_frac,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "dropout": args.dropout,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "device": str(device),
        "bf16": use_bf16,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "pool": "cls",
        "input_form": "[CLS] text_1 [SEP] text_2 [SEP]",
        "head": "Linear(hidden, 1)",
        "loss": "BCEWithLogitsLoss",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # -- training loop --
    ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16 else nullcontext
    )

    best_val_acc = -1.0
    best_val_metrics: dict = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, batch in enumerate(dl_train, 1):
            enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device, non_blocking=True)

            with ctx_factory():
                logits = model(enc)
                loss = loss_fn(logits, labels)

            optim.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            binary_labels = (labels > 0.5).float()
            running_correct += (preds == binary_labels).sum().item()
            running_total += bs

            if step % args.log_every == 0:
                lrs = " ".join(
                    f"lr{i}={pg['lr']:.2e}" for i, pg in enumerate(optim.param_groups)
                )
                log.info(
                    f"  e{epoch} step {step:>5}/{len(dl_train)}  "
                    f"loss={running_loss / running_total:.4f}  "
                    f"acc={running_correct / running_total:.4f}  {lrs}"
                )

        val_metrics, _ = evaluate(model, dl_val, device, use_bf16, loss_fn)
        log.info(
            f"[epoch {epoch}] "
            f"train_loss={running_loss / running_total:.4f} "
            f"train_acc={running_correct / running_total:.4f}  "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}  "
            f"elapsed={time.time() - t_epoch:.1f}s"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_val_metrics = val_metrics
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "cfg": cfg},
                run_dir / "best.pt",
            )
            log.info(f"  saved best.pt  val_acc={val_metrics['acc']:.4f}")

    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
        run_dir / "final.pt",
    )

    # -- test eval with best checkpoint --
    log.info("loading best.pt for test eval")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, preds_rows = evaluate(
        model, dl_test, device, use_bf16, loss_fn,
        dataset=ds_test, return_predictions=True,
    )
    log.info(
        f"[test] loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
        f"auc={test_metrics['auc']:.4f}  f1={test_metrics['f1']:.4f}"
    )

    # per-tier-pair accuracy (only meaningful for standard-schema datasets)
    pair_stats: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0}
    )
    for row in preds_rows:
        t1 = row["tier_1"] or 0
        t2 = row["tier_2"] or 0
        key = (min(t1, t2), max(t1, t2))
        pair_stats[key]["n"] += 1
        pair_stats[key]["correct"] += int((row["prob"] > 0.5) == (row["label"] == 1))
    per_pair = {
        f"T{a}-T{b}": {"n": s["n"], "acc": s["correct"] / s["n"]}
        for (a, b), s in sorted(pair_stats.items())
    }

    (run_dir / "test_metrics.json").write_text(
        json.dumps({**test_metrics, "per_tier_pair": per_pair}, indent=2)
    )
    with (run_dir / "test_predictions.jsonl").open("w") as f:
        for row in preds_rows:
            f.write(json.dumps(row) + "\n")

    # -- run_summary.json for aggregation --
    summary = {
        "dataset_slug": args.dataset_slug,
        "model_slug": args.model_slug,
        "base_model": args.base_model,
        "schema": ds_train.schema,
        "best_val_acc": round(best_val_metrics.get("acc", float("nan")), 6),
        "best_val_auc": round(best_val_metrics.get("auc", float("nan")), 6),
        "test_acc": round(test_metrics["acc"], 6),
        "test_auc": round(test_metrics["auc"], 6),
        "test_f1": round(test_metrics["f1"], 6),
        "test_loss": round(test_metrics["loss"], 6),
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"[done] run_summary.json written to {run_dir}")


if __name__ == "__main__":
    main()
