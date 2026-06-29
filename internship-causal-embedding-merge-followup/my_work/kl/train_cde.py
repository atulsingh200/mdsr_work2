"""Train CausalDensityEmbedding (CDE) on aep_causal.

Two-phase schedule (scaled for the 5 487-pair dataset):
  Phase A  (steps 0 – PHASE_A_STEPS-1)  : InfoNCE only — let the
            Gaussian distributions and drift stabilise.
  Phase B  (steps PHASE_A_STEPS – total) : + entropy ordering
            + antisymmetry penalty.

Hard negatives must be pre-mined:
  python prepare_data.py
  python mine_hard_negatives.py

Usage:
  python train_cde.py
  python train_cde.py --batch-size 16 --bf16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

KL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KL_DIR))

from dataset import TripleDataset, get_tokenizer, load_pairs, make_collate_fn  # noqa: E402
from losses import CDELoss                                                      # noqa: E402
from model import CausalDensityEmbedding                                       # noqa: E402

OUT_DIR = KL_DIR / "results" / "aep_causal"


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(
    model: CausalDensityEmbedding,
    loader: DataLoader,
    device: torch.device,
    k_values=(1, 3, 5, 10),
) -> dict:
    """Full N×N score-matrix retrieval on the val set."""
    model.eval()
    all_mu_ta, all_log_sigma_ta = [], []
    all_mu_b, all_log_sigma_b = [], []

    for a_ids, a_mask, p_ids, p_mask, *_ in loader:
        a_ids  = a_ids.to(device);  a_mask  = a_mask.to(device)
        p_ids  = p_ids.to(device);  p_mask  = p_mask.to(device)
        mu_a, log_sigma_a, delta_a = model.encode(a_ids, a_mask)
        mu_ta, log_sigma_ta = model.transport(mu_a, log_sigma_a, delta_a)
        mu_b, log_sigma_b, _ = model.encode(p_ids, p_mask)
        all_mu_ta.append(mu_ta.cpu())
        all_log_sigma_ta.append(log_sigma_ta.cpu())
        all_mu_b.append(mu_b.cpu())
        all_log_sigma_b.append(log_sigma_b.cpu())

    mu_ta = torch.cat(all_mu_ta)
    log_sigma_ta = torch.cat(all_log_sigma_ta)
    mu_b = torch.cat(all_mu_b)
    log_sigma_b = torch.cat(all_log_sigma_b)

    # Move to device for the full matrix computation.
    mu_ta = mu_ta.to(device);  log_sigma_ta = log_sigma_ta.to(device)
    mu_b  = mu_b.to(device);   log_sigma_b  = log_sigma_b.to(device)

    # Score matrix (N, N) — row i is scores of A_i against all B_j.
    S = model.score_matrix(mu_ta, log_sigma_ta, mu_b, log_sigma_b)
    N = S.size(0)
    diag = S.diagonal().unsqueeze(1)
    ranks = (S > diag).sum(dim=1).float() + 1.0

    out = {
        "n_val":       N,
        "mrr":         float((1.0 / ranks).mean()),
        "mean_rank":   float(ranks.mean()),
        "median_rank": float(ranks.median()),
    }
    for k in k_values:
        out[f"recall@{k}"] = float((ranks <= k).float().mean())
    return out


def train_step(
    model: CausalDensityEmbedding,
    loss_fn: CDELoss,
    batch: tuple,
    K: int,
    device: torch.device,
    phase: str,
) -> tuple[torch.Tensor, dict]:
    a_ids, a_mask, p_ids, p_mask, neg_ids, neg_mask = batch
    B = a_ids.size(0)

    feats = model.forward_features(a_ids, a_mask, p_ids, p_mask)
    mu_ta, log_sigma_ta = feats["mu_ta"], feats["log_sigma_ta"]
    mu_b,  log_sigma_b  = feats["mu_b"],  feats["log_sigma_b"]

    # In-batch score matrix (B, B)
    score_mat = model.score_matrix(mu_ta, log_sigma_ta, mu_b, log_sigma_b)
    score_pos_diag = score_mat.diagonal()  # (B,)

    # Hard-negative scores (B, K)
    score_hn = None
    if K > 0 and neg_ids.numel() > 0:
        neg_ids  = neg_ids.to(device,  non_blocking=True)
        neg_mask = neg_mask.to(device, non_blocking=True)
        mu_n, log_sigma_n, _ = model.encode(
            neg_ids.view(B * K, -1), neg_mask.view(B * K, -1)
        )
        mu_n        = mu_n.view(B, K, -1)
        log_sigma_n = log_sigma_n.view(B, K, -1)
        mu_ta_rep        = mu_ta.unsqueeze(1).expand(-1, K, -1)
        log_sigma_ta_rep = log_sigma_ta.unsqueeze(1).expand(-1, K, -1)
        # KL(N_neg || T(N_A)) pointwise over (B*K,)
        kl_hn = model.kl_pointwise(
            mu_n.reshape(B * K, -1),
            log_sigma_n.reshape(B * K, -1),
            mu_ta_rep.reshape(B * K, -1),
            log_sigma_ta_rep.reshape(B * K, -1),
        )
        score_hn = -kl_hn.view(B, K)

    # Antisymmetry: score(B -> A)
    score_ba = None
    if phase == "B":
        mu_tb, log_sigma_tb = feats["mu_tb"], feats["log_sigma_tb"]
        mu_a,  log_sigma_a  = feats["mu_a"],  feats["log_sigma_a"]
        kl_ba = model.kl_pointwise(mu_a, log_sigma_a, mu_tb, log_sigma_tb)
        score_ba = -kl_ba

    loss, stats = loss_fn(
        score_matrix=score_mat,
        score_hardneg=score_hn,
        log_sigma_a=feats["log_sigma_a"],
        log_sigma_b=log_sigma_b,
        score_ab=score_pos_diag,
        score_ba=score_ba,
    )

    # Diagnostics.
    with torch.no_grad():
        stats["score_pos_mean"]  = score_pos_diag.mean().item()
        off = ~torch.eye(B, dtype=torch.bool, device=device)
        stats["score_ib_neg_mean"] = score_mat[off].mean().item()
        if score_hn is not None:
            stats["score_hn_mean"] = score_hn.mean().item()
        stats["delta_norm"] = feats["delta_a"].norm(dim=-1).mean().item()
        stats["gamma"]      = model.gamma.item()
        stats["sigma_mean"] = feats["log_sigma_a"].exp().mean().item()

    return loss, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone",         default="google-bert/bert-base-uncased")
    ap.add_argument("--proj-dim",         type=int,   default=128)
    ap.add_argument("--batch-size",       type=int,   default=32)
    ap.add_argument("--phase-a-steps",    type=int,   default=500,
                    help="InfoNCE-only warmup steps.")
    ap.add_argument("--total-steps",      type=int,   default=2000)
    ap.add_argument("--backbone-lr",      type=float, default=2e-5)
    ap.add_argument("--head-lr",          type=float, default=1e-4)
    ap.add_argument("--temperature-lr",   type=float, default=1e-3)
    ap.add_argument("--temperature-init", type=float, default=0.05)
    ap.add_argument("--weight-decay",     type=float, default=0.01)
    ap.add_argument("--warmup-ratio",     type=float, default=0.05)
    ap.add_argument("--grad-clip",        type=float, default=1.0)
    ap.add_argument("--max-seq-length",   type=int,   default=256)
    ap.add_argument("--lambda-entropy",   type=float, default=0.1)
    ap.add_argument("--lambda-antisym",   type=float, default=0.5)
    ap.add_argument("--entropy-margin",   type=float, default=0.5)
    ap.add_argument("--antisym-margin",   type=float, default=2.0)
    ap.add_argument("--val-every",        type=int,   default=200,
                    help="Validate every N steps.")
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--num-workers",      type=int,   default=2)
    ap.add_argument("--bf16",             action="store_true")
    ap.add_argument("--seed",             type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = auto_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[out]    {OUT_DIR}")

    # ── data ──────────────────────────────────────────────────────────────
    train_pairs_path = OUT_DIR / "train_pairs.jsonl"
    hard_neg_path    = OUT_DIR / "hard_negatives.npy"
    if not train_pairs_path.exists() or not hard_neg_path.exists():
        raise SystemExit(
            "Missing prepared data. Run:\n"
            "  python prepare_data.py\n"
            "  python mine_hard_negatives.py"
        )
    train_pairs = load_pairs(train_pairs_path)
    hard_negs   = np.load(hard_neg_path)
    print(f"[data] {len(train_pairs)} train pairs  K={hard_negs.shape[1]}")

    val_pairs_path = OUT_DIR / "val_pairs.jsonl"
    if val_pairs_path.exists():
        val_pairs = load_pairs(val_pairs_path)
        # Cap val for speed (full N×N matrix on GPU).
        if len(val_pairs) > 2000:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(val_pairs), size=2000, replace=False)
            idx.sort()
            val_pairs = [val_pairs[i] for i in idx]
        print(f"[data] {len(val_pairs)} val pairs")
    else:
        # Hold out 10 % of train as validation.
        rng = random.Random(args.seed)
        idx_all = list(range(len(train_pairs)))
        rng.shuffle(idx_all)
        n_val = max(1, int(len(train_pairs) * 0.10))
        val_pairs   = [train_pairs[i] for i in sorted(idx_all[:n_val])]
        train_activ = sorted(idx_all[n_val:])
        print(f"[data] {len(val_pairs)} val pairs (held-out from train)")

    train_active_indices = locals().get("train_activ", None)
    train_ds = TripleDataset(train_pairs, hard_negs,
                             active_indices=train_active_indices)
    val_ds   = TripleDataset(val_pairs, hard_negatives=None)

    # ── model ─────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(args.backbone)
    model = CausalDensityEmbedding(args.backbone, args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CDE  backbone={args.backbone}  proj_dim={args.proj_dim}  params={n_params:,}")

    collate = make_collate_fn(tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # ── loss + optimiser ──────────────────────────────────────────────────
    loss_fn = CDELoss(
        temperature_init=args.temperature_init,
        entropy_margin=args.entropy_margin,
        antisym_margin=args.antisym_margin,
        lambda_entropy=0.0,  # set in phase B
        lambda_antisym=0.0,
    ).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.mu_head.parameters())
        + list(model.log_sigma_head.parameters())
        + list(model.delta_head.parameters())
        + [model.gamma]
    )
    optimizer = AdamW([
        {"params": backbone_params,           "lr": args.backbone_lr,    "weight_decay": args.weight_decay},
        {"params": head_params,               "lr": args.head_lr,        "weight_decay": args.weight_decay},
        {"params": [loss_fn.log_temperature], "lr": args.temperature_lr, "weight_decay": 0.0},
    ])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.backbone_lr, args.head_lr, args.temperature_lr],
        total_steps=args.total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

    amp_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (args.bf16 and device.type == "cuda")
        else (lambda: nullcontext())
    )

    # ── config snapshot ───────────────────────────────────────────────────
    cfg = vars(args)
    cfg.update({"n_train": len(train_ds), "n_val": len(val_ds),
                "k_hard_negs": train_ds.n_hard_negatives, "device": str(device)})
    (OUT_DIR / "cde_config.json").write_text(json.dumps(cfg, indent=2))

    # ── training loop ─────────────────────────────────────────────────────
    K = train_ds.n_hard_negatives
    best_mrr = -1.0
    patience = 0
    history: list[dict] = []
    best_path   = OUT_DIR / "cde_best.pt"
    latest_path = OUT_DIR / "cde_latest.pt"

    def run_val(global_step: int) -> dict:
        nonlocal best_mrr, patience
        metrics = validate(model, val_loader, device)
        improved = metrics["mrr"] > best_mrr + 1e-4
        new_best = metrics["mrr"] > best_mrr
        if new_best:
            best_mrr = metrics["mrr"]
        if improved:
            patience = 0
            torch.save({"step": global_step, "model_state_dict": model.state_dict(),
                        "loss_fn_state_dict": loss_fn.state_dict(),
                        "metrics": metrics, "cfg": cfg}, best_path)
            flag = "  * new best"
        else:
            patience += 1
            flag = f"  (patience {patience}/{args.early_stopping_patience})"
        torch.save({"step": global_step, "model_state_dict": model.state_dict(),
                    "loss_fn_state_dict": loss_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics, "cfg": cfg}, latest_path)
        print(
            f"  [val step={global_step}]  MRR={metrics['mrr']:.4f}  "
            f"R@1={metrics['recall@1']:.3f}  R@5={metrics['recall@5']:.3f}  "
            f"R@10={metrics['recall@10']:.3f}  med_rank={metrics['median_rank']:.0f}"
            + flag
        )
        history.append({"step": global_step, "val_metrics": metrics})
        return metrics

    print()
    print("=" * 78)
    print(f"CDE training  total_steps={args.total_steps}  phase_A={args.phase_a_steps}")
    print(f"batch={args.batch_size}  K={K}  backbone_lr={args.backbone_lr}  head_lr={args.head_lr}")
    print("=" * 78)

    global_step = 0
    stop = False
    data_iter = iter(train_loader)

    while global_step < args.total_steps and not stop:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Phase gating.
        if global_step < args.phase_a_steps:
            phase = "A"
            loss_fn.lambda_entropy = 0.0
            loss_fn.lambda_antisym = 0.0
        else:
            phase = "B"
            loss_fn.lambda_entropy = args.lambda_entropy
            loss_fn.lambda_antisym = args.lambda_antisym

        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        model.train()
        optimizer.zero_grad()

        with amp_ctx():
            loss, stats = train_step(model, loss_fn, batch, K, device, phase)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % 50 == 0:
            print(
                f"  step={global_step:>5}/{args.total_steps} [{phase}]"
                f"  loss={stats['loss_total']:.4f}"
                f"  info={stats['loss_infonce']:.4f}"
                + (f"  ent={stats.get('loss_entropy', 0):.4f}" if "loss_entropy" in stats else "")
                + (f"  anti={stats.get('loss_antisym', 0):.4f}" if "loss_antisym" in stats else "")
                + f"  τ={stats['temperature']:.4f}"
                f"  s+={stats['score_pos_mean']:.2f}"
                f"  s-={stats['score_ib_neg_mean']:.2f}"
                f"  γ={stats['gamma']:.4f}"
                f"  δ={stats['delta_norm']:.3f}"
            )

        if global_step % args.val_every == 0 or global_step == args.total_steps:
            run_val(global_step)
            if patience >= args.early_stopping_patience:
                print(f"\n[early stop] no MRR improvement for {patience} checks.")
                stop = True

    (OUT_DIR / "cde_history.json").write_text(json.dumps(history, indent=2))
    print()
    print("=" * 78)
    print(f"[done]  best val MRR={best_mrr:.4f}")
    print(f"  best checkpoint:  {best_path}")
    print(f"  latest:           {latest_path}")


if __name__ == "__main__":
    main()
