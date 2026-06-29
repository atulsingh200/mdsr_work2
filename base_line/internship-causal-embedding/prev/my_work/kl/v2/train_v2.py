"""Train CDEv2 on aep_causal.

Three-phase training schedule (5000 steps total):
  Phase A  (0 – 1000)   : InfoNCE only — let dual distributions stabilise
  Phase B  (1000 – 3000): + entropy ordering + antisymmetry + sigma lower-bound
  Phase C  (3000 – 5000): + BPR ranking loss (directly optimises AUC)

Key improvements over train_cde.py (v1):
  - Dual-encoder CDEv2 model (cause_enc + effect_enc)
  - batch_size = 64  (2x in-batch negatives → stronger InfoNCE)
  - K = 8 hard negatives (from hard_negatives_k8.npy)
  - Normalised hybrid KL+cosine scoring (fixes AUC calibration)
  - Phase C BPR loss for direct AUC optimisation
  - Sigma lower-bound regularisation (prevents variance collapse)

Pre-requisites:
  python ../prepare_data.py          # creates train/val/test_pairs.jsonl
  python mine_hard_neg_v2.py         # creates hard_negatives_k8.npy (K=8)

Usage:
  python train_v2.py
  python train_v2.py --batch-size 32 --total-steps 5000 --bf16
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

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
sys.path.insert(0, str(KL_DIR))

from dataset import TripleDataset, get_tokenizer, load_pairs, make_collate_fn  # noqa: E402
from v2.losses_v2 import CDELossV2                                             # noqa: E402
from v2.model_v2 import CDEv2                                                  # noqa: E402

PROJECT_ROOT = KL_DIR.parent.parent
OUT_DIR = KL_DIR / "results" / "aep_causal"   # default; overridden by --dataset in main()


def _resolve_dataset(name):
    """Set up results/<name>/ with train/val/test pairs + hard_negatives copied
    from finetune_eval/results/<name>/ if not already present. Returns the dir."""
    import shutil
    src = PROJECT_ROOT / "finetune_eval" / "results" / name
    dst = KL_DIR / "results" / name
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("train_pairs.jsonl", "val_pairs.jsonl", "test_pairs.jsonl",
               "hard_negatives.npy", "hard_negatives_k8.npy"):
        s = src / fn
        if s.exists() and not (dst / fn).exists():
            shutil.copy2(s, dst / fn)
    return dst


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(
    model: CDEv2,
    loader: DataLoader,
    device: torch.device,
    k_values: tuple = (1, 3, 5, 10),
) -> dict:
    """Full N×N retrieval on the val set using hybrid scores."""
    model.eval()
    all_mu_ta, all_log_sigma_ta = [], []
    all_mu_b, all_log_sigma_b = [], []

    for a_ids, a_mask, p_ids, p_mask, *_ in loader:
        a_ids = a_ids.to(device); a_mask = a_mask.to(device)
        p_ids = p_ids.to(device); p_mask = p_mask.to(device)
        mu_a, log_sigma_a, delta_a = model.encode_cause(a_ids, a_mask)
        mu_ta, log_sigma_ta = model.transport(mu_a, log_sigma_a, delta_a)
        mu_b, log_sigma_b = model.encode_effect(p_ids, p_mask)
        all_mu_ta.append(mu_ta.cpu());   all_log_sigma_ta.append(log_sigma_ta.cpu())
        all_mu_b.append(mu_b.cpu());     all_log_sigma_b.append(log_sigma_b.cpu())

    mu_ta       = torch.cat(all_mu_ta).to(device)
    log_sigma_ta = torch.cat(all_log_sigma_ta).to(device)
    mu_b        = torch.cat(all_mu_b).to(device)
    log_sigma_b = torch.cat(all_log_sigma_b).to(device)

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
    model: CDEv2,
    loss_fn: CDELossV2,
    batch: tuple,
    K: int,
    device: torch.device,
    phase: str,
) -> tuple[torch.Tensor, dict]:
    a_ids, a_mask, p_ids, p_mask, neg_ids, neg_mask = batch
    B = a_ids.size(0)
    include_reverse = (phase in ("B", "C"))

    feats = model.forward_features(a_ids, a_mask, p_ids, p_mask,
                                   include_reverse=include_reverse)
    mu_ta, log_sigma_ta = feats["mu_ta"], feats["log_sigma_ta"]
    mu_b,  log_sigma_b  = feats["mu_b"],  feats["log_sigma_b"]

    # In-batch score matrix (B, B)
    score_mat = model.score_matrix(mu_ta, log_sigma_ta, mu_b, log_sigma_b)
    score_pos = score_mat.diagonal()  # (B,)

    # Hard-negative scores (B, K)
    score_hn = None
    if K > 0 and neg_ids.numel() > 0:
        neg_ids  = neg_ids.to(device,  non_blocking=True)
        neg_mask = neg_mask.to(device, non_blocking=True)
        mu_n, log_sigma_n = model.encode_effect(
            neg_ids.view(B * K, -1), neg_mask.view(B * K, -1)
        )
        mu_n        = mu_n.view(B, K, -1)
        log_sigma_n = log_sigma_n.view(B, K, -1)
        mu_ta_exp   = mu_ta.unsqueeze(1).expand(-1, K, -1)
        lsig_ta_exp = log_sigma_ta.unsqueeze(1).expand(-1, K, -1)
        score_hn = model.score_pointwise(
            mu_ta_exp.reshape(B * K, -1),
            lsig_ta_exp.reshape(B * K, -1),
            mu_n.reshape(B * K, -1),
            log_sigma_n.reshape(B * K, -1),
        ).view(B, K)

    # Reverse scores for antisymmetry (phases B, C)
    score_ba = None
    if include_reverse and "mu_tb" in feats:
        mu_tb, log_sigma_tb = feats["mu_tb"], feats["log_sigma_tb"]
        mu_ae, log_sigma_ae = feats["mu_ae"], feats["log_sigma_ae"]
        score_ba = model.score_pointwise(mu_tb, log_sigma_tb, mu_ae, log_sigma_ae)

    loss, stats = loss_fn(
        score_matrix=score_mat,
        score_hardneg=score_hn,
        log_sigma_a=feats["log_sigma_a"],
        log_sigma_b=log_sigma_b,
        score_ab=score_pos,
        score_ba=score_ba,
    )

    with torch.no_grad():
        stats["score_pos_mean"]    = score_pos.mean().item()
        off = ~torch.eye(B, dtype=torch.bool, device=device)
        stats["score_ib_neg_mean"] = score_mat[off].mean().item()
        if score_hn is not None:
            stats["score_hn_mean"] = score_hn.mean().item()
        stats["delta_norm"] = feats["delta_a"].norm(dim=-1).mean().item()
        stats["gamma"]      = model.gamma.item()
        stats["cos_weight"] = model.cos_weight.item()
        stats["sigma_a_mean"] = feats["log_sigma_a"].exp().mean().item()
        stats["sigma_b_mean"] = log_sigma_b.exp().mean().item()

    return loss, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",           default="aep_causal",
                    help="Dataset name under finetune_eval/results/. Sets OUT_DIR=results/<dataset>/.")
    ap.add_argument("--backbone",          default="google-bert/bert-base-uncased")
    ap.add_argument("--proj-dim",          type=int,   default=256)
    ap.add_argument("--batch-size",        type=int,   default=64)
    ap.add_argument("--shared-encoder",    action="store_true",
                    help="Share one BERT backbone between cause and effect "
                         "(asymmetry via heads only). Better on small data.")
    ap.add_argument("--kl-scale",          default="dim", choices=["dim", "none"],
                    help="'dim' divides KL by proj_dim (AUC); 'none' raw KL (MRR).")
    ap.add_argument("--init-cos-weight",   type=float, default=1.0,
                    help="Initial cosine-term weight in hybrid score; 0 disables.")
    ap.add_argument("--pooling",           default="mean", choices=["mean", "cls"],
                    help="Token pooling; use 'cls' to match a BiEncoder warm-start.")
    ap.add_argument("--mu-identity",       action="store_true",
                    help="mu = pooled embedding directly (proj_dim must == hidden).")
    ap.add_argument("--log-sigma-init",    type=float, default=0.0,
                    help="Constant init for log_sigma (large => KL~0 at warm-start).")
    ap.add_argument("--init-from-biencoder", default=None,
                    help="Path to a BiEncoder checkpoint to warm-start encoders.")
    ap.add_argument("--score-type", default="kl", choices=["kl", "mmd"],
                    help="'kl' = -KL density score (v1); 'mmd' = directed expected-RBF "
                         "kernel score with a learned directional sigmoid gate.")
    ap.add_argument("--phase-a-steps",     type=int,   default=1000,
                    help="InfoNCE-only warmup steps.")
    ap.add_argument("--phase-b-steps",     type=int,   default=2000,
                    help="Steps in phase B (entropy + antisym + sigma_lb).")
    ap.add_argument("--total-steps",       type=int,   default=5000)
    ap.add_argument("--backbone-lr",       type=float, default=2e-5)
    ap.add_argument("--head-lr",           type=float, default=1e-4)
    ap.add_argument("--temperature-lr",    type=float, default=1e-3)
    ap.add_argument("--temperature-init",  type=float, default=0.05)
    ap.add_argument("--weight-decay",      type=float, default=0.01)
    ap.add_argument("--warmup-ratio",      type=float, default=0.05)
    ap.add_argument("--grad-clip",         type=float, default=1.0)
    ap.add_argument("--max-seq-length",    type=int,   default=256)
    ap.add_argument("--lambda-entropy",    type=float, default=0.1)
    ap.add_argument("--lambda-antisym",    type=float, default=0.5)
    ap.add_argument("--lambda-bpr",        type=float, default=0.3,
                    help="Weight for BPR AUC-proxy loss in Phase C.")
    ap.add_argument("--lambda-sigma-lb",   type=float, default=0.05,
                    help="Sigma lower-bound regularisation (all phases).")
    ap.add_argument("--sigma-lb",          type=float, default=-3.0,
                    help="log_sigma lower bound (default: -3 → sigma ≥ 0.05).")
    ap.add_argument("--entropy-margin",    type=float, default=0.5)
    ap.add_argument("--antisym-margin",    type=float, default=2.0)
    ap.add_argument("--val-every",         type=int,   default=200)
    ap.add_argument("--early-stopping-patience", type=int, default=6)
    ap.add_argument("--num-workers",       type=int,   default=4)
    ap.add_argument("--bf16",              action="store_true", default=True)
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--hard-neg-file",     default=None,
                    help="Path to hard_negatives npy file. "
                         "Defaults to results/aep_causal/hard_negatives_k8.npy "
                         "falling back to hard_negatives.npy (K=4).")
    ap.add_argument("--out-suffix",        default="",
                    help="Suffix for output files (e.g. '_run3a') so parallel "
                         "runs don't clobber each other's checkpoints.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    global OUT_DIR
    OUT_DIR = _resolve_dataset(args.dataset)
    device = auto_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[out]    {OUT_DIR}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_pairs_path = OUT_DIR / "train_pairs.jsonl"
    if not train_pairs_path.exists():
        raise SystemExit("Run prepare_data.py first.")

    # Prefer K=8 hard negatives, fall back to K=4.
    if args.hard_neg_file:
        hard_neg_path = Path(args.hard_neg_file)
    else:
        hard_neg_path = OUT_DIR / "hard_negatives_k8.npy"
        if not hard_neg_path.exists():
            hard_neg_path = OUT_DIR / "hard_negatives.npy"
            print(f"[data] K=8 file not found; using {hard_neg_path.name}")

    if not hard_neg_path.exists():
        raise SystemExit(
            "No hard negatives found. Run:\n"
            "  python v2/mine_hard_neg_v2.py   (recommended: K=8)\n"
            "  python mine_hard_negatives.py   (fallback: K=4)"
        )

    train_pairs = load_pairs(train_pairs_path)
    hard_negs   = np.load(hard_neg_path)
    print(f"[data] {len(train_pairs)} train pairs  K={hard_negs.shape[1]}")

    val_pairs_path = OUT_DIR / "val_pairs.jsonl"
    if val_pairs_path.exists():
        val_pairs = load_pairs(val_pairs_path)
        if len(val_pairs) > 2000:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(val_pairs), size=2000, replace=False)
            val_pairs = [val_pairs[i] for i in sorted(idx)]
        print(f"[data] {len(val_pairs)} val pairs")
    else:
        rng = random.Random(args.seed)
        idx_all = list(range(len(train_pairs)))
        rng.shuffle(idx_all)
        n_val = max(1, int(len(train_pairs) * 0.10))
        val_pairs = [train_pairs[i] for i in sorted(idx_all[:n_val])]
        train_pairs = [train_pairs[i] for i in sorted(idx_all[n_val:])]
        print(f"[data] {len(val_pairs)} val pairs (held-out)")

    train_ds = TripleDataset(train_pairs, hard_negs)
    val_ds   = TripleDataset(val_pairs, hard_negatives=None)

    # ── Model ─────────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(args.backbone)
    model = CDEv2(
        args.backbone, args.proj_dim,
        shared_encoder=args.shared_encoder,
        kl_scale=args.kl_scale,
        init_cos_weight=args.init_cos_weight,
        pooling=args.pooling,
        mu_identity=args.mu_identity,
        log_sigma_init=args.log_sigma_init,
        score_type=args.score_type,
    )
    if args.init_from_biencoder:
        print(f"[warm-start] from {args.init_from_biencoder}")
        model.load_from_biencoder(args.init_from_biencoder)
    model = model.to(device)
    print(f"[model] shared_encoder={args.shared_encoder}  kl_scale={args.kl_scale}  "
          f"init_cos_weight={args.init_cos_weight}  pooling={args.pooling}  "
          f"mu_identity={args.mu_identity}  log_sigma_init={args.log_sigma_init}")
    # Enable DataParallel when multiple GPUs are available.
    if torch.cuda.device_count() > 1:
        print(f"[model] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CDEv2  backbone={args.backbone}  proj_dim={args.proj_dim}  params={n_params:,}")

    collate = make_collate_fn(tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    loss_fn = CDELossV2(
        temperature_init=args.temperature_init,
        entropy_margin=args.entropy_margin,
        antisym_margin=args.antisym_margin,
        sigma_lb=args.sigma_lb,
        lambda_entropy=0.0,
        lambda_antisym=0.0,
        lambda_bpr=0.0,
        lambda_sigma_lb=0.0,
    ).to(device)

    core = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Use id-dedup so a shared encoder isn't counted twice.
    seen: set[int] = set()
    backbone_params = []
    for enc in (core.cause_enc, core.effect_enc):
        for p in enc.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                backbone_params.append(p)
    head_params = (
        list(core.cause_mu.parameters())
        + list(core.cause_log_sigma.parameters())
        + list(core.cause_delta.parameters())
        + list(core.effect_mu.parameters())
        + list(core.effect_log_sigma.parameters())
        + [core.gamma]
    )
    if core.use_cos:
        head_params.append(core.log_cos_weight)
    if core.score_type == "mmd":
        head_params += [core.log_gamma_k, core.gate_w, core.gate_b]
    # Dedup heads too (shared encoder shares no heads, but be safe).
    head_params = [p for p in head_params if id(p) not in seen and not seen.add(id(p))]

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

    use_bf16 = args.bf16 and device.type == "cuda"
    amp_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16
        else (lambda: nullcontext())
    )

    # ── Config snapshot ───────────────────────────────────────────────────────
    phase_b_end = args.phase_a_steps + args.phase_b_steps
    cfg = vars(args).copy()
    cfg.update({
        "model": "CDEv2",
        "phase_b_end": phase_b_end,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "k_hard_negs": train_ds.n_hard_negatives,
        "device": str(device),
        "bf16": use_bf16,
        "n_gpus": torch.cuda.device_count(),
    })
    sfx = args.out_suffix
    (OUT_DIR / f"cde_v2{sfx}_config.json").write_text(json.dumps(cfg, indent=2))

    # ── Training loop ─────────────────────────────────────────────────────────
    K = train_ds.n_hard_negatives
    best_mrr = -1.0
    patience = 0
    history: list[dict] = []
    best_path   = OUT_DIR / f"cde_v2{sfx}_best.pt"
    latest_path = OUT_DIR / f"cde_v2{sfx}_latest.pt"

    def run_val(step: int) -> dict:
        nonlocal best_mrr, patience
        m_core = model.module if isinstance(model, torch.nn.DataParallel) else model
        metrics = validate(m_core, val_loader, device)
        improved = metrics["mrr"] > best_mrr + 1e-4
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
        if improved:
            patience = 0
            torch.save({
                "step": step,
                "model_state_dict": m_core.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "metrics": metrics,
                "cfg": cfg,
            }, best_path)
            flag = "  * new best"
        else:
            patience += 1
            flag = f"  (patience {patience}/{args.early_stopping_patience})"
        torch.save({
            "step": step,
            "model_state_dict": m_core.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "cfg": cfg,
        }, latest_path)
        print(
            f"  [val step={step}]  MRR={metrics['mrr']:.4f}  "
            f"R@1={metrics['recall@1']:.3f}  R@5={metrics['recall@5']:.3f}  "
            f"R@10={metrics['recall@10']:.3f}  med={metrics['median_rank']:.0f}"
            + flag
        )
        history.append({"step": step, "val_metrics": metrics})
        return metrics

    print()
    print("=" * 80)
    print(f"CDEv2 training  total={args.total_steps}  phA={args.phase_a_steps}  "
          f"phB_end={phase_b_end}")
    print(f"batch={args.batch_size}  K={K}  bf16={use_bf16}  "
          f"backbone_lr={args.backbone_lr}  head_lr={args.head_lr}")
    print("=" * 80)

    global_step = 0
    stop = False
    prev_phase = "A"
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
            loss_fn.lambda_entropy  = 0.0
            loss_fn.lambda_antisym  = 0.0
            loss_fn.lambda_bpr      = 0.0
            loss_fn.lambda_sigma_lb = 0.0
        elif global_step < phase_b_end:
            phase = "B"
            loss_fn.lambda_entropy  = args.lambda_entropy
            loss_fn.lambda_antisym  = args.lambda_antisym
            loss_fn.lambda_bpr      = 0.0
            loss_fn.lambda_sigma_lb = args.lambda_sigma_lb
        else:
            phase = "C"
            loss_fn.lambda_entropy  = args.lambda_entropy
            loss_fn.lambda_antisym  = args.lambda_antisym
            loss_fn.lambda_bpr      = args.lambda_bpr
            loss_fn.lambda_sigma_lb = args.lambda_sigma_lb

        # Reset early-stopping patience at each phase transition so a new loss
        # term (e.g. BPR in Phase C) gets a fresh budget to improve val MRR.
        if phase != prev_phase:
            print(f"  [phase {prev_phase} -> {phase}] resetting patience")
            patience = 0
            prev_phase = phase

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
                + (f"  bpr={stats.get('loss_bpr', 0):.4f}" if "loss_bpr" in stats else "")
                + (f"  slb={stats.get('loss_sigma_lb', 0):.4f}" if "loss_sigma_lb" in stats else "")
                + f"  τ={stats['temperature']:.4f}"
                f"  λcos={stats['cos_weight']:.3f}"
                f"  s+={stats['score_pos_mean']:.3f}"
                f"  s-={stats['score_ib_neg_mean']:.3f}"
                f"  γ={stats['gamma']:.4f}"
            )

        if global_step % args.val_every == 0 or global_step == args.total_steps:
            run_val(global_step)
            if patience >= args.early_stopping_patience:
                print(f"\n[early stop] no MRR improvement for {patience} checks.")
                stop = True

    (OUT_DIR / f"cde_v2{sfx}_history.json").write_text(json.dumps(history, indent=2))
    print()
    print("=" * 80)
    print(f"[done]  best val MRR={best_mrr:.4f}")
    print(f"  best:   {best_path}")
    print(f"  latest: {latest_path}")


if __name__ == "__main__":
    main()
