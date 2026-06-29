"""Stage 3 training: InfoNCE (space) + optional Lorentzian time-ordering.

Modes:
  - Vanilla baseline:  bidirectional=False, lambda_ord=0, beta=0, max_negatives=4
                       → plain forward InfoNCE bi-encoder (matches BERT baseline recipe).
  - Full Lorentz:      bidirectional=True, lambda_ord>0, beta>0
                       → space InfoNCE + time-ordering loss + combined space+time scoring.

Time head wiring:
  - Negatives are pre-encoded once per epoch (both space AND time embeddings),
    avoiding the B×K backward-graph memory spike.
  - Ordering loss pushes t_pos > t_anchor (positive is "after") and t_neg < t_anchor.
  - Optional beta term adds the SteepSigmoid time score into the InfoNCE logits.
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ..model.encoder import LorentzEncoder
from ..data.hard_neg_miner import load_hard_negatives


def tokenize_texts(tokenizer, texts, prefix, max_length):
    if prefix:
        texts = [prefix + t for t in texts]
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def steep_sigmoid_mean(t_a, t_b, tau):
    """Asymmetric precedence score: high when t_b is 'after' t_a in every dim."""
    return torch.sigmoid(tau * (t_b - t_a)).mean(dim=-1)


# ------------------------------------------------------------------ #
#  Dataset serving pre-encoded negative space + time embeddings
# ------------------------------------------------------------------ #
class NegEmbDataset(Dataset):
    def __init__(self, pairs, hard_idx, neg_space, neg_time):
        self.anchors = [a for a, _ in pairs]
        self.positives = [p for _, p in pairs]
        self.hard_idx = hard_idx        # (N, K)
        self.neg_space = neg_space      # (N, K, D_s)
        self.neg_time = neg_time        # (N, K, D_t)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return (self.anchors[idx], self.positives[idx],
                self.neg_space[idx], self.neg_time[idx],
                (self.hard_idx[idx] >= 0))


def make_collate(tokenizer, max_length, anchor_prefix, positive_prefix, need_reverse=False):
    def collate(batch):
        a = [b[0] for b in batch]
        p = [b[1] for b in batch]
        ns = np.stack([b[2] for b in batch])   # (B, K, D_s)
        nt = np.stack([b[3] for b in batch])   # (B, K, D_t)
        v = np.stack([b[4] for b in batch])    # (B, K) bool
        a_ids, a_mask = tokenize_texts(tokenizer, a, anchor_prefix, max_length)
        p_ids, p_mask = tokenize_texts(tokenizer, p, positive_prefix, max_length)
        out = [a_ids, a_mask, p_ids, p_mask,
               torch.from_numpy(ns).float(), torch.from_numpy(nt).float(),
               torch.from_numpy(v).bool()]
        if need_reverse:
            # a through positive tower, b(=p) through anchor tower (for directional loss)
            ap_ids, ap_mask = tokenize_texts(tokenizer, a, positive_prefix, max_length)
            ba_ids, ba_mask = tokenize_texts(tokenizer, p, anchor_prefix, max_length)
            out += [ap_ids, ap_mask, ba_ids, ba_mask]
        return tuple(out)
    return collate


@torch.no_grad()
def pre_encode_negatives(model, tokenizer, pairs, hard_idx, prefix, max_length, device,
                         batch_size=256):
    """Return (neg_space[N,K,Ds], neg_time[N,K,Dt]) float32 arrays."""
    positives = [p for _, p in pairs]
    N, K = hard_idx.shape
    Ds, Dt = model.space_dim, model.time_dim

    unique = sorted(set(int(j) for j in hard_idx.flatten() if j >= 0))
    smap, tmap = {}, {}
    model.eval()
    for i in range(0, len(unique), batch_size):
        idxs = unique[i:i+batch_size]
        ids, mask = tokenize_texts(tokenizer, [positives[j] for j in idxs], prefix, max_length)
        s, t = model.encode_positive(ids.to(device), mask.to(device))
        s = s.float().cpu().numpy(); t = t.float().cpu().numpy()
        for j, idx in enumerate(idxs):
            smap[idx] = s[j]; tmap[idx] = t[j]
    model.train()

    neg_space = np.zeros((N, K, Ds), dtype=np.float32)
    neg_time = np.zeros((N, K, Dt), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            j = int(hard_idx[i, k])
            if j >= 0 and j in smap:
                neg_space[i, k] = smap[j]
                neg_time[i, k] = tmap[j]
    return neg_space, neg_time


# ------------------------------------------------------------------ #
#  Losses
# ------------------------------------------------------------------ #
def info_nce(s_a, s_p, neg_space, neg_valid, tau, margin,
             t_a=None, t_p=None, neg_time=None, beta=0.0, tau_score=10.0,
             bidirectional=True):
    """InfoNCE on space cosine, optionally augmented with the time score (beta>0)."""
    B = s_a.size(0)
    device = s_a.device

    pos = (s_a * s_p).sum(-1)
    hn = (s_a.unsqueeze(1) * neg_space).sum(-1)               # (B, K)
    ib = s_a @ s_p.t()                                         # (B, B)

    if beta > 0 and t_a is not None:
        pos = pos + beta * steep_sigmoid_mean(t_a, t_p, tau_score)
        hn = hn + beta * torch.sigmoid(tau_score * (neg_time - t_a.unsqueeze(1))).mean(-1)
        diff = t_p.unsqueeze(0) - t_a.unsqueeze(1)            # (B, B, Dt)
        ib = ib + beta * torch.sigmoid(tau_score * diff).mean(-1)

    ib.fill_diagonal_(float("-inf"))

    pos_l = (pos - margin) / tau
    hn_l = (hn / tau).masked_fill(~neg_valid, float("-inf"))
    ib_l = ib / tau
    logits = torch.cat([pos_l.unsqueeze(1), hn_l, ib_l], dim=1)
    target = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, target)

    if not bidirectional:
        return loss

    ib_rev = s_p @ s_a.t()
    if beta > 0 and t_a is not None:
        diff_r = t_a.unsqueeze(0) - t_p.unsqueeze(1)
        ib_rev = ib_rev + beta * torch.sigmoid(tau_score * diff_r).mean(-1)
    ib_rev.fill_diagonal_(float("-inf"))
    pos_l_r = (pos - margin) / tau
    logits_r = torch.cat([pos_l_r.unsqueeze(1), ib_rev / tau], dim=1)
    loss_r = F.cross_entropy(logits_r, target)
    return 0.5 * (loss + loss_r)


def ordering_loss(t_a, t_p, neg_time, neg_valid, margin=0.5, slack=0.1):
    """Push t_pos > t_a by margin (positive in the future); push t_neg < t_a by slack."""
    # positive: want (t_p - t_a) > margin in every dim
    loss_pos = F.relu(margin - (t_p - t_a)).mean(-1)               # (B,)
    # negative: want (t_neg - t_a) < -slack  → penalise where t_neg > t_a - slack
    diff_neg = neg_time - t_a.unsqueeze(1)                          # (B, K, Dt)
    per_neg = F.relu(slack + diff_neg).mean(-1)                     # (B, K)
    per_neg = per_neg.masked_fill(~neg_valid, 0.0)
    n_valid = neg_valid.sum(-1).clamp(min=1).float()
    loss_neg = per_neg.sum(-1) / n_valid                           # (B,)
    return (loss_pos + loss_neg).mean()


# ------------------------------------------------------------------ #
#  Validation (space-only, and optionally combined)
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate(model, tokenizer, pairs, device, max_length, anchor_prefix, positive_prefix,
             batch_size=128, k_values=(1, 3, 5, 10), beta=0.0, tau_score=10.0,
             dir_beta=0.0):
    model.eval()
    sa, sp, ta, tp = [], [], [], []
    # also encode the reverse-pair towers for a proper directional-accuracy read
    sap, sbanc, tap, tbanc = [], [], [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        a_ids, a_mask = tokenize_texts(tokenizer, [a for a,_ in chunk], anchor_prefix, max_length)
        p_ids, p_mask = tokenize_texts(tokenizer, [p for _,p in chunk], positive_prefix, max_length)
        a_ids, a_mask = a_ids.to(device), a_mask.to(device)
        p_ids, p_mask = p_ids.to(device), p_mask.to(device)
        s_a, t_a = model.encode_anchor(a_ids, a_mask)
        s_p, t_p = model.encode_positive(p_ids, p_mask)
        # reverse encodings: a through positive tower, b through anchor tower
        s_ap, t_ap = model.encode_positive(a_ids, a_mask)
        s_ba, t_ba = model.encode_anchor(p_ids, p_mask)
        sa.append(s_a.float().cpu()); sp.append(s_p.float().cpu())
        ta.append(t_a.float().cpu()); tp.append(t_p.float().cpu())
        sap.append(s_ap.float().cpu()); sbanc.append(s_ba.float().cpu())
        tap.append(t_ap.float().cpu()); tbanc.append(t_ba.float().cpu())
    model.train()
    A, P = torch.cat(sa), torch.cat(sp)
    TA, TP = torch.cat(ta), torch.cat(tp)
    Sap, Sba = torch.cat(sap), torch.cat(sbanc)
    Tap, Tba = torch.cat(tap), torch.cat(tbanc)

    def metrics_from_sim(sim):
        ranks = (sim > sim.diag().unsqueeze(1)).sum(1).float() + 1
        out = {"mrr": float((1/ranks).mean()), "median_rank": float(ranks.median())}
        for k in k_values:
            out[f"recall@{k}"] = float((ranks <= k).float().mean())
        return out

    sim_space = A @ P.T
    out = metrics_from_sim(sim_space)
    out["n_val"] = len(pairs)
    out["mean_rank"] = float(((sim_space > sim_space.diag().unsqueeze(1)).sum(1).float()+1).mean())

    if beta > 0:
        diff = TP.unsqueeze(0) - TA.unsqueeze(1)          # (Q, K, Dt)
        time_term = torch.sigmoid(tau_score * diff).mean(-1)
        sim_comb = sim_space + beta * time_term
        comb = metrics_from_sim(sim_comb)
        out["mrr_combined"] = comb["mrr"]
        out["median_rank_combined"] = comb["median_rank"]

    # Directional accuracy: score(a->b) > score(b->a) on properly re-encoded reverse pair.
    db = dir_beta if dir_beta > 0 else 1.0
    cos_fwd = (A * P).sum(-1)                                   # cos(anchor(a), pos(b))
    cos_rev = (Sba * Sap).sum(-1)                               # cos(anchor(b), pos(a))
    tt_fwd = torch.sigmoid(tau_score * (TP - TA)).mean(-1)      # time(a->b)
    tt_rev = torch.sigmoid(tau_score * (Tap - Tba)).mean(-1)    # time(b->a)
    out["dir_acc_space"] = float((cos_fwd > cos_rev).float().mean())
    out["dir_acc_combined"] = float(((cos_fwd + db * tt_fwd) > (cos_rev + db * tt_rev)).float().mean())
    return out


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def run_stage3(
    dataset, hard_neg_dir, data_dir=None, out_dir=None,
    backbone="intfloat/e5-base-v2",
    anchor_prefix="query: ", positive_prefix="passage: ",
    space_dim=0, time_dim=20, space_hidden=768, time_hidden=256, time_layers=2,
    dropout=0.1, pooling="mean", max_length=256,
    batch_size=128, grad_accumulation=1, epochs=5,
    lr=2e-5, temperature_lr=5e-4, temperature_init=0.05,
    weight_decay=0.01, warmup_ratio=0.1, grad_clip=1.0,
    alpha=1.0, beta=0.0, tau_score=10.0, nce_margin=0.02,
    lambda_nce=1.0, lambda_ord=0.0, ord_margin=0.5, ord_slack=0.1,
    lambda_dir=0.0, dir_margin=0.2, dir_beta=1.0,
    bidirectional=True, max_negatives=None,
    bf16=True, early_stopping_patience=3, early_stopping_min_delta=0.001,
    val_frac=0.1, max_val_samples=2000, num_workers=2, seed=42,
    stage2_ckpt=None, val_every_n_steps=None,
    select_on="mrr",
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hard_neg_dir = Path(hard_neg_dir)
    out_dir = Path(out_dir) if out_dir else hard_neg_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage3] device={device}  backbone={backbone}")
    print(f"[Stage3] prefix anchor={anchor_prefix!r} positive={positive_prefix!r}")

    hard_idx, train_pairs = load_hard_negatives(hard_neg_dir, split="train")
    if max_negatives is not None and max_negatives < hard_idx.shape[1]:
        hard_idx = hard_idx[:, :max_negatives]
    K = hard_idx.shape[1]
    print(f"[Stage3] {len(train_pairs)} pairs  K={K}  "
          f"bidir={bidirectional}  λ_ord={lambda_ord}  beta={beta}  time_layers={time_layers}")

    sys.path.insert(0, str(ROOT))
    from finetune_eval.datasets import split_path
    from finetune_eval.data import load_pairs as _lp
    vpath = split_path(dataset, "val")
    if vpath and vpath.exists():
        vp = _lp(vpath, cap=None)
        if max_val_samples and len(vp) > max_val_samples:
            rng = np.random.default_rng(seed)
            vp = [vp[int(i)] for i in sorted(rng.choice(len(vp), max_val_samples, replace=False))]
        val_pairs = vp
    else:
        n_v = min(max(1, int(len(train_pairs)*val_frac)), max_val_samples or len(train_pairs))
        rng = random.Random(seed); idx = list(range(len(train_pairs))); rng.shuffle(idx)
        val_pairs = [train_pairs[i] for i in sorted(idx[:n_v])]
    print(f"[Stage3] val pairs: {len(val_pairs)}")

    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    hidden = AutoConfig.from_pretrained(backbone).hidden_size
    if space_dim == 0:
        space_dim = hidden

    model = LorentzEncoder(
        backbone_name=backbone, space_dim=space_dim, time_dim=time_dim,
        space_hidden=space_hidden, time_hidden=time_hidden, time_layers=time_layers,
        dropout=dropout, pooling=pooling, tied=False,
    ).to(device)

    if stage2_ckpt and os.path.exists(stage2_ckpt):
        ck = torch.load(stage2_ckpt, map_location="cpu")
        model.load_state_dict(ck.get("model_state_dict", ck), strict=False)

    log_temp = nn.Parameter(torch.tensor(float(temperature_init)).log())
    cur_temp = temperature_init
    optimizer = AdamW([
        {"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay},
        {"params": [log_temp], "lr": temperature_lr, "weight_decay": 0.0},
    ])

    use_time = (lambda_ord > 0) or (beta > 0) or (lambda_dir > 0)
    # beta used at eval/scoring time: if we trained a directional time head, score with dir_beta
    eval_beta = beta if beta > 0 else (dir_beta if lambda_dir > 0 else 0.0)
    print(f"Stage3: {epochs}ep batch={batch_size} accum={grad_accumulation} "
          f"use_time={use_time} λ_dir={lambda_dir} dir_margin={dir_margin} "
          f"eval_beta={eval_beta} select_on={select_on}")

    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
              if (bf16 and device.type == "cuda") else (lambda: nullcontext())

    best_score, patience = -1.0, 0
    best_path = out_dir / "lorentz_best.pt"
    history, gstep, stop = [], 0, False

    def check_val(tag, ep, step):
        nonlocal best_score, patience, cur_temp
        vm = validate(model, tokenizer, val_pairs, device, max_length,
                      anchor_prefix, positive_prefix, beta=eval_beta, tau_score=tau_score,
                      dir_beta=dir_beta)
        score = vm.get("mrr_combined" if select_on == "combined" and "mrr_combined" in vm else "mrr", vm["mrr"])
        improved = score > best_score + early_stopping_min_delta
        if score > best_score:
            best_score = score
        if improved:
            patience = 0
            torch.save({
                "epoch": ep, "global_step": step, "model_state_dict": model.state_dict(),
                "metrics": vm,
                "cfg": {"backbone": backbone, "space_dim": space_dim, "time_dim": time_dim,
                        "space_hidden": space_hidden, "time_hidden": time_hidden,
                        "time_layers": time_layers, "pooling": pooling, "max_length": max_length,
                        "anchor_prefix": anchor_prefix, "positive_prefix": positive_prefix,
                        "alpha": alpha, "beta": eval_beta, "tau_score": tau_score, "tied": False,
                        "bidirectional": bidirectional, "temperature": cur_temp,
                        "lambda_ord": lambda_ord, "lambda_dir": lambda_dir,
                        "dir_beta": dir_beta, "use_time": use_time},
            }, best_path)
            tail = f"  -> new best {best_score:.4f} (saved)"
        else:
            patience += 1
            tail = f"  -> no impr (best={best_score:.4f}, p={patience}/{early_stopping_patience})"
        history.append({"tag": tag, "epoch": ep+1, "step": step, "val": vm})
        comb = f" comb={vm['mrr_combined']:.4f}" if "mrr_combined" in vm else ""
        dacc = f" dirAcc={vm['dir_acc_combined']:.3f}(sp {vm['dir_acc_space']:.3f})" if "dir_acc_combined" in vm else ""
        print(f"  [{tag}] MRR={vm['mrr']:.4f}{comb}{dacc}  R@1={vm['recall@1']:.3f}  "
              f"R@10={vm['recall@10']:.3f}  med={vm['median_rank']:.0f}  τ={cur_temp:.4f}")
        print(tail)

    for epoch in range(epochs):
        if stop:
            break
        print(f"\n[epoch {epoch+1}] pre-encoding {len(train_pairs)} negatives (space+time)…")
        t0 = time.time()
        neg_space, neg_time = pre_encode_negatives(
            model, tokenizer, train_pairs, hard_idx, positive_prefix, max_length, device,
            batch_size=min(batch_size*4, 512),
        )
        print(f"  done {time.time()-t0:.1f}s  space={neg_space.shape} time={neg_time.shape}")

        ds = NegEmbDataset(train_pairs, hard_idx, neg_space, neg_time)
        collate = make_collate(tokenizer, max_length, anchor_prefix, positive_prefix,
                               need_reverse=(lambda_dir > 0))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=collate, num_workers=num_workers, pin_memory=True)

        if epoch == 0:
            total_steps = max(len(loader) * epochs // max(grad_accumulation, 1), 1)
            scheduler = OneCycleLR(optimizer, max_lr=[lr, temperature_lr],
                                   total_steps=total_steps, pct_start=warmup_ratio)

        t0 = time.time()
        model.train()
        run_loss, run_ord, run_dir, n_b = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()

        for bi, batch in enumerate(loader):
            if lambda_dir > 0:
                a_ids, a_mask, p_ids, p_mask, neg_s, neg_t, neg_v, ap_ids, ap_mask, ba_ids, ba_mask = batch
                ap_ids, ap_mask = ap_ids.to(device), ap_mask.to(device)
                ba_ids, ba_mask = ba_ids.to(device), ba_mask.to(device)
            else:
                a_ids, a_mask, p_ids, p_mask, neg_s, neg_t, neg_v = batch
            a_ids, a_mask = a_ids.to(device), a_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)
            neg_s = neg_s.to(device); neg_t = neg_t.to(device); neg_v = neg_v.to(device)

            with amp_ctx():
                s_a, t_a = model.encode_anchor(a_ids, a_mask)
                s_p, t_p = model.encode_positive(p_ids, p_mask)

            s_a, s_p = s_a.float(), s_p.float()
            t_a, t_p = t_a.float(), t_p.float()
            tau = log_temp.exp().clamp(0.01, 0.5).float()

            l_nce = info_nce(
                s_a, s_p, neg_s, neg_v, tau, nce_margin,
                t_a=t_a if use_time else None, t_p=t_p if use_time else None,
                neg_time=neg_t if use_time else None,
                beta=beta, tau_score=tau_score, bidirectional=bidirectional,
            )
            loss = lambda_nce * l_nce
            l_ord_val = 0.0
            if lambda_ord > 0:
                l_ord = ordering_loss(t_a, t_p, neg_t, neg_v, margin=ord_margin, slack=ord_slack)
                loss = loss + lambda_ord * l_ord
                l_ord_val = l_ord.item()

            # ---- Directional margin loss: push score(a->b) > score(b->a) + margin ----
            # Direction lives in the UNTIED TOWERS (space), so we train the towers
            # directly (full backbone forward, with grad). dir_beta>0 additionally
            # routes the signal through the time head. We re-encode the reverse pair
            # (a through positive tower, b through anchor tower) with grad.
            l_dir_val = 0.0
            if lambda_dir > 0:
                # Reuse forward encodings from the InfoNCE block (s_a=a·anchor, s_p=b·positive);
                # only the reverse-pair towers (a·positive, b·anchor) are new.
                with amp_ctx():
                    s_a_pos, t_a_pos = model.encode_positive(ap_ids, ap_mask)
                    s_b_anc, t_b_anc = model.encode_anchor(ba_ids, ba_mask)
                cos_fwd = (s_a * s_p).sum(-1)                          # already fp32
                cos_rev = (s_b_anc.float() * s_a_pos.float()).sum(-1)
                score_fwd, score_rev = cos_fwd, cos_rev
                if dir_beta > 0:
                    tt_fwd = torch.sigmoid(tau_score * (t_p - t_a)).mean(-1)
                    tt_rev = torch.sigmoid(tau_score * (t_a_pos.float() - t_b_anc.float())).mean(-1)
                    score_fwd = score_fwd + dir_beta * tt_fwd
                    score_rev = score_rev + dir_beta * tt_rev
                l_dir = F.relu(dir_margin - (score_fwd - score_rev)).mean()
                loss = loss + lambda_dir * l_dir
                l_dir_val = l_dir.item()

            if not torch.isfinite(loss):
                optimizer.zero_grad(); continue

            (loss / grad_accumulation).backward()
            run_loss += l_nce.item(); run_ord += l_ord_val; run_dir += l_dir_val; n_b += 1

            if (bi + 1) % grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if log_temp.grad is not None:
                    if not torch.isfinite(log_temp.grad):
                        log_temp.grad.zero_()
                    else:
                        log_temp.grad.clamp_(-0.3, 0.3)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                cur_temp = float(log_temp.exp().clamp(0.01, 0.5).item())
                gstep += 1
                if val_every_n_steps and gstep % val_every_n_steps == 0:
                    check_val("intra", epoch, gstep)
                    model.train()
                    if patience >= early_stopping_patience:
                        stop = True; break
        if stop:
            break
        print(f"epoch {epoch+1} ({time.time()-t0:.0f}s) nce={run_loss/max(n_b,1):.4f} "
              f"ord={run_ord/max(n_b,1):.4f} dir={run_dir/max(n_b,1):.4f} τ={cur_temp:.4f}")
        check_val("eoe", epoch, gstep)
        if patience >= early_stopping_patience:
            print("[early stop]"); break

    (out_dir / "lorentz_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[Stage3 done] best val {select_on}={best_score:.4f} → {best_path}")
    return best_score
