"""Directional classification on aep_causal_classification using the SOTA
CDEv2 + CE-rerank architecture (the same components that won the retrieval task).

Task: given an ordered pair (text_1, text_2), predict label=1 if text_1 causally
precedes text_2, else 0.

Two trainable components (both bert-base-uncased — the CDEv2 base model):

  --arch cde   Stage-1 retriever as a directional classifier. Uses the real
               CDEv2 KL+cosine score with causal transport:
                   logit = scale * (score(t1->t2) - score(t2->t1)) + bias
               Warm-started from the trained retriever checkpoint (run5c).

  --arch ce    Stage-2 cross-encoder as a directional classifier:
                   [CLS] text_1 [SEP] text_2 [SEP] -> BERT CLS -> Linear(1)
               (the exact CrossEncoder class used by the reranker).

Then `--arch fuse` evaluates the full architecture: z-scored CDEv2 margin +
CE logit, blended (same fusion as the retrieval reranker), reporting the
combined accuracy/AUC/F1.

Loss BCEWithLogits. Metrics identical to the reference classifier:
  acc=mean((sigmoid>0.5)==y), AUC=roc_auc(y,prob), F1=f1(y,pred).
Reference (bge-small dual+MLP, trained on this data): acc 0.8571 AUC 0.9031 F1 0.8558
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
PROJECT_ROOT = KL_DIR.parent.parent
sys.path.insert(0, str(KL_DIR)); sys.path.insert(0, str(KL_DIR / "cross_encoder"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v2.model_v2 import CDEv2                                 # noqa: E402
from model_ce import CrossEncoder                             # noqa: E402

DATA = PROJECT_ROOT / "data_6" / "aep_causal_classification"
OUT_ROOT = KL_DIR / "results" / "directional"
RETR_CKPT = KL_DIR / "results" / "aep_causal" / "cde_v2_run5c_best.pt"


class PairRows(Dataset):
    def __init__(self, path, limit=None):
        self.rows = []
        for i, line in enumerate(open(path)):
            if limit and i >= limit:
                break
            r = json.loads(line)
            self.rows.append((r["text_1"], r["text_2"], float(r["label"])))

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


# ── collators ──────────────────────────────────────────────────────────────
class CECollator:
    def __init__(self, tok, max_len): self.tok, self.max_len = tok, max_len
    def __call__(self, batch):
        t1=[b[0] for b in batch]; t2=[b[1] for b in batch]
        y=torch.tensor([b[2] for b in batch],dtype=torch.float32)
        enc=self.tok(t1,t2,padding=True,truncation="longest_first",max_length=self.max_len,
                     return_tensors="pt",return_token_type_ids=True)
        return enc, y

class CDECollator:
    def __init__(self, tok, max_len): self.tok, self.max_len = tok, max_len
    def _t(self, texts):
        e=self.tok(texts,padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
        return e["input_ids"], e["attention_mask"]
    def __call__(self, batch):
        t1=[b[0] for b in batch]; t2=[b[1] for b in batch]
        y=torch.tensor([b[2] for b in batch],dtype=torch.float32)
        return (self._t(t1), self._t(t2)), y


# ── models ───────────────────────────────────────────────────────────────────
class CEClassifier(nn.Module):
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.ce = CrossEncoder(backbone, dropout=dropout)   # BERT + Linear(1)
    def forward(self, enc):
        return self.ce(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))


class CDEDirectional(nn.Module):
    """Wrap the trained CDEv2 retriever; logit = scale*(s_fwd - s_rev) + bias."""
    def __init__(self, backbone, warm_ckpt=None):
        super().__init__()
        if warm_ckpt is not None and Path(warm_ckpt).exists():
            ck = torch.load(warm_ckpt, map_location="cpu", weights_only=False); cfg = ck["cfg"]
            self.cde = CDEv2(cfg["backbone"], cfg["proj_dim"], shared_encoder=cfg.get("shared_encoder", False),
                             kl_scale=cfg.get("kl_scale","dim"), init_cos_weight=cfg.get("init_cos_weight",1.0),
                             pooling=cfg.get("pooling","mean"), mu_identity=cfg.get("mu_identity",False),
                             log_sigma_init=cfg.get("log_sigma_init",0.0), score_type=cfg.get("score_type","kl"))
            self.cde.load_state_dict(ck["model_state_dict"], strict=False)
            print(f"  [warm-start CDE from {Path(warm_ckpt).name}]")
        else:
            self.cde = CDEv2(backbone, 768, pooling="cls", mu_identity=True, log_sigma_init=3.0,
                             kl_scale="dim", init_cos_weight=1.0, score_type="kl")
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _score(self, c_ids, c_mask, e_ids, e_mask):
        mu, ls, de = self.cde.encode_cause(c_ids, c_mask)
        mu, ls = self.cde.transport(mu, ls, de)
        mb, lb = self.cde.encode_effect(e_ids, e_mask)
        return self.cde.score_pointwise(mu, ls, mb, lb)

    def forward(self, pair):
        (i1, m1), (i2, m2) = pair
        s_fwd = self._score(i1, m1, i2, m2)
        s_rev = self._score(i2, m2, i1, m1)
        return self.scale * (s_fwd - s_rev) + self.bias


@torch.no_grad()
def evaluate(model, loader, device, arch, return_probs=False):
    model.eval(); probs=[]; labels=[]
    for batch, y in loader:
        if arch == "ce":
            batch = {k: v.to(device) for k, v in batch.items()}
        else:
            batch = tuple((i.to(device), m.to(device)) for (i, m) in batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = model(batch)
        probs.extend(torch.sigmoid(logit).float().cpu().tolist()); labels.extend(y.tolist())
    probs=np.array(probs); labels=np.array(labels); preds=(probs>0.5).astype(int)
    m={"acc":float((preds==labels).mean()),"auc":float(roc_auc_score(labels,probs)),"f1":float(f1_score(labels,preds))}
    return (m, probs, labels) if return_probs else m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch", default="ce", choices=["ce","cde"])
    ap.add_argument("--backbone", default="google-bert/bert-base-uncased")
    ap.add_argument("--warm-ckpt", default=str(RETR_CKPT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-suffix", default="")
    args=ap.parse_args()

    device=torch.device("cuda")
    tok=AutoTokenizer.from_pretrained(args.backbone)
    out_dir=OUT_ROOT/f"{args.arch}{args.out_suffix}"; out_dir.mkdir(parents=True, exist_ok=True)
    coll = CECollator(tok,args.max_length) if args.arch=="ce" else CDECollator(tok,args.max_length)

    dl_tr=DataLoader(PairRows(DATA/"directional_train.jsonl",args.limit),batch_size=args.batch_size,
                     shuffle=True,collate_fn=coll,num_workers=args.num_workers,pin_memory=True)
    dl_va=DataLoader(PairRows(DATA/"directional_val.jsonl"),batch_size=args.batch_size*2,
                     shuffle=False,collate_fn=coll,num_workers=args.num_workers,pin_memory=True)
    dl_te=DataLoader(PairRows(DATA/"directional_test.jsonl"),batch_size=args.batch_size*2,
                     shuffle=False,collate_fn=coll,num_workers=args.num_workers,pin_memory=True)
    print(f"[data] train={len(dl_tr.dataset)} val={len(dl_va.dataset)} test={len(dl_te.dataset)}  arch={args.arch}")

    model=(CEClassifier(args.backbone) if args.arch=="ce"
           else CDEDirectional(args.backbone, args.warm_ckpt)).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    enc_p,head_p=[],[]
    for nm,p in model.named_parameters():
        is_enc = any(k in nm for k in ("bert","cause_enc","effect_enc","backbone"))
        (enc_p if is_enc else head_p).append(p)
    opt=torch.optim.AdamW([{"params":enc_p,"lr":args.lr},{"params":head_p,"lr":args.head_lr}],
                          weight_decay=args.weight_decay)
    total=len(dl_tr)*args.epochs
    sched=get_cosine_schedule_with_warmup(opt,int(total*args.warmup_ratio),total)
    loss_fn=nn.BCEWithLogitsLoss()

    best_acc=-1; best_state=None; hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); t0=time.time(); rc=rt=0
        for step,(batch,y) in enumerate(dl_tr):
            if args.arch=="ce":
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            else:
                batch=tuple((i.to(device,non_blocking=True),m.to(device,non_blocking=True)) for (i,m) in batch)
            y=y.to(device)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                logit=model(batch); loss=loss_fn(logit,y)
            opt.zero_grad(); loss.backward()
            if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            opt.step(); sched.step()
            rc+=((torch.sigmoid(logit.detach())>0.5).float()==y).sum().item(); rt+=len(y)
            if step%200==0:
                print(f"  ep{ep} {step}/{len(dl_tr)} loss={loss.item():.4f} acc={rc/max(rt,1):.4f}",flush=True)
        vm=evaluate(model,dl_va,device,args.arch)
        hist.append({"epoch":ep,"train_acc":rc/rt,**{f"val_{k}":v for k,v in vm.items()}})
        print(f"[epoch {ep}] train_acc={rc/rt:.4f} val_acc={vm['acc']:.4f} val_auc={vm['auc']:.4f} val_f1={vm['f1']:.4f} ({time.time()-t0:.0f}s)",flush=True)
        if vm["acc"]>best_acc:
            best_acc=vm["acc"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f"  * new best val_acc={best_acc:.4f}")

    model.load_state_dict(best_state)
    tm,probs,labels=evaluate(model,dl_te,device,args.arch,return_probs=True)
    print(f"\n[TEST {args.arch}]  acc={tm['acc']:.4f}  AUC={tm['auc']:.4f}  F1={tm['f1']:.4f}")
    print(f"[reference bge-small]  acc=0.8571  AUC=0.9031  F1=0.8558  | acc delta {(tm['acc']-0.8571)/0.8571*100:+.1f}%")
    np.save(out_dir/"test_probs.npy", probs); np.save(out_dir/"test_labels.npy", labels)
    (out_dir/"test_metrics.json").write_text(json.dumps(
        {"test":tm,"val_best_acc":best_acc,"history":hist,"config":vars(args),
         "reference":{"acc":0.8571,"auc":0.9031,"f1":0.8558}},indent=2))
    torch.save({"model_state_dict":best_state,"cfg":vars(args)}, out_dir/"best.pt")
    print(f"[saved] {out_dir}")


if __name__=="__main__":
    main()
