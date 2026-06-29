"""
Pairwise / cross-encoder approach for causal direction.

Instead of two separate embeddings + cosine (previous run), here the model
sees the FULL PAIR as a single input:

  positive input: instruction + "PASSAGE 1: a\n\nPASSAGE 2: b"  → scalar score ↑
  negative input: instruction + "PASSAGE 1: b\n\nPASSAGE 2: a"  → scalar score ↓

Architecture:
  GRITLM-7B backbone (LoRA, bidirectional attn, mean pool, no instr tokens)
  + Linear head (4096 → 1) → causal order score

Loss:
  L_rank = relu(margin - score_pos + score_neg).mean()   ranking loss
  L_bce  = BCE([score_pos, score_neg], [1, 0])           binary calibration
  L = L_rank + L_bce

Direction acc:  score(a,b) > score(b,a)   (N=562 test pairs)
Retrieval:      for each anchor a, rank score(a, b_i) over N candidates (cross-encoder is O(N²))
"""
import json, os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda:0"
MODEL = "GritLM/GritLM-7B"
DATA = "/mnt/localssd/internship-causal-embedding-merge-followup/data_6/aep_causal"

MAXLEN   = int(os.environ.get("MAXLEN",   512))
B        = int(os.environ.get("B",        4))     # per-step batch; use ACCUM for effective batch
ACCUM    = int(os.environ.get("ACCUM",    2))     # gradient accumulation steps
EPOCHS   = int(os.environ.get("EPOCHS",   1))
N_TRAIN  = int(os.environ.get("N_TRAIN",  4000))
MAX_STEPS= int(os.environ.get("MAX_STEPS",0))
LR       = float(os.environ.get("LR",     1e-4))
MARGIN   = float(os.environ.get("MARGIN", 0.3))
N_EVAL_DIR = int(os.environ.get("N_EVAL_DIR", 562))
N_EVAL_RET = int(os.environ.get("N_EVAL_RET", 50))   # cross-encoder is O(N²), keep modest

# Instruction for the PAIR (not role-specific, just ordering)
INSTR = ("<|user|>\n"
         "The following two passages are connected in time: PASSAGE 1 happens first "
         "(at time t) and PASSAGE 2 happens second (at time t+1). Represent this ordered "
         "pair so that a valid causal sequence — where PASSAGE 1 leads to or causes "
         "PASSAGE 2 — scores higher than a reversed or non-causal ordering.\n"
         "<|embed|>\n")

SEP = "\n\n"

def make_input(text1, text2):
    return INSTR + "PASSAGE 1: " + text1 + SEP + "PASSAGE 2: " + text2

def load(split):
    rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl") if l.strip()]
    return [(r["anchor"], r["positive"], r.get("causal_type","")) for r in rows]

# ---- model ----------------------------------------------------------------
print("Loading model + tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right", trust_remote_code=True)
if not tok.pad_token: tok.pad_token = tok.eos_token

backbone = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16).to(DEV)

# scalar head in float32 for stability
score_head = nn.Linear(backbone.config.hidden_size, 1, bias=True).to(DEV).float()
nn.init.normal_(score_head.weight, std=0.01)

# instruction token length (for pooling mask)
INSTR_LEN = len(tok(INSTR, add_special_tokens=True)["input_ids"])

def embed_pair(texts, grad):
    """texts: list of already-formatted 'INSTR + PASSAGE1 + SEP + PASSAGE2' strings"""
    enc = tok(texts, padding=True, truncation=True, max_length=MAXLEN,
              return_tensors="pt", add_special_tokens=True).to(DEV)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = backbone(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"],
                       is_causal=False)
        h = out[0]                                      # [B, seq, 4096]
        m = enc["attention_mask"].clone()
        m[:, :INSTR_LEN] = 0                            # exclude instruction from pooling
        m = m.unsqueeze(-1).float()
        e = (h.float() * m).sum(1) / m.sum(1).clamp(min=1)   # mean pool, float32
    return e                                            # [B, 4096], NOT normalized

def score(texts, grad):
    e = embed_pair(texts, grad)
    return score_head(e).squeeze(-1)                    # [B]

# ---- eval -----------------------------------------------------------------
@torch.no_grad()
def eval_direction(pairs, n):
    backbone.eval(); score_head.eval()
    sub = pairs[:n]
    correct = 0; fwds = []; revs = []
    bs = 16
    for i in range(0, len(sub), bs):
        batch = sub[i:i+bs]
        pos_texts = [make_input(a, b) for a,b,_ in batch]
        neg_texts = [make_input(b, a) for a,b,_ in batch]
        s_pos = score(pos_texts, grad=False)
        s_neg = score(neg_texts, grad=False)
        correct += (s_pos > s_neg).sum().item()
        fwds.extend(s_pos.cpu().tolist())
        revs.extend(s_neg.cpu().tolist())
    acc = correct / len(sub)
    return acc, float(np.mean(fwds)), float(np.mean(revs)), float(np.mean(np.array(fwds)-np.array(revs)))

@torch.no_grad()
def eval_retrieval(pairs, n):
    """Cross-encoder retrieval: score every (anchor_i, effect_j) pair. O(N²)."""
    backbone.eval(); score_head.eval()
    sub = pairs[:n]
    A = [p[0] for p in sub]; Bt = [p[1] for p in sub]
    # build N x N score matrix
    S = torch.zeros(len(sub), len(sub))
    bs = 8
    for i in range(len(sub)):
        texts = [make_input(A[i], Bt[j]) for j in range(len(sub))]
        for k in range(0, len(texts), bs):
            s = score(texts[k:k+bs], grad=False)
            S[i, k:k+len(s)] = s.cpu()
    ranks = []
    for i in range(len(sub)):
        order = torch.argsort(S[i], descending=True)
        ranks.append(int((order == i).nonzero()[0,0]) + 1)
    ranks = np.array(ranks)
    return float((ranks==1).mean()), float((1/ranks).mean())

# ---- LoRA -----------------------------------------------------------------
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
backbone = get_peft_model(backbone, lcfg)
backbone.print_trainable_parameters()
# gradient checkpointing reduces activation memory ~50% at cost of ~30% speed
try:
    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()
except Exception as e:
    print("grad ckpt:", e)

params = (list(p for p in backbone.parameters() if p.requires_grad)
          + list(score_head.parameters()))
opt = torch.optim.AdamW(params, lr=LR)

train = load("train"); test = load("test")
random.shuffle(train); train = train[:N_TRAIN]
print(f"train={len(train)}  test={len(test)}  MAXLEN={MAXLEN}  B={B}  ACCUM={ACCUM}  "
      f"effective_batch={B*ACCUM}  EPOCHS={EPOCHS}")

# ---- pre-eval -------------------------------------------------------------
print("\n=== PRE-finetune ===")
acc,f,r,mg = eval_direction(test, N_EVAL_DIR)
a1,mrr = eval_retrieval(test, N_EVAL_RET)
print(f"direction acc={acc:.3f}  score_pos={f:+.3f}  score_neg={r:+.3f}  margin={mg:+.3f}")
print(f"retrieval acc@1={a1:.3f}  MRR={mrr:.3f}")

# ---- train ----------------------------------------------------------------
print("\n=== TRAIN ===")
# Effective batch = B * ACCUM; optimizer step every ACCUM mini-batches
step = 0; t0 = time.time()
opt.zero_grad()
for ep in range(EPOCHS):
    backbone.train(); score_head.train()
    random.shuffle(train)
    accum_loss = accum_rank = accum_bce = 0.0
    accum_pos = accum_neg = 0.0
    for i in range(0, len(train)-B+1, B):
        batch = train[i:i+B]
        pos_texts = [make_input(a, b) for a,b,_ in batch]
        neg_texts = [make_input(b, a) for a,b,_ in batch]
        s_pos = score(pos_texts, grad=True)
        s_neg = score(neg_texts, grad=True)
        L_rank = F.relu(MARGIN - s_pos + s_neg).mean()
        targets = torch.cat([torch.ones(B), torch.zeros(B)]).to(DEV)
        L_bce = F.binary_cross_entropy_with_logits(
            torch.cat([s_pos, s_neg]), targets)
        loss = (L_rank + L_bce) / ACCUM
        loss.backward()
        accum_loss += (L_rank + L_bce).item()
        accum_rank += L_rank.item()
        accum_bce  += L_bce.item()
        accum_pos  += s_pos.mean().item()
        accum_neg  += s_neg.mean().item()
        mini = (i // B) + 1
        if mini % ACCUM == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step % 25 == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated(DEV)/1e9
                print(f"  ep{ep} step{step:4d}  loss={accum_loss/ACCUM:.4f} "
                      f"(rank={accum_rank/ACCUM:.4f} bce={accum_bce/ACCUM:.4f})  "
                      f"pos={accum_pos/ACCUM:+.3f}  neg={accum_neg/ACCUM:+.3f}  "
                      f"maxmem={mem:.1f}GB  t={time.time()-t0:.0f}s", flush=True)
            accum_loss = accum_rank = accum_bce = accum_pos = accum_neg = 0.0
            if MAX_STEPS and step >= MAX_STEPS:
                break
    if MAX_STEPS and step >= MAX_STEPS:
        break

# ---- post-eval ------------------------------------------------------------
print("\n=== POST-finetune ===")
acc,f,r,mg = eval_direction(test, N_EVAL_DIR)
a1,mrr = eval_retrieval(test, N_EVAL_RET)
print(f"direction acc={acc:.3f}  score_pos={f:+.3f}  score_neg={r:+.3f}  margin={mg:+.3f}")
print(f"retrieval acc@1={a1:.3f}  MRR={mrr:.3f}")
print("\nDONE")
