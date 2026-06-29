"""
LoRA-finetune GRITLM-7B to capture CAUSAL DIRECTION on aep_causal, then test
whether it ranks forward (a->b) above reverse (b->a).

Objective per batch of pairs (a=cause@t, b=effect@t+1):
  Ca = embed(a, CAUSE_instr)   Ea = embed(a, EFFECT_instr)
  Cp = embed(b, CAUSE_instr)   Ep = embed(b, EFFECT_instr)
  forward_i = Ca_i . Ep_i      reverse_i = Cp_i . Ea_i
  L_info = CE( Ca @ [Ep ; Ea]^T / temp , labels=i )   # in-batch + a-as-effect hard neg
  L_dir  = relu(margin - forward + reverse).mean()      # rank correct direction first
  L = L_info + lambda_dir * L_dir
"""
import json, os, time, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda:0"
MODEL = "GritLM/GritLM-7B"
DATA = "/mnt/localssd/internship-causal-embedding-merge-followup/data_6/aep_causal"
MAXLEN = int(os.environ.get("MAXLEN", 256))
B = int(os.environ.get("B", 8))
EPOCHS = int(os.environ.get("EPOCHS", 1))
N_TRAIN = int(os.environ.get("N_TRAIN", 4000))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))  # 0 = no cap
LR = 1e-4
TEMP = 0.05
MARGIN = 0.05
LAMBDA_DIR = 1.0
N_EVAL_DIR = 200      # pairs for direction accuracy
N_EVAL_RET = 50       # pairs for retrieval acc@1 / MRR

EMB = "<|embed|>\n"
def cause_instr():  return "<|user|>\n" + (
    "Causality is directional: a cause or preceding step happens first (time t) and "
    "produces an effect or next step later (time t+1). The following text is the CAUSE / "
    "preceding step. Represent it to match the text describing the effect or next step it "
    "leads to.") + "\n" + EMB
def effect_instr(): return "<|user|>\n" + (
    "Causality is directional: a cause or preceding step happens first (time t) and "
    "produces an effect or next step later (time t+1). The following text is the EFFECT / "
    "next step. Represent it to match the text describing the cause or preceding step that "
    "produced it.") + "\n" + EMB
CAUSE, EFFECT = cause_instr(), effect_instr()

def load(split):
    rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl") if l.strip()]
    return [(r["anchor"], r["positive"], r.get("causal_type","")) for r in rows]

print("Loading model + tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right", trust_remote_code=True)
if not tok.pad_token: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEV)

# precompute instruction token lengths (with BOS) for pooling masks
LEN_CAUSE  = len(tok(CAUSE,  add_special_tokens=True)["input_ids"])
LEN_EFFECT = len(tok(EFFECT, add_special_tokens=True)["input_ids"])

def embed(texts, instruction, instr_len, grad):
    full = [instruction + t for t in texts]
    enc = tok(full, padding=True, truncation=True, max_length=MAXLEN,
              return_tensors="pt", add_special_tokens=True).to(DEV)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], is_causal=False)
        h = out[0]
        m = enc["attention_mask"].clone()
        m[:, :instr_len] = 0                       # exclude instruction tokens from pooling
        m = m.unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
    return F.normalize(emb.float(), dim=-1)

# ---------------- evaluation ----------------
@torch.no_grad()
def eval_direction(pairs, n):
    model.eval()
    sub = pairs[:n]
    A = [p[0] for p in sub]; Bt = [p[1] for p in sub]
    Ca = embed(A,  CAUSE,  LEN_CAUSE,  False)
    Ea = embed(A,  EFFECT, LEN_EFFECT, False)
    Cp = embed(Bt, CAUSE,  LEN_CAUSE,  False)
    Ep = embed(Bt, EFFECT, LEN_EFFECT, False)
    fwd = (Ca * Ep).sum(-1)      # a->b
    rev = (Cp * Ea).sum(-1)      # b->a
    acc = (fwd > rev).float().mean().item()
    return acc, fwd.mean().item(), rev.mean().item(), (fwd - rev).mean().item()

@torch.no_grad()
def eval_retrieval(pairs, n):
    model.eval()
    sub = pairs[:n]
    A = [p[0] for p in sub]; Bt = [p[1] for p in sub]
    Ca = embed(A,  CAUSE,  LEN_CAUSE,  False)
    Ep = embed(Bt, EFFECT, LEN_EFFECT, False)
    S = Ca @ Ep.T                # n x n
    ranks = []
    for i in range(len(sub)):
        order = torch.argsort(S[i], descending=True)
        ranks.append(int((order == i).nonzero()[0,0]) + 1)
    ranks = np.array(ranks)
    return float((ranks==1).mean()), float((1/ranks).mean())

train = load("train")
test  = load("test")
random.shuffle(train)
train = train[:N_TRAIN]

print(f"train pairs={len(train)}  test pairs={len(test)}  MAXLEN={MAXLEN} B={B} epochs={EPOCHS}")
print("\n=== PRE-finetune ===")
acc,f,r,m = eval_direction(test, N_EVAL_DIR)
a1,mrr = eval_retrieval(test, N_EVAL_RET)
print(f"direction acc={acc:.3f}  fwd={f:+.4f} rev={r:+.4f} margin={m:+.4f}")
print(f"retrieval acc@1={a1:.3f} MRR={mrr:.3f}")

# ---------------- LoRA ----------------
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.base_model.gradient_checkpointing_enable() if hasattr(model.base_model,"gradient_checkpointing_enable") else None
try:
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
except Exception as e:
    print("grad ckpt note:", e)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

def encode_role(texts, instruction, instr_len):
    return embed(texts, instruction, instr_len, grad=True)

print("\n=== TRAIN ===")
model.train()
step = 0
t0 = time.time()
for ep in range(EPOCHS):
    random.shuffle(train)
    for i in range(0, len(train) - B + 1, B):
        batch = train[i:i+B]
        A  = [x[0] for x in batch]
        Bt = [x[1] for x in batch]
        Ca = encode_role(A,  CAUSE,  LEN_CAUSE)    # cause role
        Ea = encode_role(A,  EFFECT, LEN_EFFECT)   # a as effect (hard neg side)
        Cp = encode_role(Bt, CAUSE,  LEN_CAUSE)    # b as cause (reverse query)
        Ep = encode_role(Bt, EFFECT, LEN_EFFECT)   # effect role (true positive)
        # InfoNCE: Ca_i should pick Ep_i over all Ep_j and all Ea_j
        docs = torch.cat([Ep, Ea], dim=0)          # 2B x d
        logits = Ca @ docs.T / TEMP                # B x 2B
        labels = torch.arange(B, device=DEV)
        L_info = F.cross_entropy(logits, labels)
        # direction margin
        fwd = (Ca * Ep).sum(-1)
        rev = (Cp * Ea).sum(-1)
        L_dir = F.relu(MARGIN - fwd + rev).mean()
        loss = L_info + LAMBDA_DIR * L_dir
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 25 == 0 or step == 1:
            mem = torch.cuda.max_memory_allocated(DEV)/1e9
            print(f"  ep{ep} step{step:4d}  loss={loss.item():.4f} (info={L_info.item():.4f} dir={L_dir.item():.4f}) "
                  f"fwd={fwd.mean().item():+.3f} rev={rev.mean().item():+.3f} maxmem={mem:.1f}GB t={time.time()-t0:.0f}s", flush=True)
        if MAX_STEPS and step >= MAX_STEPS:
            break
    if MAX_STEPS and step >= MAX_STEPS:
        break

print("\n=== POST-finetune ===")
acc,f,r,m = eval_direction(test, N_EVAL_DIR)
a1,mrr = eval_retrieval(test, N_EVAL_RET)
print(f"direction acc={acc:.3f}  fwd={f:+.4f} rev={r:+.4f} margin={m:+.4f}")
print(f"retrieval acc@1={a1:.3f} MRR={mrr:.3f}")
print("\nDONE")
