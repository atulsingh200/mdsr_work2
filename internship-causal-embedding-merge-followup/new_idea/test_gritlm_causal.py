"""
Test GRITLM-7B as a CAUSAL embedding model.

Goal: text at time t (cause / step) and t+1 (effect / next-step) should score HIGH,
while text that is merely semantically similar (a paraphrase of the cause) should
NOT necessarily score high. We probe whether a *causal instruction* steers GRITLM
toward causal relatedness vs a plain *semantic instruction*.
"""
import json, sys
import numpy as np
import torch
from gritlm import GritLM

torch.manual_seed(0)

# ----- instructions -------------------------------------------------------
def fmt(instr):
    return "<|user|>\n" + instr + "\n<|embed|>\n" if instr else "<|embed|>\n"

SEM = "Represent the text to find another text with the same meaning."
# Causal asymmetric instructions: query = cause/step at time t, doc = effect/next step at t+1
CAUSE_Q = ("Given a step, action, or event that occurs at one point in a process, "
           "represent it to retrieve the text that describes the effect, consequence, "
           "or the next step that causally follows from it.")
EFFECT_D = ("Represent this text as the effect, consequence, or next step so it can be "
            "matched to the action or event that causes it.")

def cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T

print("Loading GritLM-7B (embedding mode, bf16)...", flush=True)
model = GritLM("GritLM/GritLM-7B", torch_dtype=torch.bfloat16, mode="embedding", device_map="cuda:0")

def emb(texts, instr, maxlen=512):
    return np.asarray(model.encode(texts, instruction=fmt(instr), max_length=maxlen,
                                   batch_size=8)).astype(np.float32)

# ============================================================ PART A: toy
print("\n" + "="*70)
print("PART A — toy causal vs semantic example")
print("="*70)
cause = "Turn off the light switch."
cand = {
    "EFFECT (causal t+1) : The bulb goes dark and the room is no longer lit.": "effect",
    "PARAPHRASE (semantic): Flip the light switch to the off position.":       "para",
    "PARAPHRASE (semantic): Switch off the lamp.":                              "para",
    "UNRELATED            : The stock market fell sharply today.":              "unrel",
}
ctexts = list(cand.keys())

for name, qi, di in [("SEMANTIC instr", SEM, SEM), ("CAUSAL instr", CAUSE_Q, EFFECT_D)]:
    qe = emb([cause], qi)
    de = emb(ctexts, di)
    s = cos(qe, de)[0]
    print(f"\n[{name}]  query = {cause!r}")
    order = np.argsort(-s)
    for r, idx in enumerate(order):
        print(f"  #{r+1}  {s[idx]:+.4f}  {ctexts[idx]}")

# ============================================================ PART B: real data
print("\n" + "="*70)
print("PART B — real AEP causal pairs (cause=anchor @t, effect=positive @t+1)")
print("="*70)
DATA = "/mnt/localssd/internship-causal-embedding-merge-followup/data_6/aep_causal/test.jsonl"
rows = [json.loads(l) for l in open(DATA) if l.strip()]
K = 10
rng = np.random.default_rng(42)
sel = rng.choice(len(rows), size=K, replace=False)
pairs = [rows[i] for i in sel]
causes  = [p["anchor"]   for p in pairs]
effects = [p["positive"] for p in pairs]
print(f"\nSelected {K} pairs. causal_types: {[p['causal_type'][:12] for p in pairs]}")

def eval_retrieval(name, qi, di):
    qe = emb(causes, qi)
    de = emb(effects, di)
    S = cos(qe, de)                       # K x K, row=cause, col=effect
    ranks = []
    for i in range(K):
        order = np.argsort(-S[i])
        rank = int(np.where(order == i)[0][0]) + 1   # rank of TRUE effect
        ranks.append(rank)
    ranks = np.array(ranks)
    acc1 = float((ranks == 1).mean())
    mrr = float((1.0/ranks).mean())
    # mean diagonal (true) vs mean off-diagonal (distractor) similarity
    diag = np.diag(S).mean()
    off = (S.sum() - np.trace(S)) / (K*K - K)
    print(f"\n[{name}]  acc@1={acc1:.2f}  MRR={mrr:.3f}  "
          f"mean_true_sim={diag:+.3f}  mean_distractor_sim={off:+.3f}  gap={diag-off:+.3f}")
    print(f"   per-pair rank of true effect: {ranks.tolist()}")
    return S

S_sem = eval_retrieval("SEMANTIC instr", SEM, SEM)
S_cau = eval_retrieval("CAUSAL instr",   CAUSE_Q, EFFECT_D)

# Show one concrete pair: true effect score vs best distractor under each instr
print("\n--- example pair #0 (truncated) ---")
print("CAUSE  :", causes[0][:160].replace("\n"," "), "...")
print("EFFECT :", effects[0][:160].replace("\n"," "), "...")
for nm, S in [("SEM", S_sem), ("CAUSAL", S_cau)]:
    row = S[0]
    best_distractor = max((row[j], j) for j in range(K) if j != 0)
    print(f"  [{nm}] sim(true)={row[0]:+.3f}  best distractor sim={best_distractor[0]:+.3f} (pair {best_distractor[1]})")
print("\nDONE")
