"""
GRITLM causal DIRECTIONALITY test.

For each causal pair (a=cause@t, b=effect@t+1):
  forward = cos(emb_cause(a), emb_effect(b))   -> correct direction  (label 1)
  reverse = cos(emb_cause(b), emb_effect(a))   -> reversed direction  (label 0)
Correct iff forward > reverse. (Direction can ONLY emerge from the asymmetric
cause-role vs effect-role instructions, since cosine itself is symmetric.)
"""
import json
import numpy as np
import torch
from gritlm import GritLM

def fmt(instr):
    return "<|user|>\n" + instr + "\n<|embed|>\n" if instr else "<|embed|>\n"

# Full-context causal instructions, explaining that causality is DIRECTIONAL.
CAUSE_Q = ("Causality is directional: a cause or preceding step happens first (at time t) "
           "and produces an effect or next step later (at time t+1). The following text is "
           "the CAUSE / preceding step. Represent it so that it matches the text describing "
           "the effect, consequence, or next step that it leads to.")
EFFECT_D = ("Causality is directional: a cause or preceding step happens first (at time t) "
            "and produces an effect or next step later (at time t+1). The following text is "
            "the EFFECT / next step. Represent it so that it matches the text describing the "
            "cause or preceding step that produced it.")

def cos1(a, b):  # cosine between two 1-D vectors
    return float(a @ b / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

print("Loading GritLM-7B ...", flush=True)
model = GritLM("GritLM/GritLM-7B", torch_dtype=torch.bfloat16, mode="embedding", device_map="cuda:0")

def emb(texts, instr, maxlen=512):
    return np.asarray(model.encode(texts, instruction=fmt(instr), max_length=maxlen,
                                   batch_size=8)).astype(np.float32)

def run_direction(causes, effects, tag):
    # encode every text in BOTH roles
    c_as_cause  = emb(causes,  CAUSE_Q)
    c_as_effect = emb(causes,  EFFECT_D)
    e_as_cause  = emb(effects, CAUSE_Q)
    e_as_effect = emb(effects, EFFECT_D)
    fwd, rev = [], []
    for i in range(len(causes)):
        f = cos1(c_as_cause[i],  e_as_effect[i])   # a->b  (correct)
        r = cos1(e_as_cause[i],  c_as_effect[i])   # b->a  (reversed)
        fwd.append(f); rev.append(r)
    fwd, rev = np.array(fwd), np.array(rev)
    correct = (fwd > rev)
    print(f"\n[{tag}]  N={len(causes)}")
    print(f"  directional acc (forward>reverse) = {correct.mean():.3f}")
    print(f"  mean forward={fwd.mean():+.4f}  mean reverse={rev.mean():+.4f}  "
          f"mean margin={ (fwd-rev).mean():+.4f}")
    return fwd, rev, correct

# -------- toy example --------
print("="*70); print("TOY directionality")
toy_c = ["Turn off the light switch."]
toy_e = ["The bulb goes dark and the room is no longer lit."]
f,r,c = run_direction(toy_c, toy_e, "toy")
print(f"  forward (switch->dark)={f[0]:+.4f}  reverse (dark->switch)={r[0]:+.4f}  "
      f"=> ranked {'CORRECT' if c[0] else 'WRONG'}")

# -------- real AEP data --------
print("\n"+"="*70); print("REAL aep_causal directionality")
DATA = "/mnt/localssd/internship-causal-embedding-merge-followup/data_6/aep_causal/test.jsonl"
rows = [json.loads(l) for l in open(DATA) if l.strip()]
N = 50
rng = np.random.default_rng(0)
sel = rng.choice(len(rows), size=N, replace=False)
causes  = [rows[i]["anchor"]   for i in sel]
effects = [rows[i]["positive"] for i in sel]
fwd, rev, correct = run_direction(causes, effects, f"aep N={N}")

# show first 8 per-pair
print("\n  pair | forward | reverse | margin | result | causal_type")
for k in range(min(8, N)):
    ct = rows[sel[k]]["causal_type"][:18]
    print(f"  {k:>4} | {fwd[k]:+.4f} | {rev[k]:+.4f} | {fwd[k]-rev[k]:+.4f} | "
          f"{'OK ' if correct[k] else 'BAD'}    | {ct}")
print("\nDONE")
