
```python
# data/load_causenet.py
import json
from datasets import Dataset

def load_causenet(path="causenet-precision.jsonl"):
    pairs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            cause = row["causal_relation"]["cause"]["concept"]
            effect = row["causal_relation"]["effect"]["concept"]
            sources = row.get("sources", [])
            # Reconstruct a natural sentence from the sources
            for src in sources[:3]:
                sentence = src["payload"]["sentence"]
                pairs.append({
                    "cause_text": cause,
                    "effect_text": effect,
                    "context": sentence,
                })
    return Dataset.from_list(pairs)
```

### 1.2 Hard negative strategies (the make-or-break part)

Each positive pair (A, B) needs multiple flavors of negatives. Build them offline once, store in a separate dataset.

#### a) Reverse-direction negatives

Trivial — just swap the pair. Mark as negative. These teach the model basic asymmetry.

```python
def reverse_negative(pair):
    return {"cause_text": pair["effect_text"], 
            "effect_text": pair["cause_text"], 
            "label": 0}
```

#### b) Confounder negatives (from CauseNet graph)

Two effects of the same cause are correlated but not causally linked. Mine these from the graph:

```python
# A -> X and B -> X means A and B share a common effect X (a "fork")
# Yet A and B are not in a direct causal relation
def mine_confounder_negatives(causenet_df):
    cause_to_effects = causenet_df.groupby("cause_text")["effect_text"].apply(set)
    confounders = []
    for cause, effects in cause_to_effects.items():
        effects = list(effects)
        for i in range(len(effects)):
            for j in range(i+1, len(effects)):
                # check that effects[i] and effects[j] are not themselves a causal pair
                if not is_causal_pair(effects[i], effects[j], causenet_df):
                    confounders.append({
                        "cause_text": effects[i],
                        "effect_text": effects[j],
                        "label": 0,
                        "type": "confounder",
                    })
    return confounders
```

This is the **hardest class** for any model — and exactly where CDE should outperform baselines.

#### c) Semantic near-miss negatives

For each cause A, find a semantically similar A' that is *not* causally linked to B. Use any off-the-shelf sentence embedder (e.g. `all-MiniLM-L6-v2`) to find nearest neighbors of A, then filter out true causal pairs.

```python
from sentence_transformers import SentenceTransformer
import faiss

def mine_semantic_near_miss(causenet_df, k=20):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    causes = causenet_df["cause_text"].unique().tolist()
    embs = encoder.encode(causes, show_progress_bar=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    
    near_miss = []
    causal_set = set(zip(causenet_df.cause_text, causenet_df.effect_text))
    for i, cause in enumerate(causes):
        _, neighbors = index.search(embs[i:i+1], k+1)
        for j in neighbors[0][1:]:  # skip self
            near_cause = causes[j]
            # find a true effect of `cause` that is NOT an effect of `near_cause`
            true_effects = causenet_df[causenet_df.cause_text == cause].effect_text
            for effect in true_effects:
                if (near_cause, effect) not in causal_set:
                    near_miss.append({
                        "cause_text": near_cause,
                        "effect_text": effect,
                        "label": 0,
                        "type": "semantic_near_miss",
                    })
                    break
    return near_miss
```

#### d) Counterfactual edits (LLM-generated)

The most expensive but highest-quality negatives. Use GPT-4 or Claude to make minimal edits to a cause that break the causal link.

```python
# data/augment_counterfactual.py
PROMPT = """You are generating training data for a causal embedding model.
Given a true causal pair, produce a MINIMAL EDIT of the cause that no longer 
causes the effect.

Original cause: {cause}
Effect: {effect}

Minimally edit the cause so it no longer causes the effect. Keep the syntactic 
structure and most words identical. Output only the edited cause."""

def generate_counterfactuals(pairs, llm_client, n_per_pair=2):
    counterfactuals = []
    for pair in pairs:
        for _ in range(n_per_pair):
            edited = llm_client.generate(PROMPT.format(**pair))
            counterfactuals.append({
                "cause_text": edited,
                "effect_text": pair["effect_text"],
                "label": 0,
                "type": "counterfactual",
            })
    return counterfactuals
```

Budget ~$200–500 in LLM API costs to generate ~50K counterfactual negatives. Worth every cent — these are the strongest discriminators.

### 1.3 Batch composition

Each training batch should contain a *mix* of negative types. Recommended ratio per positive pair:

```
1 positive
1 reverse negative
1 confounder negative
1 semantic near-miss negative
1 counterfactual negative
4 random in-batch negatives (from other pairs)
```

Total batch size = 256 means 32 positives × 8 candidates each.

### 1.4 Validation/test splits

**Critical**: split by *cause concept*, not by row. Otherwise leakage makes evaluation meaningless.

```python
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=0.1, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["cause_text"]))
```

---

## Phase 2 — Model Architecture

### 2.1 Core model

```python
# model/cde.py
import torch
import torch.nn as nn
from transformers import AutoModel

class CausalDensityEmbedding(nn.Module):
    def __init__(self, backbone="roberta-base", proj_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.proj_dim = proj_dim
        
        # Three projection heads (parallel, share input)
        self.mu_head = nn.Linear(hidden, proj_dim)
        self.log_sigma_head = nn.Linear(hidden, proj_dim)
        self.delta_head = nn.Linear(hidden, proj_dim)
        
        # Learned global spreading rate (scalar)
        self.gamma = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1 initially
        
        # Initialize delta near zero (no drift at start)
        nn.init.normal_(self.delta_head.weight, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        
        # Initialize log_sigma near zero (sigma ≈ 1)
        nn.init.normal_(self.log_sigma_head.weight, std=1e-3)
        nn.init.zeros_(self.log_sigma_head.bias)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, 
                                attention_mask=attention_mask)
        # Mean-pool over real tokens (more stable than CLS for short texts)
        mask = attention_mask.unsqueeze(-1).float()
        h = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
        mu = self.mu_head(h)
        log_sigma = torch.clamp(self.log_sigma_head(h), min=-5.0, max=5.0)
        delta = self.delta_head(h)
        return mu, log_sigma, delta
    
    def transport(self, mu, log_sigma, delta):
        mu_T = mu + delta
        # variance inflation: sigma_T^2 = sigma^2 * exp(gamma)
        # so log_sigma_T = log_sigma + gamma/2
        log_sigma_T = log_sigma + 0.5 * self.gamma
        return mu_T, log_sigma_T
    
    def kl_gaussian(self, mu_p, log_sigma_p, mu_q, log_sigma_q):
        """KL(N(mu_p, sigma_p^2) || N(mu_q, sigma_q^2)) summed over dims."""
        sigma_p_sq = torch.exp(2 * log_sigma_p)
        sigma_q_sq = torch.exp(2 * log_sigma_q)
        
        log_ratio = log_sigma_q - log_sigma_p
        mean_term = (sigma_p_sq + (mu_p - mu_q)**2) / (2 * sigma_q_sq)
        
        kl_per_dim = log_ratio + mean_term - 0.5
        return kl_per_dim.sum(dim=-1)
    
    def score(self, A_ids, A_mask, B_ids, B_mask):
        """Score(A -> B) = -KL(N_B || T(N_A))"""
        mu_A, log_sigma_A, delta_A = self.encode(A_ids, A_mask)
        mu_B, log_sigma_B, _ = self.encode(B_ids, B_mask)
        
        mu_TA, log_sigma_TA = self.transport(mu_A, log_sigma_A, delta_A)
        kl = self.kl_gaussian(mu_B, log_sigma_B, mu_TA, log_sigma_TA)
        return -kl
    
    def score_with_features(self, A_ids, A_mask, B_ids, B_mask):
        """Same as score but returns intermediate features for auxiliary losses."""
        mu_A, log_sigma_A, delta_A = self.encode(A_ids, A_mask)
        mu_B, log_sigma_B, delta_B = self.encode(B_ids, B_mask)
        
        mu_TA, log_sigma_TA = self.transport(mu_A, log_sigma_A, delta_A)
        kl = self.kl_gaussian(mu_B, log_sigma_B, mu_TA, log_sigma_TA)
        
        return {
            "score": -kl,
            "mu_A": mu_A, "log_sigma_A": log_sigma_A, "delta_A": delta_A,
            "mu_B": mu_B, "log_sigma_B": log_sigma_B, "delta_B": delta_B,
        }
```

### 2.2 Numerical stability notes

- **Clamp `log_sigma`** to [-5, 5]. Without this, σ can blow up to 1e10 and KL goes NaN.
- **Initialize δ near zero**. Otherwise the model starts with strong random drifts and contrastive loss takes ages to recover.
- **Use bfloat16 for backbone, float32 for KL computation**. KL is sensitive to precision.
- **Add `eps=1e-6` to denominators** if you see NaNs in early training.

---

## Phase 3 — Loss Functions

```python
# model/losses.py
import torch
import torch.nn.functional as F

def infonce_loss(score_pos, score_neg, temperature=0.1):
    """
    score_pos: (B,) — positive scores
    score_neg: (B, K) — K negative scores per positive
    """
    pos = score_pos.unsqueeze(1) / temperature
    neg = score_neg / temperature
    logits = torch.cat([pos, neg], dim=1)  # (B, K+1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def entropy_ordering_loss(log_sigma_A, log_sigma_B, margin=0.5):
    """Encourage H(B) > H(A) + margin. Diag Gaussian entropy ∝ sum(log_sigma)."""
    H_A = log_sigma_A.sum(dim=-1)
    H_B = log_sigma_B.sum(dim=-1)
    return F.relu(H_A + margin - H_B).mean()


def antisymmetry_loss(score_AB, score_BA, margin=2.0):
    """Penalize when reverse score is close to or higher than forward score."""
    return F.relu(score_BA - score_AB + margin).mean()


def compositional_drift_loss(model, A_ids, A_mask, B_ids, B_mask, C_ids, C_mask):
    """
    For a chain A → B → C, the composed drift δ_A + δ_B should explain A → C.
    Compute KL between N_C and a 2-hop transported N_A.
    """
    mu_A, log_sigma_A, delta_A = model.encode(A_ids, A_mask)
    _, _, delta_B = model.encode(B_ids, B_mask)
    mu_C, log_sigma_C, _ = model.encode(C_ids, C_mask)
    
    # 2-hop transport: shift by composed drift, inflate variance twice
    mu_2hop = mu_A + delta_A + delta_B
    log_sigma_2hop = log_sigma_A + model.gamma  # two applications of γ/2
    
    kl = model.kl_gaussian(mu_C, log_sigma_C, mu_2hop, log_sigma_2hop)
    return kl.mean()


def combined_loss(outputs, lambdas):
    """Weighted sum of all losses. `outputs` is a dict built by the training loop."""
    L = lambdas["infonce"] * outputs["L_infonce"]
    if "L_entropy" in outputs:
        L = L + lambdas["entropy"] * outputs["L_entropy"]
    if "L_antisym" in outputs:
        L = L + lambdas["antisym"] * outputs["L_antisym"]
    if "L_comp" in outputs:
        L = L + lambdas["comp"] * outputs["L_comp"]
    return L
```

### 3.1 Loss weight recommendations

Start very conservative; ramp up auxiliary losses only after InfoNCE converges.

| Phase | λ_infonce | λ_entropy | λ_antisym | λ_comp |
|-------|-----------|-----------|-----------|--------|
| A (warmup, 10K steps) | 1.0 | 0 | 0 | 0 |
| B (main, 50K steps) | 1.0 | 0.1 | 0.5 | 0 |
| C (compositional, 30K steps) | 1.0 | 0.1 | 0.5 | 0.2 |

---

## Phase 4 — Training Loop

```python
# training/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from model.cde import CausalDensityEmbedding
from model.losses import (infonce_loss, entropy_ordering_loss, 
                          antisymmetry_loss, compositional_drift_loss,
                          combined_loss)

def train_step(model, batch, lambdas, phase):
    """
    batch contains:
      A_ids, A_mask           — cause tokens
      B_ids, B_mask           — effect tokens  
      neg_ids, neg_mask       — (B, K, L) flattened negatives
      C_ids, C_mask           — optional 3rd hop for compositionality
    """
    # Forward score on positive pair
    f = model.score_with_features(batch["A_ids"], batch["A_mask"],
                                   batch["B_ids"], batch["B_mask"])
    score_pos = f["score"]
    
    # Negative scores: reshape (B, K, L) -> (B*K, L), score, reshape back
    B, K = batch["neg_ids"].shape[:2]
    neg_ids = batch["neg_ids"].view(B*K, -1)
    neg_mask = batch["neg_mask"].view(B*K, -1)
    # A is repeated K times for each negative
    A_ids_rep = batch["A_ids"].unsqueeze(1).expand(-1, K, -1).reshape(B*K, -1)
    A_mask_rep = batch["A_mask"].unsqueeze(1).expand(-1, K, -1).reshape(B*K, -1)
    
    score_neg_flat = model.score(A_ids_rep, A_mask_rep, neg_ids, neg_mask)
    score_neg = score_neg_flat.view(B, K)
    
    outputs = {"L_infonce": infonce_loss(score_pos, score_neg, temperature=0.1)}
    
    if phase >= "B":
        outputs["L_entropy"] = entropy_ordering_loss(f["log_sigma_A"], f["log_sigma_B"])
        # Reverse direction
        score_BA = model.score(batch["B_ids"], batch["B_mask"],
                               batch["A_ids"], batch["A_mask"])
        outputs["L_antisym"] = antisymmetry_loss(score_pos, score_BA)
    
    if phase == "C" and "C_ids" in batch:
        outputs["L_comp"] = compositional_drift_loss(
            model, batch["A_ids"], batch["A_mask"],
                   batch["B_ids"], batch["B_mask"],
                   batch["C_ids"], batch["C_mask"])
    
    return combined_loss(outputs, lambdas), outputs


def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    model = CausalDensityEmbedding(config.backbone, config.proj_dim).cuda()
    
    train_loader = DataLoader(...)  # your loader with hard negatives
    
    # Two parameter groups — backbone slower, heads faster
    head_params = list(model.mu_head.parameters()) + \
                  list(model.log_sigma_head.parameters()) + \
                  list(model.delta_head.parameters()) + [model.gamma]
    backbone_params = list(model.backbone.parameters())
    
    optimizer = AdamW([
        {"params": backbone_params, "lr": 2e-5},
        {"params": head_params, "lr": 1e-4},
    ], weight_decay=0.01)
    
    total_steps = config.total_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    scaler = torch.cuda.amp.GradScaler()  # mixed precision
    
    for step, batch in enumerate(train_loader):
        if step < 10_000: phase = "A"
        elif step < 60_000: phase = "B"
        else: phase = "C"
        
        lambdas = config.lambdas[phase]
        batch = {k: v.cuda() for k, v in batch.items()}
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, _ = train_step(model, batch, lambdas, phase)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        if step % 100 == 0:
            wandb.log({"loss": loss.item(), "phase": phase, "step": step})
        if step % 5000 == 0:
            torch.save(model.state_dict(), f"checkpoints/step_{step}.pt")
```

### 4.1 Critical training hyperparameters

```yaml
# training/config.yaml
backbone: roberta-base       # start here; upgrade to roberta-large for SOTA
proj_dim: 256                # 256 for base, 512 for large
batch_size: 32               # 32 positives, ×8 candidates = 256 effective
total_steps: 100000          # ~3 epochs over 500K pairs
warmup_steps: 1000
backbone_lr: 2e-5
head_lr: 1e-4
weight_decay: 0.01
grad_clip: 1.0
temperature: 0.1             # InfoNCE temperature
max_seq_len: 128             # CauseNet sentences are short
mixed_precision: bf16
```

### 4.2 What to log

- `loss/total`, `loss/infonce`, `loss/entropy`, `loss/antisym`, `loss/comp`
- `metric/score_pos_mean`, `metric/score_neg_mean`, `metric/gap` (pos − neg)
- `metric/asymmetry` = mean(score(A→B) − score(B→A)) for positives
- `metric/sigma_mean`, `metric/gamma` (sanity check on entropy)
- `metric/delta_norm` (drift magnitude — should grow then plateau)

**Early warning signs**:
- δ norm collapsing to 0 → no causal direction being learned, raise λ_antisym
- σ blowing up → entropy loss too weak or λ_entropy too low; clamp tighter
- Asymmetry stuck near 0 → antisymmetry penalty not biting, raise margin
- InfoNCE oscillating wildly → temperature too low, raise to 0.2

---

## Phase 5 — Evaluation

Don't only report accuracy on your training distribution. SOTA claims require evaluation across multiple held-out benchmarks.

### 5.1 Required benchmarks

| Benchmark | What it tests | Metric |
|-----------|---------------|--------|
| **Corr2Cause** (ICLR 2024) | Distinguishing correlation from causation in natural language | F1 |
| **CausalBench** (NeurIPS 2024) | Directional causal prediction | Accuracy, F1 |
| **CausalFlip** | Hard semantic distractors with opposite causal answers | Accuracy |
| **CauseNet test split** | Held-out causal pairs | Hit@k, MRR |
| **CLadder** (Pearl's rungs) | Counterfactual reasoning | Accuracy (zero-shot) |

### 5.2 Evaluation code skeleton

```python
# eval/corr2cause.py
def evaluate_corr2cause(model, tokenizer, test_set):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for row in test_set:
            A = tokenizer(row["premise"], return_tensors="pt", 
                         padding=True, truncation=True).to("cuda")
            B = tokenizer(row["hypothesis"], return_tensors="pt",
                         padding=True, truncation=True).to("cuda")
            score = model.score(A.input_ids, A.attention_mask,
                                B.input_ids, B.attention_mask).item()
            preds.append(1 if score > THRESHOLD else 0)
            labels.append(row["label"])
    return f1_score(labels, preds)
```

### 5.3 Baselines to beat

You need at least these comparisons to claim SOTA:

1. **Plain RoBERTa + cosine** — symmetric baseline, the floor
2. **Two-BERT (Sharp 2016)** — asymmetric by encoder split
3. **CausalBERT (2021)** — modern BERT with causal pretraining
4. **CausE (2023)** — TransE-style causal KG embedding
5. **GPT-4 zero-shot** — what big LLMs can do without specialized training

Run each on the same benchmarks with the same eval protocol. If CDE beats all of them, you have a paper.

### 5.4 Required ablations

| Ablation | What it tells you |
|----------|-------------------|
| Remove δ head (set δ=0) | Is the drift actually doing work? |
| Remove σ head (set σ=1) | Does uncertainty modeling matter? |
| Replace KL with symmetric JS | Is asymmetric divergence essential? |
| Remove counterfactual negatives | How much do they help? |
| Single-loss (InfoNCE only) | Are aux losses necessary? |
| Smaller projection dim | Returns to dim? |

---

## Phase 6 — Pushing to SOTA

After Phase 5 you'll have a solid model. To actually push the field forward, layer on these:

### 6.1 Scale up

- Backbone: RoBERTa-large or DeBERTa-v3-large (DeBERTa often wins by 1–3 points on NLI-style tasks)
- Projection dim: 512 or 768
- Effective batch size: 1024+ via gradient accumulation
- Training data: combine CauseNet + WikiCausality + ATOMIC

### 6.2 Better negatives

- Use a strong LLM (Claude or GPT-4) to generate ~200K high-quality counterfactual edits
- Mine cross-domain confounders (medical + legal + scientific)
- Iterative hard negative mining: after Phase B, find pairs the model gets wrong and oversample them in Phase C

### 6.3 Multi-task pretraining

Before causal training, do a joint pretraining stage on:
- Entailment (SNLI, MNLI) — teaches directional reasoning
- Temporal ordering (CaTeRS, MATRES) — teaches "before/after"  
- Then fine-tune on CauseNet with the CDE objective

### 6.4 Refinements that often help

- **Rényi α-divergence** instead of KL — α=0.5 (closer to JS) gives smoother gradients but keeps asymmetry. Worth trying as an ablation.
- **Per-dimension γ** — replace scalar γ with vector γ ∈ ℝ^d, letting different latent dimensions have different spreading rates. ~0.5 point gain typically.
- **Cycle consistency** — for chains A→B→C, also enforce that score(A→C) ≥ min(score(A→B), score(B→C)) up to a margin.
- **Cross-encoder distillation** — train a large cross-encoder (BERT seeing [A; B] together) as a teacher; distill into your bi-encoder CDE. This is how SBERT got its biggest gains.

### 6.5 The full SOTA recipe

```
1. Pretrain DeBERTa-v3-large on entailment + temporal NLI (10 epochs)
2. Generate 200K counterfactual negatives with Claude
3. Mine 500K confounder + semantic near-miss negatives  
4. Train CDE with all four losses, phased schedule, 200K steps
5. Iteratively mine hard negatives every 50K steps
6. Distill from a cross-encoder teacher
7. Final fine-tune on each target benchmark's train split
```

Estimated cost: ~$3K–8K in compute + ~$500 in LLM API. Should yield 5–10 point gains over Two-BERT on Corr2Cause and CausalFlip.

---

## Phase 7 — Common Pitfalls

These will eat weeks if you don't watch for them.

### 7.1 KL explodes / goes NaN

**Symptom**: loss is fine, then sudden NaN.  
**Fix**: clamp `log_sigma` tighter (try [-3, 3]). Add small ε to all denominators. Check for empty `attention_mask` rows.

### 7.2 σ collapses to a constant

**Symptom**: σ values become identical across all inputs.  
**Fix**: λ_entropy too low. Raise it to 0.3. Check that entropy_ordering_loss is actually decreasing.

### 7.3 δ collapses to zero

**Symptom**: model becomes effectively symmetric, score(A→B) ≈ score(B→A).  
**Fix**: λ_antisym too low. Raise margin in antisymmetry_loss. Add explicit ‖δ‖ ≥ ε regularizer.

### 7.4 Score values drift wildly during training

**Symptom**: positive scores around -1000 sometimes, -10 other times.  
**Fix**: normalize μ by L2 norm before computing KL, or add a `BatchNorm1d` after each projection head.

### 7.5 Hard negatives don't help (or hurt)

**Symptom**: ablation shows hard negatives degrade performance.  
**Fix**: usually your "hard negatives" are actually noisy positives that the LLM mislabeled. Manually inspect 100 random counterfactuals — if >10% look causal, fix the generation prompt.

### 7.6 Eval-train distribution mismatch

**Symptom**: train loss great, eval terrible.  
**Fix**: you almost certainly leaked cause concepts across splits. Re-split by concept, not by row.

---

