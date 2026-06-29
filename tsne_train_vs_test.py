"""
Three-panel visualization (t-SNE, UMAP, PCA) comparing embedding spaces:

Train  : 500 random samples from directional_train.jsonl → text_1, text_2 (individual)
Milan  : 30 samples from test_samples_milan.json         → text1, text2 (individual)
AJO    : 30 samples from ajo_orchestrated_workflows_flat.json → t1, t2, t3 (individual)

Each text field = one embedding point (no concatenation).
Model: Qwen/Qwen3-Embedding-0.6B
"""

import json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = "/mnt/localssd/automation/internship-causal-embedding/data/aep_causal_classification_34/directional_train.jsonl"
MILAN_PATH = "/mnt/localssd/test_samples_milan.json"
AJO_PATH   = "/mnt/localssd/ajo_orchestrated_workflows_flat.json"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TRAIN_N    = 500
SEED       = 42
OUT_PATH   = "/mnt/localssd/tsne_umap_pca_train_vs_test.png"
BATCH_SIZE = 32
MAX_LEN    = 512

random.seed(SEED)
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(TRAIN_PATH) as f:
    all_train = [json.loads(l) for l in f]
train_sample = random.sample(all_train, TRAIN_N)

milan = json.load(open(MILAN_PATH))
ajo   = json.load(open(AJO_PATH))

train_texts = [r["text_1"] for r in train_sample] + [r["text_2"] for r in train_sample]
milan_texts = [r["text1"] for r in milan] + [r["text2"] for r in milan]
ajo_texts   = [r["t1"] for r in ajo] + [r["t2"] for r in ajo] + [r["t3"] for r in ajo]

n_train = len(train_texts)   # 2 * TRAIN_N
n_milan = len(milan_texts)   # 2 * len(milan)
n_ajo   = len(ajo_texts)     # 3 * len(ajo)
print(f"Train: {n_train}  Milan: {n_milan}  AJO: {n_ajo}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

# ── Embed ─────────────────────────────────────────────────────────────────────
def last_token_pool(hidden, attn_mask):
    seq_len = attn_mask.sum(dim=1) - 1
    idx = seq_len.unsqueeze(-1).expand(-1, hidden.size(-1)).unsqueeze(1)
    return hidden.gather(1, idx).squeeze(1)

@torch.no_grad()
def embed_texts(texts):
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)

print("\nEmbedding train...")
train_emb = embed_texts(train_texts)
print("Embedding Milan...")
milan_emb = embed_texts(milan_texts)
print("Embedding AJO...")
ajo_emb   = embed_texts(ajo_texts)

all_emb = np.vstack([train_emb, milan_emb, ajo_emb])
print(f"\nTotal embedding matrix: {all_emb.shape}")

# ── Projections ───────────────────────────────────────────────────────────────
perplexity = min(40, len(all_emb) // 5)

print("Running PCA...")
pca_proj = PCA(n_components=2, random_state=SEED).fit_transform(all_emb)

print("Running t-SNE...")
tsne_proj = TSNE(
    n_components=2, perplexity=perplexity, max_iter=2000,
    random_state=SEED, metric="cosine", init="pca", learning_rate="auto",
).fit_transform(all_emb)

print("Running UMAP...")
umap_proj = umap.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.1,
    metric="cosine", random_state=SEED,
).fit_transform(all_emb)

# ── Split back ────────────────────────────────────────────────────────────────
def split(proj):
    return (proj[:n_train],
            proj[n_train : n_train + n_milan],
            proj[n_train + n_milan :])

# ── Plot ──────────────────────────────────────────────────────────────────────
methods = [
    ("PCA",    pca_proj),
    ("t-SNE",  tsne_proj),
    ("UMAP",   umap_proj),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle(
    f"Embedding Space: Train vs Test  |  Model: {MODEL_NAME}\n"
    f"Each point = one individual text field  |  Train={n_train}, Milan={n_milan}, AJO={n_ajo}",
    fontsize=13, y=1.02,
)

for ax, (title, proj) in zip(axes, methods):
    tr, mi, aj = split(proj)

    ax.scatter(tr[:, 0], tr[:, 1],
               s=12, alpha=0.35, color="#4C72B0",
               label=f"Train (n={n_train})", zorder=1)
    ax.scatter(mi[:, 0], mi[:, 1],
               s=150, alpha=0.92, color="#DD8452", marker="*",
               label=f"Test: Milan (n={n_milan})", zorder=3,
               edgecolors="black", linewidths=0.5)
    ax.scatter(aj[:, 0], aj[:, 1],
               s=120, alpha=0.92, color="#55A868", marker="D",
               label=f"Test: AJO (n={n_ajo})", zorder=3,
               edgecolors="black", linewidths=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("dim 1", fontsize=10)
    ax.set_ylabel("dim 2", fontsize=10)
    ax.legend(fontsize=9, markerscale=1.2)
    ax.grid(True, alpha=0.22)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {OUT_PATH}")
