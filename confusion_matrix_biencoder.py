"""Build a score-based confusion matrix from test_samples_milan_scored_biencoder.json.

Each cell (row=label1, col=label2) shows the biencoder similarity score
rather than a binary predicted label.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

DATA_PATH = Path("/mnt/localssd/test_samples_milan_scored_crossencoder.json")
OUT_PATH  = Path("/mnt/localssd/confusion_matrix_biencoder.png")

data = json.load(open(DATA_PATH))

# collect all tier labels in sorted order
tiers = sorted(set(r["label1"] for r in data) | set(r["label2"] for r in data))
n = len(tiers)
idx = {t: i for i, t in enumerate(tiers)}

matrix = np.full((n, n), np.nan)
for r in data:
    i = idx[r["label1"]]
    j = idx[r["label2"]]
    matrix[i, j] = r["score"]

fig, ax = plt.subplots(figsize=(8, 7))

# use a diverging colormap centred at 0.5
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

# annotate each cell with the score
for i in range(n):
    for j in range(n):
        val = matrix[i, j]
        if not np.isnan(val):
            text_color = "black" if 0.25 < val < 0.75 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tiers, fontsize=11)
ax.set_yticklabels(tiers, fontsize=11)
ax.set_xlabel("label2 (text2 tier)", fontsize=12)
ax.set_ylabel("label1 (text1 tier)", fontsize=12)
ax.set_title("BiEncoder score confusion matrix\n(cell = similarity score, not binary pred)", fontsize=13)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Similarity score", fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")

# also print the matrix to stdout
print("\nScore matrix:")
header = "      " + "  ".join(f"{t:>7}" for t in tiers)
print(header)
for i, t in enumerate(tiers):
    row_str = "  ".join(f"{matrix[i,j]:7.4f}" if not np.isnan(matrix[i,j]) else "    nan"
                        for j in range(n))
    print(f"{t}  {row_str}")
