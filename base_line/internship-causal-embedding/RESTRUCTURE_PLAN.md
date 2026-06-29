# Restructure Plan: Unified Evaluation Framework

## The Problem (What We Have Today)

Every model variant has its own isolated eval script:

| Model | Eval script | Problem |
|---|---|---|
| BiEncoder | `src/evaluation/retrieval_eval.py` | Hardcoded to its retriever adapter |
| CDE (KL) | `my_work/kl/evaluate_cde.py` | Its own metric loop |
| CrossEncoder 2x | `my_work/kl/cross_encoder_2x/evaluate_ce2x.py` | Its own metric loop |
| LorentzBE | `lorentz_enc_workflow/evaluate.py` | Its own metric loop |
| Finetune eval | `finetune_eval/evaluate_finetune.py` | Its own metric loop |
| Hard-neg finetune | `finetune_eval/evaluate_finetune_hardneg.py` | Duplicate of above |

**Consequence**: To add a new negative type (e.g., "dense-mined negatives") today, you touch 6 files.
To compare CrossEncoder-2x vs BiEncoder on BM25 negatives, you write a new script from scratch.
MRR is computed 6 different ways. None of the comparisons are fair.

---

## The Goal (One Change → One File)

```
Want new negative type?   → add ONE file in evaluations/negatives/
Want new model?           → add ONE file in models/, implement Retriever protocol
Want new metric?          → edit ONE file: evaluations/metrics.py
Want to compare models?   → run ONE command with --model flags
```

Every model goes through the **identical** evaluation pipeline. Metrics are identical.
Results are reproducible by pinning a YAML config file.

---

## Step 0: Archive the Current Repo (Do This First)

Before touching a single file, snapshot the entire current repo into `prev/`.

```
internship-causal-embedding/
└── prev/                   ← entire current repo copied here, read-only reference
    ├── src/
    ├── my_work/
    ├── finetune_eval/
    ├── lorentz_enc_workflow/
    ├── evaluation_6/
    └── ... (everything as-is today)
```

**How**: `cp -r . prev/` from the repo root (excluding `prev/` itself and `data/`).
This is a local copy — not a git subtree, no gitignore complications.
Kept permanently. Any time you want to verify "what did the old CDE eval do?", read `prev/my_work/kl/evaluate_cde.py`.

---

## Target Folder Structure

```
internship-causal-embedding/
│
├── prev/                              ← FULL SNAPSHOT of current repo (read-only)
│
├── models/                            ← ONE folder per model variant
│   ├── base.py                        ← Retriever Protocol (the only contract eval knows)
│   ├── biencoder/
│   │   ├── model.py                   (moved from src/biencoder/model.py)
│   │   ├── config.py
│   │   ├── losses.py
│   │   ├── adapter.py                 (renamed from evaluation_adapter.py)
│   │   └── training/
│   │       ├── train.py
│   │       ├── data.py
│   │       └── validation.py
│   ├── cde/
│   │   ├── model.py                   (moved from my_work/kl/model.py)
│   │   ├── losses.py
│   │   ├── adapter.py                 ← NEW: wraps CDE as Retriever
│   │   └── training/
│   │       └── train.py
│   ├── cross_encoder_2x/
│   │   ├── model.py                   (moved from my_work/kl/cross_encoder_2x/model_ce2x.py)
│   │   ├── adapter.py                 ← NEW: wraps CE2x as Retriever
│   │   └── training/
│   │       └── train.py
│   ├── lorentz/
│   │   ├── model.py                   (moved from lorentz_enc_workflow/model.py)
│   │   ├── adapter.py                 ← NEW: wraps Lorentz as Retriever
│   │   └── training/
│   │       └── train.py
│   └── pretrained/
│       └── adapter.py                 (moved from src/evaluation/retrievers.py — HF + TF-IDF wrappers)
│
├── evaluations/                       ← the evaluation pipeline, fully model-agnostic
│   │
│   ├── negatives/                     ← ONE file per negative strategy
│   │   ├── base.py                    ← NegativeSampler Protocol
│   │   ├── random_corpus.py           (moved from src/followup_data/negatives.py)
│   │   ├── shuffle_positives.py       (moved from src/followup_data/negatives.py)
│   │   ├── bm25.py                    (moved from src/followup_data/negatives.py)
│   │   ├── dense_mined.py             ← example: add this alone for a new neg type
│   │   └── registry.py                ← name → class mapping ("bm25" → BM25Negatives)
│   │
│   ├── metrics.py                     ← SINGLE source of truth (promoted from src/evaluation/metrics.py)
│   │                                    MRR, Recall@K, Hit@K, AUC, P@1, mean/median rank
│   │
│   ├── runner.py                      ← THE CORE: EvalRunner(model, negatives, dataset) → metrics
│   │                                    Every model, every negative, every dataset goes through here
│   │
│   └── report.py                      ← JSON + CSV output, pretty-print table
│
├── configs/                           ← pin everything for reproducibility
│   ├── aep_causal_bm25.yaml
│   ├── aep_causal_random.yaml
│   ├── qrecc_bm25.yaml
│   └── ... (one file per experiment)
│
├── scripts/
│   └── run_eval.py                    ← unified CLI entry point
│
├── src/followup_data/                 ← UNCHANGED (dataset layer stays as-is)
│   ├── base.py
│   ├── registry.py
│   ├── io.py
│   ├── cli.py
│   └── loaders/
│
├── data/                              ← UNCHANGED
├── pyproject.toml                     ← updated entry points
└── CLAUDE.md                          ← updated
```

---

## The Retriever Protocol (The Glue)

Every model adapter must implement exactly this. Nothing else.

**File**: `models/base.py`

```python
from typing import Protocol
import numpy as np

class Retriever(Protocol):
    name: str                          # shown in results tables

    def encode_anchors(self, texts: list[str]) -> np.ndarray:
        """Returns (N, D) L2-normalized float32 array."""
        ...

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        """Returns (M, D) L2-normalized float32 array."""
        ...
```

The eval runner only calls these two methods. It doesn't know what model is inside.
BiEncoder, CDE, CrossEncoder-2x, Lorentz — all implement the same two methods.

---

## The NegativeSampler Protocol

Every negative strategy implements exactly this.

**File**: `evaluations/negatives/base.py`

```python
from typing import Protocol
from src.followup_data.base import PairExample, TripleExample
from collections.abc import Iterable, Iterator

class NegativeSampler(Protocol):
    name: str                          # "random", "bm25", "dense_mined", ...

    def __call__(
        self, pairs: Iterable[PairExample], k: int = 1
    ) -> Iterator[TripleExample]:
        ...
```

---

## The EvalRunner (models/negatives/metrics are all plugged in here)

**File**: `evaluations/runner.py`

```python
def run_eval(
    retriever: Retriever,           # any model that implements the protocol
    dataset_name: str,              # any registered dataset
    split: str,                     # "test", "val", "train"
    negative_strategy: str,         # "random", "bm25", "dense_mined", ...
    k_negatives: int = 4,
    pool_size: int = 1000,
    seed: int = 42,
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    ...
```

Internally it does:
1. Load dataset pairs via `followup_data` registry (unchanged)
2. Apply the selected negative sampler to get triples
3. Call `retriever.encode_anchors()` and `retriever.encode_candidates()`
4. Build similarity matrix
5. Call `metrics.compute_metrics()` — one and only one place
6. Return results dict

**The runner never imports any model class directly.** It only knows about the `Retriever` protocol.

---

## The Unified CLI

**File**: `scripts/run_eval.py`

```bash
# Evaluate CrossEncoder-2x on aep_causal with BM25 negatives
uv run python scripts/run_eval.py \
    --model cross_encoder_2x:my_work/kl/cross_encoder_2x/results/checkpoint_best.pt \
    --dataset aep_causal \
    --negatives bm25 \
    --split test

# Compare BiEncoder vs CDE on all datasets, all negative types
uv run python scripts/run_eval.py \
    --model biencoder:runs/aep_bge/biencoder_best.pt \
    --model cde:my_work/kl/results/cde_best.pt \
    --dataset aep_causal --dataset qrecc --dataset followupqg \
    --negatives random --negatives bm25 \
    --split test

# Run from a frozen config (fully reproducible)
uv run python scripts/run_eval.py --config configs/aep_causal_bm25.yaml
```

**Key behaviors:**
- `--model` is repeatable: runs all listed models through the same pipeline
- `--dataset` is repeatable: runs all listed datasets
- `--negatives` is repeatable: runs all listed negative strategies
- All combinations are evaluated in one run
- Results saved per (model × dataset × negatives) in a structured JSON

---

## Config File Format (Reproducibility)

**File**: `configs/aep_causal_bm25.yaml`

```yaml
# Frozen experiment config — run with:
#   uv run python scripts/run_eval.py --config configs/aep_causal_bm25.yaml

models:
  - type: cross_encoder_2x
    checkpoint: my_work/kl/cross_encoder_2x/results/checkpoint_best.pt
  - type: biencoder
    checkpoint: runs/aep_bge/biencoder_best.pt
  - type: pretrained
    hf_id: BAAI/bge-small-en-v1.5

datasets:
  - aep_causal
  - qrecc

splits:
  - test

negatives:
  - random
  - bm25

eval:
  pool_size: 1000
  k_negatives: 4
  seed: 42
  k_values: [1, 3, 5, 10]

output:
  dir: results/aep_causal_bm25/
  format: [json, csv]
```

---

## How to Add a New Negative Type (After Restructure)

**Example: add "dense-mined" negatives using a pretrained encoder**

1. Create `evaluations/negatives/dense_mined.py`
   - Implement the `NegativeSampler` protocol
   - One class, ~60 lines

2. Register it in `evaluations/negatives/registry.py`
   ```python
   from .dense_mined import DenseMinedNegatives
   REGISTRY["dense_mined"] = DenseMinedNegatives
   ```

3. Done. No other file changes.

To evaluate: `--negatives dense_mined` in the CLI or add `- dense_mined` to a config YAML.
All models automatically get evaluated with this new negative type. Metrics are identical.

---

## How to Add a New Model (After Restructure)

**Example: add a new model "HyperbolicBiEncoder"**

1. Create `models/hyperbolic/model.py` — the model architecture
2. Create `models/hyperbolic/adapter.py` — implement `encode_anchors` and `encode_candidates`
3. Register in `scripts/run_eval.py` model registry:
   ```python
   "hyperbolic": HyperbolicRetriever
   ```

4. Done. Use `--model hyperbolic:path/to/checkpoint.pt` in CLI.

---

## What Gets Deleted (Redundant Code)

After the restructure these files are dead — their logic lives in `runner.py` + `metrics.py`:

| File | Why deleted |
|---|---|
| `finetune_eval/evaluate_finetune.py` | Absorbed by `runner.py` |
| `finetune_eval/evaluate_finetune_hardneg.py` | Same |
| `my_work/kl/evaluate_cde.py` | Same |
| `my_work/kl/cross_encoder_2x/evaluate_ce2x.py` | Same |
| `lorentz_enc_workflow/evaluate.py` | Same |
| `evaluation_6/evaluate_inference.py` | Same (pool-size is now a param) |
| `evaluation_6/evaluate_full.py` | Same (embedding cache is now a param) |
| `evaluation_6/metrics_extra.py` | Merged into `evaluations/metrics.py` |
| `src/evaluation/retrieval_eval.py` | Replaced by `scripts/run_eval.py` |

**They are NOT deleted from `prev/`** — they live there permanently for reference.

---

## Migration Steps (Ordered)

### Phase 1 — Archive (No code changes)
1. `cp -r . prev/` (snapshot entire repo before anything changes)
2. Git commit: `chore: snapshot current repo into prev/ before restructure`

### Phase 2 — Lay the skeleton (No logic changes, just new files)
3. Create `models/base.py` — Retriever Protocol (copy from `src/evaluation/interfaces.py`)
4. Create `evaluations/negatives/base.py` — NegativeSampler Protocol
5. Create `evaluations/negatives/registry.py` — empty registry
6. Create `evaluations/metrics.py` — copy from `src/evaluation/metrics.py`, add AUC + P@1 from `evaluation_6/metrics_extra.py`

### Phase 3 — Migrate models (One model at a time)
7. BiEncoder: create `models/biencoder/adapter.py` (thin wrapper around existing `evaluation_adapter.py`)
8. CDE: create `models/cde/adapter.py`
9. CrossEncoder-2x: create `models/cross_encoder_2x/adapter.py`
10. Lorentz: create `models/lorentz/adapter.py`
11. Pretrained/TF-IDF: create `models/pretrained/adapter.py`

### Phase 4 — Migrate negatives (One sampler at a time)
12. Move `RandomCorpusNegatives` → `evaluations/negatives/random_corpus.py`
13. Move `ShufflePositiveNegatives` → `evaluations/negatives/shuffle_positives.py`
14. Move `BM25Negatives` → `evaluations/negatives/bm25.py`
15. Register all three in `evaluations/negatives/registry.py`

### Phase 5 — Build the runner
16. Write `evaluations/runner.py` — `run_eval()` function
17. Write `evaluations/report.py` — JSON + CSV output + table printer
18. Write `scripts/run_eval.py` — CLI with `--model`, `--dataset`, `--negatives`, `--config`

### Phase 6 — Config files
19. Write one `configs/` YAML per experiment (aep_causal × {random, bm25} to start)

### Phase 7 — Verify and clean
20. Run old eval (`prev/`) and new eval on same checkpoint → numbers must match exactly
21. Once verified, delete the now-redundant top-level eval scripts (keep `prev/` forever)
22. Update `CLAUDE.md` and `pyproject.toml` entry points

---

## What Stays Unchanged

- `src/followup_data/` — the entire dataset layer. Not touched.
- `data/` — datasets on disk. Not touched.
- Model training scripts — `models/*/training/train.py` just move location, logic unchanged.
- `pyproject.toml` dependencies — only entry points section updated.

---

## Results Directory Layout (After Restructure)

```
results/
└── <config-name>/
    ├── config.yaml                    ← copy of the config used (frozen)
    ├── summary.json                   ← all models × datasets × negatives
    ├── summary.csv                    ← same, tabular
    └── runs/
        ├── biencoder__aep_causal__bm25__test.json
        ├── cde__aep_causal__bm25__test.json
        ├── cross_encoder_2x__aep_causal__bm25__test.json
        └── cross_encoder_2x__qrecc__random__test.json
```

File naming: `{model}__{dataset}__{negatives}__{split}.json`

---

## Comparison Table (Before vs After)

| Action | Before | After |
|---|---|---|
| Evaluate CrossEncoder-2x on aep_causal | Write new script or modify `evaluate_ce2x.py` | `--model cross_encoder_2x:path --dataset aep_causal` |
| Add BM25 negatives to CDE eval | Edit `evaluate_cde.py` | `--negatives bm25` |
| Add dense-mined negatives for ALL models | Edit 5 separate scripts | Add 1 file in `evaluations/negatives/` |
| Compare 3 models head-to-head | Write aggregation script | `--model A --model B --model C` in one command |
| Reproduce an experiment | Guess flags from old script | `--config configs/experiment.yaml` |
| Verify metric definitions match | Read 5 metric implementations | Read `evaluations/metrics.py` (one file) |
| Check what old code did | Grep through scattered scripts | Read `prev/<original-path>` |
