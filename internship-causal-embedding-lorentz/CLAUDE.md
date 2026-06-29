# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rules (always follow)

These are non-negotiable. Apply on every task in this repo.

1. **Never commit to `main`.** For any code or data change, create (or check out) a working branch and commit there. If no working branch exists yet, create one named for the task (e.g. `feat/<short-desc>`, `fix/<short-desc>`, `chore/<short-desc>`).
2. **Commit every change.** After each code or data change is made, stage and commit it on the working branch. Do not leave changes uncommitted across turns. One logical change per commit; write a short, descriptive message.
3. **Push after every commit.** Run `git push -u origin <branch>` on the first push, `git push` after. Don't wait to be asked. If push fails (permissions, protected branch, etc.), surface the error and ask how to proceed — never silently skip.
4. **"Clean the code" = branch first, then clean.** When the user asks to clean / refactor / reorganize code, first create a new branch from the current state as a snapshot for recovery (e.g. `backup/<date>-<topic>`) and push/keep it intact, *then* perform the cleanup on a separate working branch. Never clean in place without that backup branch existing.
5. **Use a virtual environment for everything.** Never install packages into the system / user Python. All installs go through the project's venv via `uv` (which manages `.venv/` automatically). If a venv doesn't exist yet, create it with `uv sync` before any install. Run all Python commands via `uv run …` (or after activating `.venv`), never bare `python` / `pip install`.

## Project

`follow-up-causal-embeddings` — datasets and middleware for the **follow-up causal embedding** research project. Given an anchor text *A* and a candidate continuation *B*, the goal is a scoring function `s(A, B) → R` that is high for true follow-ups and low for unrelated text. The library normalizes every dataset into `PairExample(anchor, positive)` and converts to `TripleExample(anchor, positive, negatives)` via pluggable samplers.

Python ≥3.11, managed with `uv` (`uv.lock` committed).

## Common commands

Install (extras are intentionally split so the data-only path stays light). All installs must go into the project venv — `uv` handles this automatically:

```bash
uv sync                              # creates .venv and installs core deps
uv pip install '.[embeddings]'       # sentence-transformers for examples/score_pairs.py
uv pip install '.[training]'         # torch + transformers for train-biencoder
uv pip install '.[evaluation]'       # for evaluate-retrieval
uv pip install '.[llm-baseline]'     # for evaluate-llm-baseline (needs AWS Bedrock env)
uv pip install '.[hard-negatives]'   # rank-bm25 (BM25 sampler is TODO)
uv pip install '.[dev]'              # pytest, ruff
```

CLI entry points (defined in `pyproject.toml [project.scripts]`):

```bash
uv run followup-data list                                # registered datasets
uv run followup-data download <name> [<name> ...]        # idempotent
uv run followup-data head <name> -n 3
uv run followup-data triples <name> -k 4 -n 2
uv run followup-data export <name> -o data/<name>.jsonl

uv run train-biencoder --dataset <name> --backbone <bge-small|arctic-embed-l> ...
uv run evaluate-retrieval --dataset <name> --split <s> --candidate-pool <same-split|all-splits> --retriever <spec>
uv run evaluate-retrieval --followup-file data/aep_followup/followup_questions_eval.json --retriever <spec> [--retriever <spec> ...]
uv run evaluate-llm-baseline --followup-file data/aep_followup/followup_questions_eval.json --top-k 10
```

Sanity-check scripts:

```bash
uv run python scripts/inspect_dataset.py <name>          # basic stats + 3 sample triples
uv run python scripts/test_all_loaders.py                # smoke-test every loader
uv run python examples/score_pairs.py --dataset clariq -n 200 -k 4
```

Smoke-testing without a full run: `--max-train-samples` / `--max-val-samples` for `train-biencoder`, `--max-queries` for `evaluate-retrieval` and `evaluate-llm-baseline`.

There is no test framework wired up beyond `scripts/test_all_loaders.py`; `pytest` is listed as a dev extra but no `tests/` directory exists yet.

## Architecture

Three sibling packages under `src/`, each independently installable into the wheel (`pyproject.toml` `[tool.hatch.build.targets.wheel]`). The boundary between them is load-bearing — keep changes within the layer that owns the concept.

### `followup_data/` — dataset layer

- `base.py` defines `PairExample` (anchor, positive, dataset, context, metadata) and `BaseDataset` (abstract: `is_downloaded`, `download`, `_iter_examples`). Every dataset normalizes to this shape.
- `registry.py` — `@register` decorator + module-level `_REGISTRY` dict. Loaders **self-register on import**, so `loaders/__init__.py` must import each loader module for it to appear in `list()`.
- `negatives.py` — `RandomCorpusNegatives` (materialize then sample), `ShufflePositiveNegatives` (streaming buffer). Produces `TripleExample`.
- `io.py` — `download_file`, `extract`, `git_clone`, `hf_snapshot`. Use these in new loaders rather than re-rolling HTTP/extract logic.
- `cli.py` — Typer app exposed as `followup-data`.
- `loaders/` — one module per dataset; 12 implemented (auto-download), plus `_manual.py` stubs that read a pre-converted `<root>/<name>/converted.jsonl`.

**Adding a dataset:** create `loaders/<name>.py`, subclass `BaseDataset` with class-level `name/description/homepage/citation/license/splits`, implement the three abstract methods, decorate with `@register`, then import it from `loaders/__init__.py`. Without that final import the loader is invisible.

### `biencoder/` — model + training

Two-tower (untied anchor/positive encoders), InfoNCE with in-batch negatives and learnable temperature. **Dataset-agnostic** — reads any registered `BaseDataset` via `PairExample.anchor` / `.positive`.

- `model.py` — `BiEncoder`, `MeanPooling`, `CLSPooling`.
- `losses.py` — `InfoNCELoss`.
- `config.py` — `MODEL_CONFIGS`: `bge-small` (`BAAI/bge-small-en-v1.5`, mean pool) and `arctic-embed-l` (`Snowflake/snowflake-arctic-embed-l-v2.0`, cls pool, `query: ` prefix). Adding a backbone = adding an entry here.
- `training/` — `data.py` (`PairDataset`), `validation.py` (in-loop MRR / Recall@K), `train.py` (loop + `train-biencoder` CLI).
- `evaluation_adapter.py` — `BiEncoderRetriever` conforms to `evaluation.Retriever` so trained checkpoints plug into the eval harness via `biencoder:<path>` spec.

Per-epoch validation tracks best val MRR. Outputs to `--out-dir`: `biencoder_best.pt`, `biencoder_latest.pt`, `config.json`, `training_history.json`.

### `evaluation/` — model-agnostic retrieval harness

- `interfaces.py` — `Retriever` Protocol. Anything that encodes anchors and candidates into a shared space plugs in.
- `retrievers.py` — `TfidfRetriever`, `PretrainedRetriever` (HF encoder, default BGE-small).
- `metrics.py` — MRR, Recall@K, Hit@K, mean/median rank, multi-gold aware (Recall@K ≠ Hit@K when `n_gold_per_query > 1`). `random` analytical baseline is always added to output.
- `retrieval_eval.py` — one CLI, three modes:
  1. **same-split** — rank each anchor against its split's positives (one gold).
  2. **all-splits** — rank against positives from all splits (harder).
  3. **followup-file** — external JSON with multiple golds per query.
- `llm_baseline.py` — separate CLI because the LLM doesn't produce embeddings; it returns a ranked candidate-ID list which is converted into a synthetic similarity matrix, then run through the same metric code. Candidates are deterministically shuffled (seed `42 + query_id`) to neutralize positional bias.

Retriever specs (`--retriever`, repeatable for head-to-head): `tfidf`, `pretrained`, `pretrained:<hf-model-id>`, `biencoder:<checkpoint-path>`.

## Data layout

`./data/` is gitignored at the directory level but 11 small auto-download datasets plus the internal `aep_causal` are committed (~300 MB total), so a fresh clone is immediately usable. `data/infoquest/` is excluded (~13 GB) — fetch with `uv run followup-data download infoquest`. Re-running `download` on a committed dataset is a no-op.

The follow-up retrieval benchmark ships at `data/aep_followup/followup_questions_eval.json` (100 queries × 300 candidates × 3 golds).

## Conventions

- All text I/O goes through `PairExample` / `TripleExample`. Don't introduce parallel data classes — extend `metadata` instead.
- New retriever types should implement the `Retriever` Protocol, not subclass an existing retriever.
- LLM baseline assumes AWS Bedrock; required env: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BEDROCK_REGION`, `AWS_BEDROCK_ENDPOINT_URL`, `AWS_BEARER_TOKEN_BEDROCK`. Default model is Claude Sonnet 4.
- The literature comparison table in `README.md` is **not a leaderboard** — every row comes from a different paper's own setup. Don't draw cross-row conclusions from it without re-evaluating in a unified setting.

## Memory safety (always follow for ML scripts)

**Check CPU RAM before every heavy operation**. The system has ~1.1 TB total but other processes may use it; running out kills Python mid-run without a clean error.

```python
import psutil
def check_memory(label=""):
    avail_gb = psutil.virtual_memory().available / 1024**3
    print(f"  [mem {label}] avail={avail_gb:.1f}GB")
    if avail_gb < 4.0:
        raise MemoryError(f"OOM risk: only {avail_gb:.1f}GB available — aborting")
```

Call `check_memory()` before:
- Encoding documents with a transformer
- Loading large numpy arrays / tensors
- Running kNN / FAISS operations over N×N matrices
- Each GNN training epoch (every 5 epochs is fine)

Monitor live during long runs: `watch -n 5 free -h`

GPU: before launching training confirm the target GPU has enough free VRAM:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```
Pick the GPU with most free memory; set `CUDA_VISIBLE_DEVICES=<idx>`.

## GNN / `gnn/` sub-module

Implementation of a **Hybrid Cause→Effect GNN** for directional causal document ranking (see `gnn/cause_effect_gnn_implementation.md`).

See `gnn/RESULTS.md` for the full results table and analysis.

Files:
- `gnn/train_gnn_e2e.py` — **the main trainer.** End-to-end fine-tunable two-tower encoder (BGE) + optional GNN structural residual on a stable graph. Flags: `--no-gnn` (encoder only), `--init-from <run> --freeze-encoder` (two-stage GNN on matched features).
- `gnn/evaluate_e2e.py` — test-set eval with same metrics as `finetune_eval/`, prints Δ% vs the bert-base baseline.
- `gnn/model.py` — `HybridCauseEffectGNN`, `DirectionalGATLayer`, `CauseEffectGNN`.
- `gnn/negatives.py` — semantic hard-neg mining (mirrors `finetune_eval/mine_hard_negatives.py`) + anchor-similarity filter.
- `gnn/train_gnn{,_on_bert,_finetune}.py` — earlier frozen-feature variants (superseded by `train_gnn_e2e.py`).

**Python env**: `/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python` (torch 2.6, torch_geometric 2.7, sentence-transformers 5.5).

**Best result (aep_causal test, beats bert-base BiEncoder baseline):**
bge-large two-tower fine-tuned end-to-end → **+21.3% N×N MRR, +19.4% pool MRR, +33% R@1**. bge-base (batch 64, k=8) → +10.8% / +10.9%.

```bash
PY=/mnt/localssd/internship-causal-embedding-merge-followup/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY gnn/train_gnn_e2e.py --dataset aep_causal \
    --encoder BAAI/bge-large-en-v1.5 --run-name aep_causal_e2e_nognn_large \
    --no-gnn --epochs 8 --batch-size 24 --k-neg 8 --encoder-lr 1.5e-5 --bf16
CUDA_VISIBLE_DEVICES=0 $PY gnn/evaluate_e2e.py \
    --run-dir gnn/results/aep_causal_e2e_nognn_large --dataset aep_causal
```

**Two key findings** (aep_causal):
1. The win comes from **a stronger fine-tuned encoder** (BGE ≫ bert-base-uncased for retrieval) + more contrastive signal (batch 64, k=8 hard negs). This is the architectural lever that beats the baseline.
2. The **directional GNN itself does not help** here — tested 3 ways (frozen-feature residual, e2e joint, two-stage on matched features), the learnable mixing gate always collapses toward 0 (the model ignores the GNN). Causal links in AEP docs are text-predictable, so a fine-tuned text encoder already captures the structure. GNN would add value only where graph structure is NOT recoverable from text (citation/KG/multi-hop).

## `arch_improve/` — same-base-model architecture exploration

When the constraint is "keep the base model identical (`bert-base-uncased`) and improve the architecture", see `arch_improve/` and its `RESULTS.md`.

- `arch_improve/train_crossencoder.py` / `evaluate_crossencoder.py` — **cross-encoder (monoBERT)**: jointly encodes `[CLS] anchor [SEP] candidate [SEP]` through one `bert-base-uncased`, full cross-attention, `[CLS]→Linear→score`. Listwise softmax over `positive + hard + random` negatives.

**Result (aep_causal test, identical base model):** cross-encoder beats the BiEncoder by **+11.3% N×N MRR, +59% Recall@1, +12% Recall@5** — pure architecture change.

**Critical detail:** use `--max-seq-length 512` for the cross-encoder (≈256 tokens per text). At 384 the combined pair starves each text (~190 tokens < the bi-encoder's 256) and the cross-encoder *loses*. The cross-encoder is a re-ranker (not indexable) — in production, bi-encoder retrieve → cross-encoder re-rank.
