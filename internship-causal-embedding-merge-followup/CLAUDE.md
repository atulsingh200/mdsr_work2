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
