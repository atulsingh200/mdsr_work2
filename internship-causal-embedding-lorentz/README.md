# follow-up-causal-embeddings

Datasets and middleware for the **follow-up causal embedding** research project.

## Problem framing

Given a piece of text *A* (an anchor — e.g. a system response, a query, an
event log) and a candidate continuation *B*, learn a scoring function

    s(A, B) -> R

such that:
- **Positive pair** (B is the *true* continuation / follow-up of A) → high score
- **Negative pair** (B is unrelated text) → low score

This is the standard contrastive / ranking setup. The library normalizes
every dataset into `PairExample(anchor, positive)` and lets you turn those
into `TripleExample(anchor, positive, negatives)` via pluggable negative
samplers, ready for any contrastive trainer or eval harness.

For the broader literature this is part of, see
`obsidian-thoughts/follow-up internship research project.md`.

## Layout

```
src/followup_data/
  base.py        PairExample, TripleExample, BaseDataset
  registry.py    @register, load(), list_datasets()
  negatives.py   RandomCorpusNegatives, ShufflePositiveNegatives
  io.py          download / extract / git / hf helpers
  cli.py         followup-data {list,info,download,head,export,triples}
  loaders/
    qrecc.py           ✅ direct download
    topiocqa.py        ✅ direct download
    clariq.py          ✅ direct download
    proactive_agent.py ✅ git clone
    multiwoz.py        ✅ via HuggingFace `datasets`
    movielens.py       ✅ direct download (causal-rec smoke test)
    followupqg.py      ✅ HuggingFace mirror
    clamber.py         ✅ direct download (single jsonl)
    mtrag.py           ✅ direct download (IBM mt-rag-benchmark)
    clarq_llm.py       ✅ git clone (ygan/ClarQ-LLM)
    infoquest.py       ✅ HuggingFace snapshot
    aep_causal.py      ✅ ships with the repo (internal AEP docs)
    _manual.py         📝 stubs for datasets needing manual setup

src/biencoder/        # bi-encoder model + training pipeline
  model.py            BiEncoder (untied two-tower), MeanPooling, CLSPooling
  losses.py           InfoNCELoss (in-batch negatives, learnable temperature)
  config.py           MODEL_CONFIGS (bge-small, arctic-embed-l)
  training/
    data.py           PairDataset wrapper over any BaseDataset
    validation.py     In-training MRR / Recall@K
    train.py          Training loop + `train-biencoder` CLI entry
  evaluation_adapter.py   BiEncoderRetriever (conforms to evaluation.Retriever)

src/evaluation/       # model-agnostic evaluation harness
  interfaces.py       Retriever Protocol
  metrics.py          MRR, Recall@K, Hit@K (multi-gold aware)
  retrievers.py       PretrainedRetriever, TfidfRetriever baselines
  retrieval_eval.py   `evaluate-retrieval` CLI (dataset or follow-up modes)
  llm_baseline.py     `evaluate-llm-baseline` CLI (LLM ranker via Bedrock)

examples/score_pairs.py     pairwise-accuracy demo with sentence-transformers
scripts/inspect_dataset.py  basic stats + 3 sample triples
data/                       gitignored cache; downloads land here
```

## Development workflow

**Every change must be committed and pushed before ending a session.**

```bash
# 1. Never work on main — create a branch first
git checkout -b feat/<short-description>

# 2. After each logical change: stage → commit → push
git add <changed files>
git commit -m "feat/fix/chore: short imperative description"
git push -u origin feat/<short-description>   # first push; just `git push` after

# 3. For every subsequent change on the same branch
git add ...
git commit -m "..."
git push   # always push — never leave commits unpushed
```

Rule of thumb: **if you made a change, it must be pushed**. Don't batch pushes across sessions — if the machine reboots or the session dies, unpushed commits are gone. Smaller commits pushed often are better than large commits pushed rarely.

Active branches in this repo:

| Branch | Purpose |
|--------|---------|
| `feat/lorentz-enc-impl` | Lorentz encoder (space/time decomposition, hard-neg mining) |
| `feat/cause-effect-gnn` | Cause→Effect GNN on top of BERT bi-encoder |

---

## Quick start

```bash
cd follow-up-causal-embeddings
uv sync                                          # install
uv run followup-data list                        # see registered datasets
uv run followup-data download clariq qrecc       # fetches into ./data
uv run followup-data head qrecc -n 3
uv run followup-data triples qrecc -k 4 -n 2     # sample (A, P, [N…]) triples
uv run followup-data export qrecc -o data/qrecc_train.jsonl
```

Programmatic use:

```python
from followup_data import load, with_random_negatives

ds = load("qrecc", root="./data", split="train")
ds.download()                                    # idempotent

for ex in ds.head(3):
    print(ex.anchor, "->", ex.positive)

# (anchor, positive, [k negatives sampled from the rest of the dataset])
for triple in with_random_negatives(ds, k=4, seed=0):
    score_pos = model.score(triple.anchor, triple.positive)
    for neg in triple.negatives:
        score_neg = model.score(triple.anchor, neg)
    ...
```

Mix datasets with `chain`:

```python
from followup_data import chain, load

streams = [load("qrecc"), load("clariq"), load("topiocqa")]
for ds in streams:
    if not ds.is_downloaded():
        ds.download()
for pair in chain(streams):
    ...
```

Each `PairExample` carries `anchor`, `positive`, optional `context` (preceding
turns, where applicable), and `metadata` (dataset-specific fields like
conversation id, topic, source).

## Datasets

Each entry below carries a paper link, a one-line `(anchor → positive)`
mapping, the auto-counted pair count, and the source URL. The full
literature catalogue (citations, key ideas, ★ ratings) lives in
`obsidian-thoughts/follow-up internship research project.md`.

### Implemented (auto-download, 12)

Pair counts are with `split=splits[0]` and the loader's natural mapping.

| name              | (anchor → positive) mapping                              | pairs   | paper                                                                                              | source                                                                       |
|-------------------|----------------------------------------------------------|---------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `aep_causal`      | source doc chunk → causal follow-up / prerequisite chunk | 5,487   | Internal (Adobe Experience Platform docs)                                                          | Ships with the repo (no public URL)                                          |
| `qrecc`           | system answer at turn *t* → user question at turn *t+1*  | ~50K    | [Anantha et al., NAACL 2021](https://arxiv.org/abs/2010.04898)                                      | [github.com/apple/ml-qrecc](https://github.com/apple/ml-qrecc)               |
| `topiocqa`        | answer *t* → question *t+1* (with topic switches)        | ~46K    | [Adlakha et al., TACL 2022](https://aclanthology.org/2022.tacl-1.27/)                               | [github.com/McGill-NLP/topiocqa](https://github.com/McGill-NLP/topiocqa)     |
| `clariq`          | initial query → clarifying question                      | ~3K     | [Aliannejadi et al., 2020](https://arxiv.org/abs/2009.11352)                                        | [github.com/aliannejadi/ClariQ](https://github.com/aliannejadi/ClariQ)       |
| `proactive_agent` | event/scene context → proactive task suggestion          | varies  | [Lu et al., 2024](https://arxiv.org/abs/2410.12361)                                                 | [github.com/thunlp/ProactiveAgent](https://github.com/thunlp/ProactiveAgent) |
| `multiwoz_v24`    | user utterance → system response                          | ~57K    | [Zang et al., 2020](https://arxiv.org/abs/2007.12720)                                               | [Brendan/multiwoz_turns_v24](https://huggingface.co/datasets/Brendan/multiwoz_turns_v24) |
| `movielens_1m`    | "previous movie (genres)" → "next movie (genres)"        | ~1M     | [Harper & Konstan, TiiS 2015](https://dl.acm.org/doi/10.1145/2827872)                               | [grouplens.org/datasets/movielens/1m](https://grouplens.org/datasets/movielens/1m/) |
| `followupqg`      | (question + answer) → follow-up question                 | 2,790   | [Meng et al., IJCNLP 2023](https://aclanthology.org/2023.ijcnlp-main.17/) ([arXiv](https://arxiv.org/abs/2309.05007)) | [Vivian12300/FollowupQG](https://huggingface.co/datasets/Vivian12300/FollowupQG) |
| `clamber`         | ambiguous query → clarifying question                    | 1,601   | [Zhang et al., ACL 2024](https://aclanthology.org/2024.acl-long.578/) ([arXiv](https://arxiv.org/abs/2405.12063)) | [github.com/SCUNLP/CLAMBER](https://github.com/SCUNLP/CLAMBER)               |
| `mtrag`           | multi-turn RAG turn *t* → turn *t+1*                     | 1,574   | [Katsis et al., TACL 2025](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.19/132114/)       | [github.com/IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)   |
| `clarq_llm`       | task background → provider full answer                   | 310     | [Gan et al., 2024](https://arxiv.org/abs/2409.06097)                                                | [github.com/ygan/ClarQ-LLM](https://github.com/ygan/ClarQ-LLM)               |
| `infoquest`       | chat turn *t* → turn *t+1* (over many model configs)     | 50K+    | [de Oliveira et al., ICLR Workshop 2025](https://arxiv.org/abs/2502.12257)                          | [bryanlincoln/infoquest](https://huggingface.co/datasets/bryanlincoln/infoquest) |

### Stubs (manual download, 8)

Each stub exposes the same `BaseDataset` interface and looks for a converted
`<data>/<name>/converted.jsonl` (one
`{"anchor": …, "positive": …, "context": [...], "metadata": {...}}` per line).
Once you have the data, drop the JSONL in place and everything else
(`triples`, `head`, `export`, the example scorer) just works.

| name               | paper                                                                                                                                  | notes                                          |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| `share_fqg`        | [FollowGPT, CIKM 2025](https://dl.acm.org/doi/10.1145/3746252.3761401)                                                                  | Released as CIKM artifact                      |
| `ambigsql`         | [ACT, ICLR 2025](https://iclr.cc/virtual/2025/poster/29616)                                                                             | In ACT supplementary                           |
| `agent_cq`         | [Siro et al., 2024 (arXiv)](https://arxiv.org/abs/2410.19692)                                                                           | LLM-generated CQs + crowd-simulated judges     |
| `sim4ia_bench`     | [Sim4IA-Bench, SIGIR 2025 (arXiv)](https://arxiv.org/abs/2511.09329)                                                                    | Next-query prediction                          |
| `coral`            | [Wang et al., NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.72/)                                                    | Multi-turn conversational retrieval            |
| `promise`          | [Butala et al., EACL Findings 2024](https://aclanthology.org/2024.findings-eacl.124/) ([Amazon Science](https://www.amazon.science/publications/promise-a-proactive-multi-turn-dialogue-dataset-for-information-seeking-intent-resolution)) | Proactive intent resolution + suggested QA pairs |
| `trec_ikat`        | [TREC iKAT 2023/2024](https://www.trecikat.com/)                                                                                        | Personalized multi-turn (TREC participation needed) |
| `aiopslab`         | [Chen et al., MLSys 2025 (arXiv)](https://arxiv.org/abs/2501.06706)                                                                     | Microservice fault-injection AIOps env         |

## Adding a dataset

1. Create `src/followup_data/loaders/your_dataset.py`.
2. Subclass `BaseDataset`, set class-level `name`, `description`, `homepage`, `citation`, `license`, `splits`.
3. Implement `is_downloaded()`, `download()`, `_iter_examples()` (yields `PairExample`).
4. Decorate the class with `@register`.
5. Import the new module from `src/followup_data/loaders/__init__.py`.

Helpers in `io.py`: `download_file(url, dest)`, `extract(archive, dest)`, `git_clone(url, dest)`, `hf_snapshot(repo_id, dest)`.

## Negatives

```python
from followup_data import RandomCorpusNegatives, ShufflePositiveNegatives, with_random_negatives

# Materialize a pool, then sample uniformly
triples = list(with_random_negatives(ds, k=4, seed=0))

# Streaming variant (no full corpus in memory)
sampler = ShufflePositiveNegatives(k=4, buffer_size=2048)
for triple in sampler(ds):
    ...
```

For BM25 hard negatives (`pip install '.[hard-negatives]'`) — TODO; the
abstraction is in place, the BM25 sampler is the obvious next addition.

## Smoke-test the embedding side

```bash
uv pip install '.[embeddings]'
uv run python examples/score_pairs.py --dataset clariq -n 200 -k 4
# prints pairwise accuracy: fraction of triples where sim(A,P) > sim(A,N)
```

## Training a bi-encoder

The `biencoder` module trains a two-tower model (untied anchor / positive
encoders) using InfoNCE contrastive loss with in-batch negatives. The trainer
is **dataset-agnostic** — it reads `PairExample.anchor` / `.positive` from any
registered loader, so the same script works for `aep_causal`, `qrecc`,
`clariq`, etc.

```bash
uv pip install '.[training]'

# Train on AEP causal pairs (has its own val split)
uv run train-biencoder \
    --dataset aep_causal --val-split val \
    --backbone bge-small --epochs 15 --batch-size 64 \
    --out-dir runs/aep_causal_bge

# Train on a dataset without a val split — hold out 10% of train
uv run train-biencoder \
    --dataset clariq \
    --backbone bge-small --epochs 5 --batch-size 32

# Smoke test (1 epoch, tiny sample) to check the pipeline
uv run train-biencoder \
    --dataset aep_causal --val-split val \
    --epochs 1 --batch-size 8 \
    --max-train-samples 100 --max-val-samples 50
```

Per-epoch validation runs over the val split and reports MRR, Recall@K
(K=1, 3, 5, 10), mean rank, and median rank. The best checkpoint
(`biencoder_best.pt`) tracks best val MRR; `biencoder_latest.pt` always
holds the most recent epoch. Both, plus `config.json` and
`training_history.json`, are written to `--out-dir`.

Backbones registered in `biencoder.config.MODEL_CONFIGS`:

| key             | model                                         | dim  | pool | prefix       |
|-----------------|-----------------------------------------------|------|------|--------------|
| `bge-small`     | `BAAI/bge-small-en-v1.5`                       | 384  | mean | none         |
| `arctic-embed-l`| `Snowflake/snowflake-arctic-embed-l-v2.0`      | 1024 | cls  | `query: `    |

Full flag reference: `uv run train-biencoder --help`.

## Evaluating retrievers

The `evaluation` module is a **model-agnostic** retrieval harness. Any model
that satisfies the `Retriever` protocol (encodes anchors and candidates into
a shared vector space) plugs into the same CLI without modifying eval code.
Future model approaches (cross-encoders, classifier-based scorers, etc.) just
ship their own adapter — no eval-side changes needed.

```bash
uv pip install '.[evaluation]'
```

### Retriever specs

Pass `--retriever <spec>` one or more times. Recognized specs:

| Spec                                | What it is                                                                                  |
|-------------------------------------|---------------------------------------------------------------------------------------------|
| `tfidf`                             | Classical TF-IDF cosine similarity (no neural component, fast).                              |
| `pretrained`                        | Off-the-shelf HuggingFace encoder. Defaults to `BAAI/bge-small-en-v1.5`.                     |
| `pretrained:<hf-model-id>`          | Off-the-shelf encoder with a specific HF model id (e.g. `pretrained:intfloat/e5-base-v2`).   |
| `biencoder:<checkpoint-path>`       | Trained two-tower bi-encoder loaded from a `biencoder_best.pt` / `_latest.pt` checkpoint.    |

A `random` analytical baseline (`K * n_gold / n_candidates`) is always
included in the output JSON.

### Three retrieval modes (one CLI)

```bash
# 1) Same-split retrieval — rank each anchor against its split's positives
#    (Each query has exactly one gold: the diagonal partner.)
uv run evaluate-retrieval \
    --dataset aep_causal --split test \
    --candidate-pool same-split \
    --retriever biencoder:./runs/aep_causal_bge/biencoder_best.pt

# 2) Full-corpus retrieval — rank against positives from ALL splits.
#    Harder: the candidate pool is much larger (~5,700 for aep_causal).
uv run evaluate-retrieval \
    --dataset aep_causal --split test \
    --candidate-pool all-splits \
    --retriever biencoder:./runs/aep_causal_bge/biencoder_best.pt

# 3) Follow-up retrieval — external JSON, multiple golds per query.
#    100 source questions × 300 candidates × 3 golds (ships with the repo).
uv run evaluate-retrieval \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --retriever biencoder:./runs/aep_causal_bge/biencoder_best.pt
```

### Running each baseline in isolation

```bash
# TF-IDF only
uv run evaluate-retrieval \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --retriever tfidf

# Pretrained baseline (default BGE-small) — no fine-tuning
uv run evaluate-retrieval \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --retriever pretrained

# Pretrained with a specific HF model
uv run evaluate-retrieval \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --retriever pretrained:intfloat/e5-base-v2

# Trained bi-encoder
uv run evaluate-retrieval \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --retriever biencoder:./runs/aep_causal_bge/biencoder_best.pt
```

### Head-to-head comparison in one invocation

Pass `--retriever` multiple times — each runs against the same task and the
metrics for all of them go into a single JSON output:

```bash
uv run evaluate-retrieval \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --retriever biencoder:./runs/aep_causal_bge/biencoder_best.pt \
    --retriever pretrained:BAAI/bge-small-en-v1.5 \
    --retriever tfidf \
    --out results/followup_comparison.json
```

Each retriever reports MRR, Recall@K, Hit@K, mean rank, and median rank.
Note: when `n_gold_per_query == 1`, Recall@K and Hit@K coincide; for multi-gold
tasks (like the follow-up benchmark) they differ — `Recall@K` is the average
fraction of golds found in top-K, while `Hit@K` is the fraction of queries
with at least one gold in top-K.

### Smoke-testing without a full eval

```bash
# Only evaluate the first 20 queries (useful for sanity checks)
uv run evaluate-retrieval \
    --dataset aep_causal --split test --candidate-pool same-split \
    --retriever tfidf --max-queries 20
```

Full flag reference: `uv run evaluate-retrieval --help`.

### LLM baseline

For the follow-up retrieval task, a separate script (`evaluate-llm-baseline`)
asks an LLM to rank candidates directly. The LLM doesn't fit the embedding
`Retriever` protocol — it produces a ranking from raw text reasoning — so it
gets its own CLI. Output JSON is in the same shape as `evaluate-retrieval`,
so the two are directly comparable.

```bash
uv pip install '.[llm-baseline]'

# Requires AWS Bedrock credentials in the environment:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#   AWS_BEDROCK_REGION, AWS_BEDROCK_ENDPOINT_URL, AWS_BEARER_TOKEN_BEDROCK

# Default: Claude Sonnet 4
uv run evaluate-llm-baseline \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --top-k 10

# Use a different Bedrock model (e.g. Haiku for cheaper / faster eval)
uv run evaluate-llm-baseline \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
    --top-k 10

# Smoke test: just the first 5 queries
uv run evaluate-llm-baseline \
    --followup-file data/aep_followup/followup_questions_eval.json \
    --max-queries 5 --delay 0
```

For each query, the 300 candidates are deterministically shuffled (seed =
`42 + query_id`) before being shown to the LLM, so the model can't exploit
positional bias. The LLM returns a JSON array of top-K candidate IDs in
ranked order; that ranking is converted into a synthetic similarity matrix
and fed through the same metric code as the embedding retrievers.

Full flag reference: `uv run evaluate-llm-baseline --help`.

## Datasets currently downloaded in this checkout

The 10 small auto-download datasets plus the internal `aep_causal` dataset
(11 in total) are committed under `./data/<name>/`, so a fresh clone is
immediately usable — no `download` step needed for them. Total committed
data: ~300 MB.

`data/infoquest/` is excluded from the repo (~13 GB) — fetch it locally with:

```bash
uv run followup-data download infoquest
```

Re-running `download` on any committed dataset is a no-op if files are
already present.

## Reported results from the literature

Each row below is a method drawn from the §1–§3 lit-review at
`obsidian-thoughts/follow-up internship research project.md`. Each column is
a `dataset · metric` pair as reported in the method's *own paper*. Cells are
the headline values from each paper's main results table; "—" = method
exists but no usable value was retrievable (paywall / PDF-only / no
public mirror).

**Read this carefully before drawing conclusions.** The table is intentionally
sparse and **cross-row comparisons are mostly not apples-to-apples**: every
paper picks its own splits, backbones, baselines, and metrics. A higher number
in column X for method A doesn't mean A > B unless both report on X with the
same setup. The columns aren't a unified leaderboard — they're a map of
"where each method has been benchmarked so we know what to re-evaluate on
when picking a baseline." Numbers were sourced 2026-05-11; trust the original
papers over this snapshot.

### Metric legend (column suffix → meaning)

| Suffix | Meaning |
|---|---|
| `MTEB-Avg`, `MTEB-Retr` | MTEB main score / retrieval-subtask nDCG@10 |
| `Faith`, `CtxP` | Ragas faithfulness / context-precision (0-100) |
| `HR@k`, `NDCG@k`, `R@k` | top-k hit-rate / nDCG / recall |
| `MCC` | mean correlation coefficient (identifiability) |
| `CtxFID` | Context-FID, time-series generation quality (lower better) |
| `WGA` | worst-group accuracy |
| `MAE`, `MSE` | mean absolute / squared error (lower better) |
| `F1`, `Acc` | classification F1 / accuracy |
| `EM` | exact match |
| `SR` | task success rate |
| `WR` | win rate vs a named baseline (see notes) |
| `RIM` | Requested-Information-Match (LLM-as-judge faithfulness) |
| `pass@1` | code-gen pass rate, single sample |
| `Goal` | task-oriented dialog goal-completion rate |
| `Lift` | delta vs the strongest baseline (absolute value not disclosed) |

### The table

Format: each cell is the value as reported. `+X` means a relative or absolute
*lift* over the paper's strongest baseline (used when the absolute value
isn't disclosed — common in industry papers). Empty cell = method didn't
report on this dataset.

| Method | MTEB-Avg | MTEB-Retr | OpenAlex-Faith | OpenAlex-CtxP | MSSD-NDCG@10 | GPT4books-NDCG@10 | ZhihuRec-NDCG@10 | Tenrec-NDCG@10 | KuaiRand-NDCG@10 | Synth-MCC | Human3.6M-MSE | Weather-CtxFID | CivilComm-WGA | MultiNLI-WGA | Netflix-MAE | Amazon-MAE | ProactiveBench-F1 | CTR-lift | CtxAgent-AccP-OOD | MultiInterp-AnsF1 | Abg-CoQA-F1 | PACIFIC-F1 | AmbigSQL-F1 | PartialSpec-WR | FUQG-TopicCons | FUQG-Inform | Patient-RIM | MultiWOZ-Goal | Persuasion-Reward | iKAT-F1 | Bamboogle-Acc | 2Wiki-Acc | Musique-Acc | Craigslist-SR | ESConv-SR | CIMA-SR | NetflixInt-Lift | ML-10M-Lift | Criteo-Lift | Adressa-R@20 | Yelp-R@20 | ML-10M-NDCG@50 | Netflix-R@20 | CausalRec-Lift | Coat-AUC | TLDR-WR | HH-WR | OpenAI-WR-vs-GPT3 | AlpacaEval-2.0 | MT-Bench | HotpotQA-EM | FEVER-Acc | ALFWorld-SR | WebShop-SR | HumanEval-pass@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **§1 — Causal embeddings** | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Causal2Vec (Mistral-7B) | 66.10 | 57.28 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CausalRAG | | | 78.00 | 92.86 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CSRec | | | | | 0.2117 | 0.0463 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CaseRec | | | | | | | 0.0540 | 0.0888 | 0.1350 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| IDOL | | | | | | | | | | 0.929 | 0.0658 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CHiLD | | | | | | | | | | 0.852 | | 0.507 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CCR | | | | | | | | | | | | | 0.7067 | 0.7517 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| New-User Event Pred. (C-NH / C-RMTPP) | | | | | | | | | | | | | | | 5.63 | 4.17 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| IEM (Identifiable Exchangeable Mech.) | — | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CPNS | — | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CCL (implicit hate) | — | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| C2lRec | — | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| **§2 — Follow-up generation** | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| FollowGPT (CIKM 2025) | — | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Proactive Agent (LLaMA-3.1-8B FT) | | | | | | | | | | | | | | | | | 66.5% | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| GQS (CTR-Guided) | | | | | | | | | | | | | | | | | | +5.21pts | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| ContextAgent (Llama-3.1-8B) | | | | | | | | | | | | | | | | | | | 90.9% | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Double-Turn DPO (Modeling Future Turns) | | | | | | | | | | | | | | | | | | | | +5pts | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| ACT (Action-Based Contrastive ST) | | | | | | | | | | | | | | | | | | | | | 52.3 | 82.2 | 80.7 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Teaching LMs to Gather Info Proactively | | | | | | | | | | | | | | | | | | | | | | | | +18% | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| KG+LLM Follow-up QG ("Superficial to Deep") | | | | | | | | | | | | | | | | | | | | | | | | | 54.42% | 1.98 | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| FollowupQ | | | | | | | | | | | | | | | | | | | | | | | | | | | +17% | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| ProTOD | | | | | | | | | | | | | | | | | | | | | | | | | | | | +10pts | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Related Insight Gen. (SCOpE-QA) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Grounded in Reality (Learn-to-Ask) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Causal Persuasive Dialogue | | | | | | | | | | | | | | | | | | | | | | | | | | | | | $507 | | | | | | | | | | | | | | | | | | | | | | | | | |
| Modelling User Actions in CIR | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 0.630 | | | | | | | | | | | | | | | | | | | | | | | | |
| **§3 — Baselines** | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Self-Ask (text-davinci-002) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 57.6% | 30.0% | 13.8% | | | | | | | | | | | | | | | | | | | | | |
| Intent-aware Clarifying QG | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| PPDPP | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 0.623 | 0.578 | 0.463 | | | | | | | | | | | | | | | | | | |
| FM-Intent (Netflix) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | +7.4% | | | | | | | | | | | | | | | | | | |
| Empowering Retrieval-CRS (CORAL) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CausE | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | +21% | +8% | | | | | | | | | | | | | | | | |
| MACR (MF) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 0.164 | 0.144 | | | | | | | | | | | | | | |
| DICE | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | +15% | +20% | | | | | | | | | | | | |
| CausalRec | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | +4.17% | | | | | | | | | | | |
| FCSRec | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| PGCR | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| IPS / SNIPS (Schnabel 2016) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| MRDR (MR, AAAI 2023) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 0.774 | | | | | | | | | | |
| DCRec | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| CounterCLR | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| DPO | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 61.0% | ~58% | | | | | | | | |
| PPO-RLHF (InstructGPT) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 85% | | | | | | | |
| ODPO ("DPO w/ Offset") | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | >DPO | | | | | | | | | |
| ORPO (Mistral-ORPO-β 7B) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 12.2% | 7.32 | | | | | |
| ReAct (PaLM-540B) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 35.1% | 64.6% | 71.0% | 40.0% | |
| Reflexion (GPT-4) | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | ~51% | | ~98% | | 91.0% |

### Notes on specific cells

- **Causal2Vec** — beats next-best public-training baseline `bge-en-icl` (66.08) by 0.02 on MTEB-Avg. With ICL prefix, rises to 66.85.
- **CausalRAG** — context-recall is *lower* than baselines (49.46 vs RegularRAG ~70); the win is in faithfulness/precision balance.
- **CSRec** — wins 5/6 HR/NDCG cells on MSSD vs Few-Shot baselines; on synthetic GPT4-books, traditional SASRec / BERT4Rec are within noise.
- **CaseRec** — values are CaseRec-S variant. Improvements: ZhihuRec +3.23×, Tenrec +1.28×, KuaiRand +1.75× over best baseline.
- **IDOL / CHiLD** — `Synth-MCC` is the mean over their respective synthetic suites (A–F for IDOL, A–G for CHiLD); not directly comparable since CHiLD adds an extra hierarchy-stress dataset (G).
- **CCR** — paper claims *best among methods that don't require group labels*; absolute WGA on MultiNLI is 0.7517 vs Group-DRO (which uses labels) 0.777.
- **NUEP** (New-User Event Prediction) — `C-NH` variant for Netflix, `C-RMTPP` for Amazon; classification-accuracy variants are 0.347 / 0.274.
- **Proactive Agent** — F1 on ProactiveBench's 6,790-event held-out split; vs LLaMA-3.1-8B-Instruct base 55.1%.
- **GQS** — CTR lift is from live A/B in industrial conversational search; baseline is DPO-only. Also +2.28 relevance, +22.66 diversity.
- **ContextAgent** — OOD subset; in-distribution F1 is +7.0 pts over best baseline (absolute not disclosed in extract).
- **Double-Turn DPO** (Modeling Future Conversation Turns) — `+5pts` is over prior clarification methods on a multi-interpretation QA benchmark; also +3pts on the clarify-vs-direct decision accuracy.
- **ACT** — values are Macro-F1 at 50-conversation setting; AmbigSQL also reports execution-match 48.0%.
- **Teaching LMs to Gather Info** — `+18%` is auto-eval win rate of Qwen-2.5-7B (RL fine-tuned) over o3-mini; human pairwise preference on clarifying questions is +42%.
- **KG+LLM FUQG** — TopicCons of 54.42% is *worse* than baselines (GPT-Neo 77.11%); the headline wins are Distinct-1 (33.84% vs gpt-3.5-turbo 31.73%) and Informativeness (1.98 vs 1.76).
- **FollowupQ** — `+17%` is RIM (Requested-Information-Match) lift on real patient-provider messages; also reduces required follow-ups by 34%.
- **ProTOD** — `+10pts` on MultiWOZ 2.1 with only 10% of the training data used.
- **Causal Persuasive Dialogue** — `$507` is cumulative donation reward on the PersuasionForGood corpus, BiCoGAN+DPPR case 1; ground-truth human dialogues achieve $478.
- **Modelling User Actions in CIR** — Macro-F1 0.630 is for *labelling* user actions; per-action prediction F1 ranges 0.18–0.64.
- **Self-Ask** — text-davinci-002, no search-engine assistance; with search the numbers rise (Bamboogle → 60.0%, 2Wiki → 40.1%, Musique → 15.2%).
- **PPDPP** — Success-Level percentage on each dialog domain.
- **FM-Intent** — Netflix internal, +7.4% relative lift over TransAct; no public-benchmark absolute number disclosed.
- **CausE** — `Lift` is MSE improvement over WSP2V (MovieLens-10M) and counterfactual evaluation on Criteo's REG ad-prediction split.
- **MACR / DICE / CausalRec** — RecSys benchmarks; DICE/CausalRec/CORAL only published relative improvements in fetchable snippets.
- **DPO** — TL;DR 61.0% GPT-4 win rate vs PPO best ~57%; Anthropic HH ~58% best-of-N=128. On IMDb the paper reports a Pareto-dominated reward/KL frontier rather than a single scalar.
- **PPO-RLHF** — 85% labeller-preference win rate of InstructGPT over GPT-3 175B on the OpenAI API prompts split; vs FLAN 78%.
- **ODPO** — "DPO with an Offset" (Findings ACL 2024 #592); improves over DPO on IMDb reward/KL frontier, RealToxicityPrompts toxicity, and Reddit TL;DR — no single scalar in the fetched extract.
- **ORPO** — Mistral-ORPO-β 7B after UltraFeedback fine-tuning.
- **ReAct** — PaLM-540B + CoT-SC self-consistency for HotpotQA; ALFWorld is best-of-6-trials.
- **Reflexion** — HotpotQA ~+20pts over ReAct baseline (paper reports ~51%); ALFWorld 130/134 = ~98% after 12 reflection iterations.

### Methods listed in the lit-review but not in the table

- **Intent-aware Clarifying QG, FCSRec, PGCR, DCRec, CounterCLR, IPS/SNIPS** — papers are accessible but the main results table didn't extract cleanly (PDF-binary or paywalled venue with no arxiv mirror).
- **FollowGPT, C2lRec, CPNS, CCL** — paywalled or no public mirror with surface-able numbers.
- **Identifiable Exchangeable Mechanisms (IEM)** — purely theoretical, no benchmarks reported.
- **Related Insight Generation, Grounded in Reality** — fetched but no headline scalar metric on a standard benchmark.

To plug a gap: locate the paper PDF, extract the main results table, and add a row to the table above with the relevant column(s) populated. Keep cell values to 3–4 significant figures; flag with `+X` lifts when only deltas are disclosed.

## Hardware & Resource Guidelines

### GPU — 4 × A100 (use all of them)

This machine has four A100 GPUs. Use all four wherever the workload benefits from it.

**Training (`train-biencoder`)** — launch with `torchrun` for DDP:

```bash
torchrun --nproc_per_node=4 -m biencoder.training.train \
    --dataset aep_causal --val-split val \
    --backbone bge-small --epochs 15 --batch-size 256 \
    --out-dir runs/aep_causal_bge_4gpu
```

With 4 GPUs the effective batch size is `--batch-size × 4`, which gives InfoNCE a much larger
in-batch negative pool and typically improves contrastive loss quality. Halve `--batch-size`
compared to single-GPU runs to keep per-device memory the same.

**Encoding / evaluation** — set `CUDA_VISIBLE_DEVICES=0,1,2,3` and split the candidate pool
across devices if you are encoding large corpora in a one-off script. `torch.nn.DataParallel`
works for inference-only tasks; DDP is preferred for training.

```python
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model = torch.nn.DataParallel(model)   # inference; use DDP for training
```

### CPU / RAM — avoid OOM crashes

The system's CPU RAM is a **shared, finite resource**. Unlike GPU OOM (which raises a Python
exception and recovers), CPU OOM can cause the kernel OOM-killer to terminate your process or
destabilize the machine for everyone.

Rules to follow:

| Situation | Guideline |
|-----------|-----------|
| Loading a full dataset into memory | Use streaming / iterator APIs (`_iter_examples()`, `ShufflePositiveNegatives`) rather than materializing the full corpus into a list unless you explicitly need random access. |
| `RandomCorpusNegatives` (materializes a pool) | Limit pool size with `max_pool` when the dataset is large (e.g. `movielens_1m` at ~1 M pairs). Consider `ShufflePositiveNegatives` instead. |
| `infoquest` (13 GB, not committed) | Never load in full; iterate with `_iter_examples()` and batch. |
| Multi-worker DataLoaders | Keep `num_workers ≤ 4`; each worker forks the parent address space. More workers → proportional RAM multiplication. |
| Large HuggingFace datasets | Pass `keep_in_memory=False` (the default) and never call `.to_pandas()` on multi-GB splits. |
| Encoding large candidate pools | Encode in chunks of ≤ 50 k sentences and accumulate results on disk or a memory-mapped array, not a growing Python list. |

Quick check before a long run:

```bash
free -h          # available RAM
nvidia-smi       # GPU memory headroom
```

If `available` RAM is below ~20 GB before starting, reduce batch size or disable in-memory caching before launching.

## License

Code: same license as the parent project (TBD). Each dataset retains its own
license — see the class-level `license` attribute and the homepage URL before
distributing derivatives.
