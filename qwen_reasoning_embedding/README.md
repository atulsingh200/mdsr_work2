# qwen_reasoning_embedding

Focused study: can we add **reasoning** to **Qwen3-Embedding-0.6B** ("reason, then
embed") for the AEP/AJO causal-embedding project — and is it worth it?

This is **research + a code SKETCH only**. No training, no weight downloads.

## Contents
- **`REPORT.md`** — the exhaustive report: SOTA survey of reasoning-augmented
  embedders/retrievers under 1B (O1-Embedder, ReasonIR, RaDeR, DIVER, BRIGHT, GritLM,
  LLM2Vec, Promptriever, etc.), six candidate architectures (A–F) with a comparison
  table, per-task fit, the asymmetry problem, and a clear **verdict** + recommended
  architecture. Real arXiv ids only.
- **`reason_then_embed_sketch.py`** — runnable-shape sketch of the recommended
  architecture (reason-then-embed wrapper over Qwen3-Embedding-0.6B + asymmetric
  cause/effect head). The real torch path is guarded PSEUDOCODE; the safe path is:

  ```bash
  python reason_then_embed_sketch.py --dry-run   # numpy-only, prints tensor shapes
  ```

## Verdict (one line)
Add reasoning as a **query-side reason-then-embed + asymmetric head**, used as a
**complement** for follow-up reranking and directional stage-1 recall — **not** as a
replacement for the DeBERTa cross-encoder + MFAS/Kemeny on workflow ordering.

## Related survey material
- `causal_embedding_survey/sections/06c_reasoning_embeddings.tex` — the survey section
  this study produced (reasoning *embedder* angle).
- Cross-references survey sec. 03 (embeddings), 05 (asymmetric geometry),
  06 (reasoning *reranker*), 11 (system design).
