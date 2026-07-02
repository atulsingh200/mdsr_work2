# Survey contributor style guide (READ BEFORE WRITING)

You are writing ONE section of a multi-author LaTeX survey titled
*"Causal Embedding Models for Procedural Documentation: A Survey of Methods
for Follow-up Reranking and Workflow Ordering."*

## Output contract
1. Write your section to `sections/<NN_name>.tex` (path given in your task).
2. Write your bibliography entries to `bib/<NN_name>.bib`.
3. Do NOT edit `main.tex`, `refs.bib`, or any other agent's files.

## LaTeX rules (so it compiles with `tectonic`)
- Start your `.tex` file directly with `\section{...}` (no preamble, no `\documentclass`).
- Use only packages already loaded in `main.tex`: amsmath, amssymb, graphicx, booktabs,
  array, multirow, xcolor, hyperref, url, enumitem, natbib. Do NOT add `\usepackage`.
- Citations: `\citep{key}` (parenthetical) / `\citet{key}` (textual). natbib numeric style.
- Escape special chars in text: `%` -> `\%`, `&` -> `\&`, `_` -> `\_`, `#` -> `\#`.
- Tables: use `booktabs` (`\toprule`/`\midrule`/`\bottomrule`). For wide tables in this
  two-column doc, use `table*` + `\resizebox` is NOT available (no graphicx resizebox issues:
  graphicx IS loaded, `\resizebox` is fine). Prefer `\small`/`\footnotesize` tables.
- Figures you generate go in `figures/` as PDF or PNG; include with
  `\includegraphics[width=\linewidth]{name}`.

## BibTeX rules
- Prefix every bib key with your section tag to avoid collisions, e.g. for the embeddings
  section use `emb:karpukhin2020dpr`. Your task tells you the prefix.
- Prefer real, verifiable references. Use `@inproceedings`/`@article`/`@misc`.
- For arXiv papers use `@misc` with `eprint`, `archivePrefix={arXiv}`, `year`, `note={arXiv:XXXX.XXXXX}`.
- DO NOT invent DOIs. If unsure of exact venue/year, use `@misc` and arXiv id. It is better
  to be slightly vague than to fabricate a precise-looking wrong citation.

## Content rules
- This is a SURVEY: be exhaustive, technical, and organized. Cover the landscape, compare
  methods, give equations where they clarify, and end your section with a short
  "Relevance to causal embedding for AEP/AJO" paragraph connecting back to the project.
- Target length: 1.5–3 pages of dense two-column text per section (≈ 900–1800 words) unless
  your task says otherwise.
- Use `\subsection` and `\paragraph` to structure.
- Be accurate. When you state a number/result, it should come from a real paper or from the
  repo data your task points you at. Flag genuine uncertainty rather than fabricating.

## Project context (the "why")
The goal is a **causal embedding model** (< 1B params) that understands Adobe Experience
Platform (AEP) and Adobe Journey Optimizer (AJO) docs + video tutorials, and can:
(1) judge directional causal precedence between two steps/texts,
(2) **rerank follow-up questions**, and
(3) **sort workflow steps into ascending order**.
Existing repo findings live in `/mnt/localssd/research/01..08_*.md` — read the ones relevant
to your section and build on them (don't contradict without saying so).
