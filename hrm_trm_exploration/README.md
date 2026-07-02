# HRM / TRM Exploration — Latent & Recursive Reasoning for AEP/AJO Workflow Ordering

Feasibility study: can **HRM** (Hierarchical Reasoning Model) / **TRM** (Tiny Recursive Model)
and the broader **latent / recursive reasoning** family help this project's sub-1B causal
ordering problem?

## Contents
- `REPORT.md` — the full feasibility study: accurate summaries of HRM, TRM, the ARC Prize
  team's HRM ablation, and the broader latent-reasoning family; task mapping; honest
  pros/cons; comparison vs. the survey's CE + MFAS/Kemeny + distillation + GRPO plan;
  a verdict; the TRM-Orderer architecture proposal; and a real reference list.
- `trm_orderer_sketch.py` — runnable-shape PyTorch *sketch* of the proposed TRM-style
  workflow-orderer (model definition + forward pass on random tensors to prove shapes).
  Guards all heavy ops; **no training is launched**. Numpy `--dry-run` fallback if torch
  is unavailable.

## Run the sketch
```bash
python trm_orderer_sketch.py            # torch forward on random tensors (prints shapes)
python trm_orderer_sketch.py --dry-run  # numpy-only shape walk-through (no torch needed)
```

## One-line verdict
**Worth pursuing as a Track-B research bet:** a tiny (5-7M-param) TRM-style recursive
network on top of the **frozen cross-encoder**, used as a **learned rank-aggregation /
listwise ordering head** — i.e. a learned alternative/complement to MFAS/Kemeny at the
aggregation stage. It does **not** replace the cross-encoder (it can't recover the
cross-attention that drives the CE's strength) and only beats MFAS/Kemeny in the
noisy-but-decent, cyclic, partial-order regime. Build Tier-1 (MFAS/Kemeny) first. Main
risks: scarce gold full-orderings (overfit) and the ARC-team finding that HRM's gains come
from the refinement loop + augmentation, not the architecture. See `REPORT.md`.

## Companion survey deliverables (written elsewhere, per task)
- `causal_embedding_survey/sections/06b_latent_reasoning.tex`
- `causal_embedding_survey/bib/06b_latent_reasoning.bib`
