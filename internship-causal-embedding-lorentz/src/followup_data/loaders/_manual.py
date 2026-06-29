"""Stubs for datasets that aren't trivially auto-downloadable.

Each subclass keeps the BaseDataset interface so the rest of the pipeline
(registry, CLI, negatives, examples) works uniformly. `download()` raises
`ManualDownloadRequired` with the homepage URL and any extra setup notes.

When access is sorted, replace `_iter_examples` and `is_downloaded` with the
real implementation; or open a small JSONL adapter at the documented path.

Each adapter looks for a converted JSONL at:
    <data_dir>/<name>/converted.jsonl
with one record per line: {"anchor": ..., "positive": ..., ...}.
This lets the intern hand-convert the data once and have everything else
just work.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..registry import register


class ManualDownloadRequired(RuntimeError):
    pass


class _ManualDataset(BaseDataset):
    """Common 'check for converted.jsonl' implementation for stub datasets."""

    splits = ("train",)
    setup_notes: str = ""

    def _converted(self) -> Path:
        return self.data_dir / "converted.jsonl"

    def is_downloaded(self) -> bool:
        return self._converted().exists() and self._converted().stat().st_size > 0

    def download(self) -> None:
        if self.is_downloaded():
            return
        raise ManualDownloadRequired(
            f"\n{self.name}: manual download required.\n"
            f"  Homepage: {self.homepage}\n"
            f"  {self.setup_notes}\n"
            f"  Once obtained, write a JSONL with one "
            f'{{\"anchor\": ..., \"positive\": ...}} record per line to:\n'
            f"    {self._converted()}\n"
        )

    def _iter_examples(self) -> Iterator[PairExample]:
        with self._converted().open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                anchor = row.get("anchor", "").strip()
                positive = row.get("positive", "").strip()
                if not anchor or not positive:
                    continue
                yield PairExample(
                    anchor=anchor,
                    positive=positive,
                    dataset=self.name,
                    context=tuple(row.get("context", ())),
                    metadata=row.get("metadata", {}),
                )


@register
class ShareFQG(_ManualDataset):
    name = "share_fqg"
    description = (
        "ShareFQG (FollowGPT, CIKM 2025): follow-up intents mined from real "
        "LLM chat logs, hierarchical filtering + intent-transition synthesis."
    )
    homepage = "https://dl.acm.org/doi/10.1145/3746252.3761401"
    citation = "FollowGPT, CIKM 2025"
    setup_notes = "Released with the CIKM 2025 paper; check the artifacts repo."


@register
class AmbigSQL(_ManualDataset):
    name = "ambigsql"
    description = (
        "AmbigSQL: ambiguous-intent SQL clarification benchmark from the ACT "
        "(Action-Based Contrastive Self-Training) ICLR 2025 paper."
    )
    homepage = "https://iclr.cc/virtual/2025/poster/29616"
    citation = "ACT, ICLR 2025"
    setup_notes = "Look for AmbigSQL release in the ACT paper's supplementary."


@register
class AgentCQ(_ManualDataset):
    name = "agent_cq"
    description = (
        "AGENT-CQ: LLM-generated clarifying questions with crowd-simulated "
        "judges; useful for synthetic follow-up training data."
    )
    homepage = "https://arxiv.org/abs/2410.19692"
    citation = "AGENT-CQ, 2024"


@register
class Sim4IABench(_ManualDataset):
    name = "sim4ia_bench"
    description = (
        "Sim4IA-Bench: 160 real CORE search-engine sessions for next-query / "
        "next-utterance prediction. First public next-query benchmark."
    )
    homepage = "https://arxiv.org/abs/2511.09329"
    citation = "Sim4IA-Bench, SIGIR 2025"


@register
class CORAL(_ManualDataset):
    name = "coral"
    description = (
        "CORAL: large multi-turn conversational retrieval corpus (NAACL "
        "Findings 2025). MRR/NDCG over sessions."
    )
    homepage = "https://aclanthology.org/2025.findings-naacl.72/"
    citation = "CORAL, NAACL Findings 2025"


@register
class ProMISe(_ManualDataset):
    name = "promise"
    description = (
        "ProMISe: 1,025 dialogues / 4,453 turns / 17,812 SQA pairs of proactive "
        "multi-turn intent resolution + suggested-question prediction."
    )
    homepage = "https://aclanthology.org/2024.findings-eacl.124/"
    citation = "ProMISe, EACL Findings 2024"


@register
class IKAT(_ManualDataset):
    name = "trec_ikat"
    description = (
        "TREC iKAT 2023/2024: personalized multi-turn passage ranking with a "
        "Personal Text Knowledge Base. Closest to operational-insights setting."
    )
    homepage = "https://www.trecikat.com/"
    citation = "TREC iKAT 2023/2024"
    setup_notes = "Requires TREC participant agreement; see homepage for access."


@register
class AIOpsLab(_ManualDataset):
    name = "aiopslab"
    description = (
        "AIOpsLab: microservice cloud env with fault injection for AIOps "
        "agents (detection / RCA / mitigation). Closest framework to our domain."
    )
    homepage = "https://github.com/microsoft/AIOpsLab"
    citation = "AIOpsLab, MLSys 2025"
    setup_notes = (
        "Requires container runtime to launch the env; convert agent rollouts "
        "to (state -> next action) pairs."
    )
