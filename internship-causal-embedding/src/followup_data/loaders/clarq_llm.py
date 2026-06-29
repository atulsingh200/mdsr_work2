"""ClarQ-LLM: task-oriented clarifying-question benchmark (EN + ZH).

https://github.com/ygan/ClarQ-LLM

31 task types × 10 dialogue scenarios per language. Each scenario describes a
game-like task with `background`, `all_response` (provider's eventual full
answer), `background_splitted`, and three model-specific dialog logs
(h2h / h2l / l2l) which may be empty until the benchmark is run.

(anchor, positive) mapping (best-effort static framing):
    anchor   = background    (the task scenario the seeker faces)
    positive = all_response  (the provider's full answer chain)

This is a coarse pair — for finer per-turn pairs you can run the benchmark
agents and convert the resulting h2h/h2l/l2l logs into PairExamples.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import git_clone
from ..registry import register

_REPO = "https://github.com/ygan/ClarQ-LLM.git"


@register
class ClarQLLM(BaseDataset):
    name = "clarq_llm"
    description = (
        "ClarQ-LLM: 31 task types × 10 scenarios, EN+ZH task-oriented "
        "clarifying-question benchmark. Background -> provider full answer."
    )
    homepage = "https://github.com/ygan/ClarQ-LLM"
    citation = "Gan et al., 2024 (arXiv 2409.06097)"
    license = "MIT"
    splits = ("english", "chinese")

    def _repo(self) -> Path:
        return self.data_dir / "ClarQ-LLM"

    def _lang_dir(self) -> Path:
        return self._repo() / "data" / ("English" if self.split == "english" else "Chinese")

    def is_downloaded(self) -> bool:
        d = self._lang_dir()
        return d.is_dir() and any(d.glob("*.json"))

    def download(self) -> None:
        git_clone(_REPO, self._repo())

    def _iter_examples(self) -> Iterator[PairExample]:
        for path in sorted(self._lang_dir().glob("*.json")):
            with path.open() as f:
                scenarios = json.load(f)
            for idx, sc in enumerate(scenarios):
                anchor = (sc.get("background") or "").strip()
                positive = (sc.get("all_response") or "").strip()
                if not anchor or not positive:
                    continue
                yield PairExample(
                    anchor=anchor,
                    positive=positive,
                    dataset=self.name,
                    metadata={
                        "task_file": path.name,
                        "scenario_idx": idx,
                        "language": self.split,
                    },
                )
