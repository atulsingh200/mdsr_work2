"""Proactive Agent / ProactiveBench (THUNLP).

https://github.com/thunlp/ProactiveAgent

The repo ships per-scenario test data under `dataset/test_data/*.json`. Each
file is a list of observation/agent_response steps:

    {
      "observation": {"time": ..., "event": "..."},
      "agent_response": {"candidate_task": ["...", "...", ...]},
      "task_status": bool
    }

(anchor, positive) mapping (one PairExample per candidate task):
    anchor   = observation.event              (what the user just did)
    positive = candidate_task[i]              (the proactive task to suggest)

Steps with empty candidate_task lists are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import git_clone
from ..registry import register

_REPO = "https://github.com/thunlp/ProactiveAgent.git"


@register
class ProactiveAgent(BaseDataset):
    name = "proactive_agent"
    description = (
        "ProactiveAgent / ProactiveBench: 6,790 events for proactive task "
        "prediction across coding/writing/daily-life. (event -> candidate "
        "proactive task) pairs from dataset/test_data."
    )
    homepage = "https://github.com/thunlp/ProactiveAgent"
    citation = "Lu et al., 2024 (arXiv 2410.12361)"
    license = "Apache-2.0"
    splits = ("train",)

    def _repo_dir(self) -> Path:
        return self.data_dir / "ProactiveAgent"

    def _test_data_dir(self) -> Path:
        return self._repo_dir() / "dataset" / "test_data"

    def is_downloaded(self) -> bool:
        return self._test_data_dir().is_dir() and any(
            self._test_data_dir().glob("*.json")
        )

    def download(self) -> None:
        git_clone(_REPO, self._repo_dir())

    def _iter_examples(self) -> Iterator[PairExample]:
        for path in sorted(self._test_data_dir().glob("*.json")):
            if path.name == "splits.json":
                continue
            with path.open() as f:
                steps = json.load(f)
            history: list[str] = []
            for i, step in enumerate(steps):
                obs = (step.get("observation") or {}).get("event") or ""
                obs = obs.strip()
                tasks = (step.get("agent_response") or {}).get("candidate_task") or []
                for j, task in enumerate(tasks):
                    task = (task or "").strip()
                    if not obs or not task:
                        continue
                    yield PairExample(
                        anchor=obs,
                        positive=task,
                        dataset=self.name,
                        context=tuple(history),
                        metadata={
                            "scenario": path.stem,
                            "step": i,
                            "candidate_idx": j,
                            "task_status": step.get("task_status"),
                            "time": (step.get("observation") or {}).get("time"),
                        },
                    )
                if obs:
                    history.append(obs)
