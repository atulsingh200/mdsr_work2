"""InfoQuest (ICLR Workshop 2025): multi-turn ambiguous-prompt benchmark.

https://huggingface.co/datasets/bryanlincoln/infoquest

The repo contains:
- `seed_messages.jsonl`        — ambiguous initial prompts and personas
- `chats-{user_model}-{assistant_model}-{judge_model}-{seed}.jsonl`
                                 — generated multi-turn dialogues per config
- `evaluations-*.jsonl`, `settings.jsonl`, `traits.jsonl` — auxiliary

Each chats record contains `user_history1` and `user_history2` — lists of
`{role, content}` chat messages from two evaluation runs. We emit
(message t -> message t+1) pairs from each history, skipping system prompts.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import hf_snapshot
from ..registry import register

_REPO_ID = "bryanlincoln/infoquest"


@register
class InfoQuest(BaseDataset):
    name = "infoquest"
    description = (
        "InfoQuest: multi-turn ambiguous-prompt benchmark with hidden personas; "
        "evaluates clarifying-question behavior. (turn t -> turn t+1) pairs "
        "from generated chats."
    )
    homepage = "https://huggingface.co/datasets/bryanlincoln/infoquest"
    citation = "de Oliveira et al., ICLR Workshop 2025 (arXiv 2502.12257)"
    license = "see HF dataset card"
    splits = ("chats",)

    def _root(self) -> Path:
        return self.data_dir / "snapshot"

    def is_downloaded(self) -> bool:
        return self._root().is_dir() and any(self._root().glob("chats-*.jsonl"))

    def download(self) -> None:
        hf_snapshot(_REPO_ID, self._root(), repo_type="dataset")

    def _iter_examples(self) -> Iterator[PairExample]:
        for path in sorted(self._root().glob("chats-*.jsonl")):
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip malformed line, keep iterating
                    for hk in ("user_history1", "user_history2"):
                        history = row.get(hk) or []
                        # Drop system messages.
                        msgs = [m for m in history if m.get("role") != "system"]
                        ctx: list[str] = []
                        for i in range(len(msgs) - 1):
                            anchor = (msgs[i].get("content") or "").strip()
                            positive = (msgs[i + 1].get("content") or "").strip()
                            if not anchor or not positive:
                                continue
                            yield PairExample(
                                anchor=anchor,
                                positive=positive,
                                dataset=self.name,
                                context=tuple(ctx),
                                metadata={
                                    "id": row.get("id"),
                                    "file": path.name,
                                    "history_run": hk,
                                    "anchor_role": msgs[i].get("role"),
                                    "positive_role": msgs[i + 1].get("role"),
                                },
                            )
                            ctx.append(anchor)
