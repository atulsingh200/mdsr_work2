"""Prompt construction for the workflow step-ordering task.

Adapts the step-ordering prompt from arXiv:2511.04688v2 (Figure 7), replacing the cooking-recipe
framing with generic procedural-workflow framing while keeping the strict JSON-only output
contract: {"reordered_steps": [...], "order": [...]}.

Steps are presented 1-indexed. "order" is the sequence of 1-indexed positions in the shuffled
list that recovers the correct order.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are given a list of randomly shuffled steps from a procedural workflow. These steps "
    "are out of their intended logical order. Your task is to reorder them based on your "
    "understanding of how the workflow should be carried out.\n"
    "Workflows usually follow a logical progression: prerequisites and setup first, then the "
    "core actions in the order they depend on one another, ending with final or confirming steps.\n\n"
    "### Instructions:\n"
    "- Analyze the shuffled steps and infer the most logical correct order.\n"
    "- Return ONLY a JSON object in the exact format described below — no extra text or explanation.\n"
    "- Do not renumber or reword the steps; return them as-is from the input, just reordered.\n"
    "- You must respond with ONLY the JSON object — no explanations, comments, or markdown formatting.\n"
    "  Your output will be parsed automatically, so format strictly.\n"
    "### Output (JSON format only):\n"
    '{\n'
    '"reordered_steps": [<step_1>, <step_2>, ..., <step_n>],\n'
    '"order": [<1-indexed position in the shuffled list of step_1>, ..., <position of step_n>]\n'
    '}'
)


def _format_input(title: str, shuffled_steps: list[str]) -> str:
    lines = [f"Workflow Name: {title}", "Shuffled steps:"]
    for i, s in enumerate(shuffled_steps, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def _format_answer(shuffled_steps: list[str], order: list[int]) -> str:
    reordered = [shuffled_steps[k - 1] for k in order]
    return json.dumps({"reordered_steps": reordered, "order": order})


def build_messages(title: str, shuffled_steps: list[str], few_shot: list[dict] | None) -> list[dict]:
    """few_shot: list of demo dicts with keys title, shuffled_steps, gold_order."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for d in few_shot:
            msgs.append({"role": "user", "content": _format_input(d["title"], d["shuffled_steps"])})
            msgs.append({"role": "assistant",
                         "content": _format_answer(d["shuffled_steps"], d["gold_order"])})
    msgs.append({"role": "user", "content": _format_input(title, shuffled_steps)})
    return msgs
