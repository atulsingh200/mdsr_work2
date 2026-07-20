"""Paper-faithful prompts for TRC — matching arXiv:2410.10476 exactly.

KEY DIFFERENCES from prompts.py:
  1. NO system message — paper uses no system prompt at all.
  2. P prompt: raw "Given the context: [event1]...[/event1] and [event2]...[/event2]. -> LABEL"
     exactly as Table 9 of the paper shows.
  3. QA1: "Given the context: ... Answer the question: Does [event1]X[/event1] happen
     before [event2]Y[/event2]? YES/NO" — one question only, no guidance on format.
  4. QA2: all questions answered in sequence inline — model sees its own prior answers
     as context for each next question (paper's consistency trick).
  5. Full text passed — NO truncation. The paper also passes full sentences.
  6. Few-shot examples inlined directly in the user turn (paper standard).
"""

from __future__ import annotations
import json
import random
import re
from pathlib import Path

PROMPT_TYPES = ("P", "QA1", "QA2")

_E1 = "[event1]"
_E1C = "[/event1]"
_E2 = "[event2]"
_E2C = "[/event2]"


def _wrap(text_1: str, text_2: str) -> str:
    """Paper's exact context format (Table 9)."""
    return (f"Given the context: {_E1}{text_1.strip()}{_E1C} "
            f"and {_E2}{text_2.strip()}{_E2C}.")


# ── Prompt P: example-label, NO system message (paper Table 9) ───────────────
def _user_P(text_1: str, text_2: str) -> str:
    return f"{_wrap(text_1, text_2)} ->"


def _answer_P(label: int) -> str:
    return "BEFORE" if label == 1 else "NOT_BEFORE"


# ── Prompt QA1: one question at a time (paper Table 9) ───────────────────────
def _user_QA1(text_1: str, text_2: str) -> str:
    ctx = _wrap(text_1, text_2)
    t1 = text_1.strip()
    t2 = text_2.strip()
    return (f"{ctx} Answer the question: "
            f"Does {_E1}{t1}{_E1C} happen before {_E2}{t2}{_E2C}? ")


def _answer_QA1(label: int) -> str:
    return "YES" if label == 1 else "NO"


# ── Prompt QA2: all questions in sequence (paper Table 9) ────────────────────
def _user_QA2(text_1: str, text_2: str) -> str:
    ctx = _wrap(text_1, text_2)
    t1 = text_1.strip()
    t2 = text_2.strip()
    return (f"{ctx} Answer the questions: "
            f"Does {_E1}{t1}{_E1C} happen before {_E2}{t2}{_E2C}? "
            f"[BEFORE_ANS] "
            f"Does {_E1}{t1}{_E1C} happen after {_E2}{t2}{_E2C}? "
            f"[AFTER_ANS] "
            f"Does {_E1}{t1}{_E1C} happen at the same time as {_E2}{t2}{_E2C}? "
            f"[EQUAL_ANS]")


def _answer_QA2(label: int) -> str:
    if label == 1:   # BEFORE: yes before, no after, no equal
        return "YES NO NO"
    else:            # NOT_BEFORE (after): no before, yes after, no equal
        return "NO YES NO"


_USER = {"P": _user_P, "QA1": _user_QA1, "QA2": _user_QA2}
_ANSWER = {"P": _answer_P, "QA1": _answer_QA1, "QA2": _answer_QA2}


def build_messages(prompt_type: str, text_1: str, text_2: str,
                   few_shot: list[dict] | None = None) -> list[dict]:
    """Paper-faithful messages: NO system role, raw example-label format.

    Paper (Table 9): all examples and the query are concatenated into a single user turn.
    No system message. No instructional framing. The model must learn from the raw format.
    """
    user_fn = _USER[prompt_type]
    ans_fn = _ANSWER[prompt_type]

    parts: list[str] = []
    if few_shot:
        for ex in few_shot:
            parts.append(user_fn(ex["text_1"], ex["text_2"]) + ans_fn(int(ex["label"])))
            parts.append("")   # blank line between examples
    parts.append(user_fn(text_1, text_2))

    # Single user turn, NO system message — exactly as the paper
    msgs = [{"role": "user", "content": "\n".join(parts)}]
    return msgs


# ── Answer parsing ────────────────────────────────────────────────────────────
_YES = re.compile(r"\b(yes|before)\b", re.I)
_NO = re.compile(r"\b(no|not[_\s]?before|after)\b", re.I)


def parse_answer(prompt_type: str, gen: str) -> int | None:
    g = gen.strip()
    if prompt_type == "P":
        low = g.lower()
        if re.search(r"not[_\s]?before", low):
            return 0
        if re.search(r"\bbefore\b", low):
            return 1
        if re.search(r"\bafter\b", low):
            return 0
        return None
    if prompt_type == "QA1":
        low = g.lower()
        has_yes = bool(re.search(r"\byes\b", low))
        has_no = bool(re.search(r"\bno\b", low))
        if has_yes and not has_no:
            return 1
        if has_no and not has_yes:
            return 0
        return None
    if prompt_type == "QA2":
        low = g.lower()
        # first YES/NO is the before-answer
        m = re.search(r"\b(yes|no)\b", low)
        if m:
            return 1 if m.group(1) == "yes" else 0
        return None
    raise ValueError(prompt_type)


# ── Few-shot sampling ─────────────────────────────────────────────────────────
def sample_few_shot(train_path: str | Path, k_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pos, neg = [], []
    with open(train_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            (pos if int(r["label"]) == 1 else neg).append(r)
    rng.shuffle(pos)
    rng.shuffle(neg)
    shots = pos[:k_per_class] + neg[:k_per_class]
    rng.shuffle(shots)
    return shots
