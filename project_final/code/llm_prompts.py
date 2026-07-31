"""Prompt construction + answer parsing for the LLM TRC baseline.

Adapts the paper's three prompt schemas (arXiv 2410.10476, Table 1 & 9) to our BINARY directional
task: given two procedural steps A (=text_1) and B (=text_2), decide whether A happens BEFORE B.

    label == 1  ->  A BEFORE B
    label == 0  ->  A NOT_BEFORE B   (i.e. B is before A / not ordered A->B)

Prompt types (each returns a `messages` list for `tokenizer.apply_chat_template`):
  P    - example-label: answer with the label word BEFORE / NOT_BEFORE.
  QA1  - single yes/no question: "Does A happen before B?"  YES->1, NO->0.
  QA2  - sequential: ask before? and after? together; resolve to one label (paper's
         consistency trick, binary form).

Few-shot exemplars are inlined into the single user turn (robust across all chat templates).
Events are wrapped with the paper's [event1]/[event2] tags.
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
    return (f"Step A: {_E1}{text_1.strip()}{_E1C}\n"
            f"Step B: {_E2}{text_2.strip()}{_E2C}")


# ── system prompts ───────────────────────────────────────────────────────────────
_SYS_P = (
    "You are an expert at temporal ordering of the steps in a procedure. "
    "You are given two steps, A and B, from the same workflow. Decide the temporal order: "
    "answer BEFORE if Step A must happen before Step B, otherwise answer NOT_BEFORE. "
    "Respond with exactly one word: BEFORE or NOT_BEFORE."
)
_SYS_QA1 = (
    "You are an expert at temporal ordering of the steps in a procedure. "
    "Answer the yes/no question about two steps A and B. Respond with exactly one word: YES or NO."
)
_SYS_QA2 = (
    "You are an expert at temporal ordering of the steps in a procedure. "
    "You are given two steps A and B and two yes/no questions. Answer BOTH questions. "
    "Respond in exactly this format:\n1) YES or NO\n2) YES or NO"
)


# ── per-example user content ─────────────────────────────────────────────────────
def _user_P(text_1: str, text_2: str) -> str:
    return f"{_wrap(text_1, text_2)}\n\nDoes Step A happen BEFORE or NOT_BEFORE Step B?"


def _user_QA1(text_1: str, text_2: str) -> str:
    return f"{_wrap(text_1, text_2)}\n\nQuestion: Does Step A happen before Step B?"


def _user_QA2(text_1: str, text_2: str) -> str:
    return (f"{_wrap(text_1, text_2)}\n\n"
            f"1) Does Step A happen before Step B?\n"
            f"2) Does Step A happen after Step B?")


def _answer_P(label: int) -> str:
    return "BEFORE" if label == 1 else "NOT_BEFORE"


def _answer_QA1(label: int) -> str:
    return "YES" if label == 1 else "NO"


def _answer_QA2(label: int) -> str:
    # before? / after?  ->  label 1: YES / NO ; label 0: NO / YES
    return ("1) YES\n2) NO" if label == 1 else "1) NO\n2) YES")


_SYS = {"P": _SYS_P, "QA1": _SYS_QA1, "QA2": _SYS_QA2}
_USER = {"P": _user_P, "QA1": _user_QA1, "QA2": _user_QA2}
_ANSWER = {"P": _answer_P, "QA1": _answer_QA1, "QA2": _answer_QA2}


def build_messages(prompt_type: str, text_1: str, text_2: str,
                   few_shot: list[dict] | None = None) -> list[dict]:
    """Return chat `messages` for one test example.

    few_shot: list of {"text_1","text_2","label"} exemplars, inlined into the user turn.
    """
    sys = _SYS[prompt_type]
    user_fn = _USER[prompt_type]
    ans_fn = _ANSWER[prompt_type]

    parts: list[str] = []
    if few_shot:
        parts.append("Here are some labeled examples:\n")
        for i, ex in enumerate(few_shot, 1):
            parts.append(f"### Example {i}\n{user_fn(ex['text_1'], ex['text_2'])}\n"
                         f"Answer:\n{ans_fn(int(ex['label']))}\n")
        parts.append("### Now answer this one\n")
    parts.append(user_fn(text_1, text_2))
    if few_shot:
        parts.append("Answer:")
    return [{"role": "system", "content": sys},
            {"role": "user", "content": "\n".join(parts)}]


# ── rationale / chain-of-thought mode (LLMERE-style: rationale THEN verdict) ─────
_SYS_COT = (
    "You are an expert at temporal ordering of the steps in a procedure. "
    "You are given two steps, A (=text_1) and B (=text_2), from the same workflow. "
    "Briefly explain your reasoning about their temporal order, then end your answer with a final "
    "line that is EXACTLY 'VERDICT: BEFORE' if Step A must happen before Step B, or "
    "'VERDICT: NOT_BEFORE' otherwise."
)


def build_cot_messages(text_1: str, text_2: str, few_shot: list[dict] | None = None) -> list[dict]:
    """Rationale-then-verdict prompt. few_shot exemplars carry a `reasoning` field as the target."""
    parts: list[str] = []
    if few_shot:
        parts.append("Here are some worked examples:\n")
        for i, ex in enumerate(few_shot, 1):
            parts.append(f"### Example {i}\n{_wrap(ex['text_1'], ex['text_2'])}\n"
                         f"Reasoning:\n{ex.get('reasoning','').strip()}\n")
        parts.append("### Now do this one\n")
    parts.append(f"{_wrap(text_1, text_2)}\n\nExplain briefly, then give the VERDICT line.")
    return [{"role": "system", "content": _SYS_COT}, {"role": "user", "content": "\n".join(parts)}]


_VERDICT_RE = re.compile(r"VERDICT\s*:?\s*(NOT[_\s]?BEFORE|BEFORE|AFTER)", re.I)


def parse_verdict(gen: str) -> int | None:
    """Extract the final VERDICT from a rationale generation -> label {1,0} or None.

    Prefers an explicit 'VERDICT:' line (last one wins); falls back to a trailing BEFORE/NOT_BEFORE.
    """
    matches = _VERDICT_RE.findall(gen)
    if matches:
        v = matches[-1].upper().replace(" ", "_")
        if v.startswith("NOT") or v == "AFTER":
            return 0
        if v == "BEFORE":
            return 1
    # fallback: look at the tail for a bare label
    tail = gen[-60:].lower()
    if re.search(r"not[_\s]?before|after", tail):
        return 0
    if re.search(r"\bbefore\b", tail):
        return 1
    return None


# ── answer parsing: generated text -> predicted label (0/1) or None ──────────────
_YES = re.compile(r"\b(yes|true|before)\b", re.I)
_NO = re.compile(r"\b(no|false|not[_\s]?before|after)\b", re.I)


def _yn(token: str) -> int | None:
    t = token.strip().lower()
    has_yes = bool(re.search(r"\byes\b", t))
    has_no = bool(re.search(r"\bno\b", t))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return None


def parse_answer(prompt_type: str, gen: str) -> int | None:
    """Map model generation to label {1,0} or None (unparseable / contradictory)."""
    g = gen.strip()
    if prompt_type == "P":
        low = g.lower()
        # check NOT_BEFORE before BEFORE (substring)
        m_nb = re.search(r"not[_\s]?before", low)
        m_b = re.search(r"\bbefore\b", low)
        if m_nb:
            return 0
        if m_b:
            return 1
        if re.search(r"\bafter\b", low):
            return 0
        return None
    if prompt_type == "QA1":
        return _yn(g)
    if prompt_type == "QA2":
        # parse "1) YES/NO" and "2) YES/NO"
        a1 = re.search(r"1\s*[\).:-]?\s*(yes|no)", g, re.I)
        a2 = re.search(r"2\s*[\).:-]?\s*(yes|no)", g, re.I)
        before = a1.group(1).lower() == "yes" if a1 else None
        after = a2.group(1).lower() == "yes" if a2 else None
        if before is True and after is not True:
            return 1
        if before is False and after is True:
            return 0
        if before is not None:                       # fall back to the "before" answer
            return 1 if before else 0
        if after is not None:                        # only "after" parsed
            return 0 if after else 1
        return _yn(g)                                # last resort
    raise ValueError(prompt_type)


# ── few-shot exemplar sampling (balanced, frozen seeds) ──────────────────────────
def sample_few_shot(train_path: str | Path, k_per_class: int, seed: int) -> list[dict]:
    """Sample k_per_class BEFORE and k_per_class NOT_BEFORE exemplars from train (seeded)."""
    rng = random.Random(seed)
    pos, neg = [], []
    with open(train_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            (pos if int(r["label"]) == 1 else neg).append(
                {"text_1": r["text_1"], "text_2": r["text_2"], "label": int(r["label"])})
    rng.shuffle(pos)
    rng.shuffle(neg)
    shots = pos[:k_per_class] + neg[:k_per_class]
    rng.shuffle(shots)
    return shots
