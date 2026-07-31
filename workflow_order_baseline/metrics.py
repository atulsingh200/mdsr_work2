"""Parsing + ordering metrics for the workflow step-ordering baseline.

Metrics follow arXiv:2511.04688v2 Section 5.2, computed on the predicted vs gold permutation
(both length-n integer sequences over the shuffled-list positions):
  - accuracy:                fraction of positions placed correctly
  - kendall_tau:             rank correlation (scipy.stats.kendalltau)
  - normalized_edit_distance: Levenshtein(pred, gold) / n   (lower is better)
  - normalized_lcs:          LCS(pred, gold) / n
"""

from __future__ import annotations

import json
import re
import statistics

from scipy.stats import kendalltau


def extract_order(gen: str, shuffled_steps: list[str]) -> list[int] | None:
    """Parse the model output into a 1-indexed order permutation of range(1, n+1).

    Primary signal: the "order" field. Falls back to matching "reordered_steps" text against
    the shuffled steps. Returns None if no valid permutation can be recovered.
    """
    n = len(shuffled_steps)
    obj = _find_json(gen)

    if obj is not None and isinstance(obj.get("order"), list):
        order = _normalize_order(obj["order"], n)
        if order is not None:
            return order

    if obj is not None and isinstance(obj.get("reordered_steps"), list):
        order = _order_from_steps(obj["reordered_steps"], shuffled_steps)
        if order is not None:
            return order

    # regex fallback: pull the "order" array directly even if surrounding JSON is malformed
    # (e.g. Mistral escaping underscores as "reordered\_steps", which breaks json.loads).
    gen_clean = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL)
    m = re.search(r'"order"\s*:\s*\[([\d,\s]+)\]', gen_clean)
    if m:
        try:
            raw = [int(x) for x in m.group(1).replace(" ", "").split(",") if x != ""]
        except ValueError:
            raw = []
        order = _normalize_order(raw, n)
        if order is not None:
            return order

    return None


def _find_json(gen: str) -> dict | None:
    # strip <think>...</think> blocks (reasoning models) then find the last JSON object.
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL)
    candidates = re.findall(r"\{.*?\}", gen, flags=re.DOTALL)
    # try the greediest match first (full object), then progressively smaller ones
    for pat in (r"\{.*\}",):
        m = re.search(pat, gen, flags=re.DOTALL)
        if m:
            candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _normalize_order(raw: list, n: int) -> list[int] | None:
    try:
        order = [int(x) for x in raw]
    except (ValueError, TypeError):
        return None
    if len(order) != n:
        return None
    if sorted(order) == list(range(1, n + 1)):
        return order
    if sorted(order) == list(range(n)):  # 0-indexed -> shift to 1-indexed
        return [x + 1 for x in order]
    return None


def _order_from_steps(reordered: list, shuffled_steps: list[str]) -> list[int] | None:
    if len(reordered) != len(shuffled_steps):
        return None
    remaining = {s: i + 1 for i, s in enumerate(shuffled_steps)}
    order = []
    for s in reordered:
        if not isinstance(s, str):
            return None
        key = s.strip()
        match = next((orig for orig, pos in remaining.items() if orig.strip() == key), None)
        if match is None:
            return None
        order.append(remaining.pop(match))
    return order if len(order) == len(shuffled_steps) else None


def accuracy(pred: list[int], gold: list[int]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def kendall_tau(pred: list[int], gold: list[int]) -> float:
    if len(gold) < 2:
        return 1.0  # single-step (or empty) sequence: trivially ordered
    tau, _ = kendalltau(pred, gold)
    return 0.0 if tau != tau else float(tau)  # nan guard


def _edit_distance(a: list[int], b: list[int]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def normalized_edit_distance(pred: list[int], gold: list[int]) -> float:
    return _edit_distance(pred, gold) / len(gold)


def _lcs_len(a: list[int], b: list[int]) -> int:
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def normalized_lcs(pred: list[int], gold: list[int]) -> float:
    return _lcs_len(pred, gold) / len(gold)


def aggregate(records: list[dict]) -> dict:
    """records: list of {gold_order, pred_order (or None)}. Metrics averaged over parsed ones."""
    accs, taus, neds, nlcss = [], [], [], []
    unparsed = 0
    for r in records:
        gold = r["gold_order"]
        pred = r.get("pred_order")
        if pred is None:
            unparsed += 1
            continue
        accs.append(accuracy(pred, gold))
        taus.append(kendall_tau(pred, gold))
        neds.append(normalized_edit_distance(pred, gold))
        nlcss.append(normalized_lcs(pred, gold))
    n = len(records)
    mean = lambda xs: statistics.mean(xs) if xs else float("nan")
    return {
        "n": n, "n_parsed": n - unparsed, "unparsed": unparsed,
        "unparsed_rate": unparsed / max(n, 1),
        "acc": mean(accs), "nlcs": mean(nlcss), "ktau": mean(taus), "ned": mean(neds),
    }


if __name__ == "__main__":
    # sanity checks
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert abs(kendall_tau([3, 2, 1], [1, 2, 3]) + 1.0) < 1e-9
    assert normalized_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert normalized_lcs([1, 2, 3], [1, 2, 3]) == 1.0
    assert normalized_lcs([2, 1, 3], [1, 2, 3]) == 2 / 3
    assert extract_order('{"order": [2,1]}', ["a", "b"]) == [2, 1]
    assert extract_order('```json\n{"reordered_steps":[],"order":[0,1]}\n```', ["a", "b"]) == [1, 2]
    assert extract_order('<think>hmm</think>\n{"order":[1,3,2]}', ["a", "b", "c"]) == [1, 3, 2]
    assert extract_order('{"reordered_steps":["b","a"]}', ["a", "b"]) == [2, 1]
    assert extract_order("garbage", ["a", "b"]) is None
    print("all metric/parse sanity checks passed")
