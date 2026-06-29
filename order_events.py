#!/usr/bin/env python3
"""Derive a linear event ordering from directional pair scores.

Input: a JSON file (list of pairs) as produced by score_milan.py, where each row
has  label1, label2, text1, text2, score  and

    score = P(text1 causally precedes text2)

For every unordered pair {i, j} we compare P(i->j) vs P(j->i): the larger one
decides which event comes earlier. We rank events by:
  1. tournament wins  (how many other events this one precedes), then
  2. net precedence    sum_j [ P(i->j) - P(j->i) ]  as a tiebreak.

Usage:
  python3 order_events.py [scored.json]
    # default input: /mnt/localssd/test_samples_milan_scored.json

  python3 order_events.py scored.json --score-key score
    # use a different probability column (e.g. score_semantic)
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def load_matrix(rows: list[dict], score_key: str):
    """Return (labels, P) where P[i][j] = P(i precedes j)."""
    labels: list[str] = []
    seen = set()
    for r in rows:
        for lab in (r["label1"], r["label2"]):
            if lab not in seen:
                seen.add(lab)
                labels.append(lab)
    # natural sort (t1, t2, ... t10) for stable display
    def _key(l):
        return (l[0], int(l[1:])) if l[1:].isdigit() else (l, 0)
    labels.sort(key=_key)

    P = {a: {b: None for b in labels} for a in labels}
    for r in rows:
        P[r["label1"]][r["label2"]] = float(r[score_key])
    return labels, P


def order_events(labels: list[str], P: dict):
    wins = {a: 0 for a in labels}
    net = {a: 0.0 for a in labels}

    # net precedence over every directed pair we have a score for
    for i in labels:
        for j in labels:
            if i == j:
                continue
            pij, pji = P[i].get(j), P[j].get(i)
            if pij is not None and pji is not None:
                net[i] += pij - pji

    # pairwise tournament: earlier event = the higher-prob direction
    for i, j in combinations(labels, 2):
        pij = P[i].get(j)
        pji = P[j].get(i)
        if pij is None and pji is None:
            continue
        pij = pij if pij is not None else 0.0
        pji = pji if pji is not None else 0.0
        if pij >= pji:
            wins[i] += 1
        else:
            wins[j] += 1

    order = sorted(labels, key=lambda a: (-wins[a], -net[a]))
    return order, wins, net


def main() -> None:
    ap = argparse.ArgumentParser(description="Order events from directional pair scores.")
    ap.add_argument("input", nargs="?",
                    default="/mnt/localssd/test_samples_milan_scored_crossencoder.json",
                    help="Scored pairs JSON (list of {label1,label2,...,score}).")
    ap.add_argument("--score-key", default="score",
                    help="Which probability field to use (default: score).")
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text())
    labels, P = load_matrix(rows, args.score_key)
    order, wins, net = order_events(labels, P)

    # map label -> a short snippet of its text for readability
    text_of = {}
    for r in rows:
        text_of.setdefault(r["label1"], r["text1"])
        text_of.setdefault(r["label2"], r["text2"])

    print(f"input      : {args.input}")
    print(f"score key  : {args.score_key}")
    print(f"events     : {len(labels)}\n")

    print(f"{'event':6s} {'tourn_wins':>11s} {'net_precede':>12s}")
    for a in labels:
        print(f"{a:6s} {wins[a]:>11d} {net[a]:>12.3f}")

    print("\nORDER (first -> last):")
    print("  " + " -> ".join(order) + "\n")

    for rank, a in enumerate(order, 1):
        snippet = text_of.get(a, "").strip().replace("\n", " ")
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        print(f"  {rank}. {a}: {snippet}")


if __name__ == "__main__":
    main()
