"""followup_rerank.py -- lightweight, model-free follow-up question reranker (DEMO).

Stage-2 reranking task (research 01, task 2): given a question, rank candidate
follow-up queries by relevance / causal "next-step" likelihood. This is a CPU-light,
dependency-light demo -- NO transformer / torch (which would risk a CPU OOM crash). It
uses a pluggable score function so the survey can swap in a real cross-encoder later.

Scorers provided
-----------------
- ``TfidfScorer``        : TF-IDF cosine similarity (sklearn TfidfVectorizer) -- the
                           BM25-style lexical baseline.
- ``CausalNextStepScorer``: TF-IDF similarity + a lightweight "causal next-step"
                           heuristic that up-weights candidates phrased as actionable
                           next steps ("how do I", "what happens after", troubleshooting,
                           configuration verbs) and down-weights pure definitional repeats
                           of the question -- i.e. prefers the *enabling next action*.

Operates on /mnt/localssd/z_followupq/prod_jan_feb_26-aep-ajo_follow-up-queries.json
(list of {"question", "follow_up_queries": [[candidate, ...]]}). Produces ranked lists.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA = "/mnt/localssd/z_followupq/prod_jan_feb_26-aep-ajo_follow-up-queries.json"

# cue phrases that signal an actionable "next step" follow-up (causal next action)
_NEXT_STEP_CUES = [
    "how do i", "how can i", "how to", "what happens", "after", "next",
    "then", "troubleshoot", "why", "configure", "set up", "setup", "enable",
    "apply", "create", "resume", "during", "once", "when i",
]
# cue phrases signalling a definitional repeat (less useful as a *follow-up*)
_DEFINITIONAL_CUES = ["what is", "what are", "define", "overview of", "meaning of"]


class TfidfScorer:
    """TF-IDF cosine-similarity scorer (lexical BM25-style baseline)."""

    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = ngram_range

    def __call__(self, question: str, candidates: Sequence[str]) -> np.ndarray:
        corpus = [question] + list(candidates)
        vec = TfidfVectorizer(ngram_range=self.ngram_range, stop_words="english")
        X = vec.fit_transform(corpus)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()
        return sims


class CausalNextStepScorer:
    """TF-IDF relevance + a causal 'next-step' heuristic.

    score = sim + alpha * next_step_bonus - beta * definitional_penalty
    where next_step_bonus rewards actionable/temporal phrasing and definitional_penalty
    discourages candidates that merely restate the question's definitional intent.
    """

    def __init__(self, alpha: float = 0.15, beta: float = 0.10, ngram_range=(1, 2)):
        self.alpha = alpha
        self.beta = beta
        self.tfidf = TfidfScorer(ngram_range=ngram_range)

    @staticmethod
    def _cue_score(text: str, cues: Sequence[str]) -> float:
        t = text.lower()
        return float(sum(1 for c in cues if c in t))

    def __call__(self, question: str, candidates: Sequence[str]) -> np.ndarray:
        sims = self.tfidf(question, candidates)
        bonus = np.array([self._cue_score(c, _NEXT_STEP_CUES) for c in candidates])
        penalty = np.array([self._cue_score(c, _DEFINITIONAL_CUES) for c in candidates])
        # normalise cue counts to [0,1] so weights are interpretable
        if bonus.max() > 0:
            bonus = bonus / bonus.max()
        return sims + self.alpha * bonus - self.beta * penalty


def rerank(question: str, candidates: Sequence[str],
           scorer: Callable[[str, Sequence[str]], np.ndarray]) -> List[Dict]:
    """Rank candidates by ``scorer``; return list of {rank, score, query} desc by score."""
    scores = scorer(question, candidates)
    order = np.argsort(-scores)
    return [
        {"rank": r + 1, "score": float(scores[i]), "query": candidates[i]}
        for r, i in enumerate(order)
    ]


def rerank_dataset(path: str, scorer: Callable, limit: int | None = None) -> List[Dict]:
    """Rerank every (question, candidate-list) in the follow-up dataset."""
    data = json.load(open(path))
    if limit is not None:
        data = data[:limit]
    out = []
    for entry in data:
        question = entry["question"]
        for cand_list in entry["follow_up_queries"]:
            ranked = rerank(question, cand_list, scorer)
            out.append({"question": question, "ranked": ranked})
    return out


if __name__ == "__main__":
    print("followup_rerank.py demo\n")
    import os

    path = DEFAULT_DATA if os.path.exists(DEFAULT_DATA) else None
    if path is None:
        print(f"(data file not found at {DEFAULT_DATA}; using a tiny inline example)")
        question = "What is Journey Pause feature"
        candidates = [
            "What is a journey?",
            "How do I pause multiple journeys at once in the Journey list?",
            "How do I troubleshoot why profiles were discarded during a paused journey?",
            "What permissions are required to pause or resume a journey?",
        ]
        examples = [{"question": question, "follow_up_queries": [candidates]}]
    else:
        examples = json.load(open(path))[:2]

    tfidf = TfidfScorer()
    causal = CausalNextStepScorer()

    for entry in examples:
        q = entry["question"]
        cands = entry["follow_up_queries"][0]
        print("=" * 70)
        print(f"QUESTION: {q}\n")
        for name, scorer in [("TF-IDF baseline", tfidf), ("Causal next-step", causal)]:
            ranked = rerank(q, cands, scorer)
            print(f"  [{name}] top-3:")
            for item in ranked[:3]:
                print(f"    {item['rank']}. ({item['score']:.3f}) {item['query']}")
            print()

    if path:
        all_ranked = rerank_dataset(path, causal)
        print(f"reranked {len(all_ranked)} (question, candidate-list) groups from dataset")
