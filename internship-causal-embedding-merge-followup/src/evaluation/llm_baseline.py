"""LLM baseline for follow-up retrieval.

For each source question, send all candidates (e.g. 300) to an LLM (Claude
Sonnet via AWS Bedrock by default) and ask it to return the top-K candidate
IDs ranked by relevance. Then compute the same retrieval metrics that
`retrieval_eval.py` uses, so the two can be compared head-to-head.

This baseline doesn't fit the `Retriever` protocol because it doesn't
produce embeddings — it directly returns a ranking. So it gets its own
small driver. Outputs are written in the same `retrieval_eval` format,
making downstream comparison straightforward.

Bedrock auth: same env vars as `call_claude_bedrock.py` (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_BEDROCK_REGION, AWS_BEDROCK_ENDPOINT_URL,
AWS_BEARER_TOKEN_BEDROCK).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from .metrics import compute_metrics


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------
def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v or v == "unconfigured":
        raise EnvironmentError(f"missing required env var: {name}")
    return v


def call_claude(
    prompt: str,
    system: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Single Bedrock invocation, returning the assistant's text content."""
    region = _env("AWS_BEDROCK_REGION")
    endpoint = _env("AWS_BEDROCK_ENDPOINT_URL")
    bearer = _env("AWS_BEARER_TOKEN_BEDROCK")
    url = f"{endpoint}/model/{model_id}/invoke"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    })

    creds = Credentials(_env("AWS_ACCESS_KEY_ID"), _env("AWS_SECRET_ACCESS_KEY"))
    aws_request = AWSRequest(method="POST", url=url, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    SigV4Auth(creds, "bedrock", region).add_auth(aws_request)
    headers = dict(aws_request.headers)
    headers["x-api-key"] = bearer

    resp = requests.post(url, headers=headers, data=body)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
def build_prompt(source_question: str, candidates: list[dict], top_k: int) -> str:
    """Build a ranking prompt that returns a JSON list of candidate IDs."""
    lines = "\n".join(f"  [{c['global_id']}] {c['text']}" for c in candidates)
    return f"""A user just asked the following question:
"{source_question}"

Below is a pool of {len(candidates)} candidate follow-up questions. Some of these
are natural follow-ups that a user would likely ask next after getting an answer
to the source question. Most are unrelated.

Your task: Select the {top_k} candidate questions that are the MOST LIKELY follow-ups
to the source question. Rank them from most likely (#1) to least likely (#{top_k}).

IMPORTANT:
- Return ONLY a JSON array of {top_k} candidate IDs in ranked order (most likely first).
- Use the numeric IDs in square brackets.
- Do not include any explanation or text outside the JSON array.

Candidate follow-up questions:
{lines}

Your top-{top_k} ranked candidate IDs (JSON array):"""


def parse_response(response: str, valid_ids: set[int], top_k: int) -> list[int]:
    """Extract an ordered list of candidate IDs from the LLM response."""
    match = re.search(r"\[[\s\d,]+\]", response)
    if match:
        try:
            ids = json.loads(match.group())
            seen: set[int] = set()
            result: list[int] = []
            for i in ids:
                i = int(i)
                if i in valid_ids and i not in seen:
                    seen.add(i)
                    result.append(i)
            return result[:top_k]
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: scrape any integers from the response
    seen = set()
    result = []
    for n in re.findall(r"\b(\d+)\b", response):
        n = int(n)
        if n in valid_ids and n not in seen:
            seen.add(n)
            result.append(n)
    return result[:top_k]


# ---------------------------------------------------------------------------
# Build (queries, candidates, gold_indices) — same shape as retrieval_eval
# ---------------------------------------------------------------------------
def build_task(path: Path, max_queries: int | None) -> tuple[list[dict], list[dict], dict]:
    """Load follow-up benchmark; return (queries-with-gold, candidates-with-id, info).

    Returns:
        queries:    list of {query_id, source_question, gold_global_ids}
        candidates: list of {global_id, query_id, text}
        info:       descriptive task metadata
    """
    data = json.load(open(path))
    all_queries = data["queries"]
    candidates: list[dict] = []
    gold_of: dict[int, list[int]] = {}
    gid = 0
    for q in all_queries:
        golds = []
        for fu in q["follow_ups"]:
            candidates.append({"global_id": gid, "query_id": q["query_id"], "text": fu["text"]})
            golds.append(gid)
            gid += 1
        gold_of[q["query_id"]] = golds

    queries_for_eval = all_queries if max_queries is None else all_queries[:max_queries]
    queries: list[dict] = []
    for q in queries_for_eval:
        queries.append({
            "query_id": q["query_id"],
            "source_question": q["source_question"],
            "topic": q.get("topic"),
            "gold_global_ids": gold_of[q["query_id"]],
        })
    info = {
        "task_type": "followup_llm_baseline",
        "source_file": str(path),
        "n_queries": len(queries),
        "n_candidates": len(candidates),
        "n_gold_per_query": 3,
    }
    return queries, candidates, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM ranker baseline for follow-up retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--followup-file", required=True,
                    help="Path to followup benchmark JSON.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: <followup>.llm_baseline.json")
    ap.add_argument("--model-id", default="us.anthropic.claude-sonnet-4-20250514-v1:0",
                    help="Bedrock model ID.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10,
                    help="How many candidates to request from the LLM. "
                         "Use the maximum K you want metrics for.")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="Seconds to sleep between requests (rate-limit guard).")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="Per-query candidate-shuffling seed (avoid positional bias).")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Truncate to first N queries (smoke test).")
    ap.add_argument("--k", action="append", type=int, default=None,
                    help="Metric cut-off (repeatable, default 1,3,5,10).")
    args = ap.parse_args()

    path = Path(args.followup_file)
    queries, candidates, info = build_task(path, args.max_queries)
    k_values = sorted(args.k or [1, 3, 5, 10])
    valid_ids = {c["global_id"] for c in candidates}

    print(f"[task] {info['task_type']}  queries={info['n_queries']}  candidates={info['n_candidates']}")
    print(f"[model] {args.model_id}")

    per_query: list[dict] = []
    similarity: list[list[float]] = []     # built as ranked-then-converted-to-pseudo-sim matrix
    gold_index_lists: list[list[int]] = []

    for i, q in enumerate(queries):
        shuffled = candidates.copy()
        random.Random(args.shuffle_seed + q["query_id"]).shuffle(shuffled)
        prompt = build_prompt(q["source_question"], shuffled, args.top_k)
        try:
            t0 = time.time()
            response = call_claude(
                prompt=prompt,
                system="You are an expert in Adobe Experience Platform and Adobe Journey Optimizer. Respond only with a JSON array of IDs.",
                model_id=args.model_id,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            elapsed = time.time() - t0
            ranked = parse_response(response, valid_ids, top_k=args.top_k)
        except Exception as e:
            print(f"  [{i+1}/{len(queries)}] query_id={q['query_id']}: ERROR {e}")
            ranked = []
            elapsed = -1.0

        # Convert ranked list into a synthetic similarity row over the candidate
        # pool so the same metric function works for everything: items in the
        # ranked list get descending scores; unranked items get -inf.
        row = [float("-inf")] * len(candidates)
        for rank, gid in enumerate(ranked):
            row[gid] = -float(rank)  # higher = better
        similarity.append(row)
        gold_index_lists.append(q["gold_global_ids"])

        hits = len(set(ranked[: max(k_values)]) & set(q["gold_global_ids"]))
        print(
            f"  [{i+1}/{len(queries)}] query_id={q['query_id']}: "
            f"hits@{max(k_values)}={hits}/{len(q['gold_global_ids'])}  ({elapsed:.1f}s)"
        )
        per_query.append({
            "query_id": q["query_id"],
            "topic": q.get("topic"),
            "ranked_ids": ranked,
            "gold_global_ids": q["gold_global_ids"],
            "elapsed_s": round(elapsed, 2),
        })

        if args.delay > 0:
            time.sleep(args.delay)

    # Compute aggregate metrics over the synthetic similarity matrix.
    import numpy as np

    sim_arr = np.array(similarity, dtype=np.float64)
    metrics = compute_metrics(sim_arr, gold_index_lists, k_values=k_values)

    print()
    print("=" * 60)
    print("LLM baseline aggregate")
    print("=" * 60)
    print(f"  MRR: {metrics['mrr']:.4f}")
    for k in k_values:
        print(f"  Recall@{k:<3}{metrics['recall_at_k'][k]:.4f}    Hit@{k:<3}{metrics['hit_at_k'][k]:.4f}")

    output = {
        "task": info,
        "model_id": args.model_id,
        "temperature": args.temperature,
        "top_k_requested": args.top_k,
        "k_values": k_values,
        "aggregate": metrics,
        "per_query": per_query,
    }
    out_path = Path(args.out) if args.out else path.with_suffix(".llm_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[done] results saved to {out_path}")


if __name__ == "__main__":
    main()
